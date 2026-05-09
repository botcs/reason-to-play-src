# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""Sliding-window feature extraction for human-replay sessions.

For each ``(subject, game)`` session produced by ``run_replay.py``, this
module:

1. Builds the full rolling conversation once.  Human replays are
   concatenated into one conversation per subject x game; every later
   play's messages contain the earlier plays as a strict prefix, so the
   final play's messages is a superset of everything.
2. Tokenises that conversation once with ``apply_chat_template`` and
   records per-message token offsets.
3. Plans overlapping windows sized at ``window_fraction`` of the model's
   native max context, snapped to message boundaries so no turn is cut
   mid-way.
4. Assigns each assistant-turn target position (the end-of-turn token)
   to exactly one window -- the one where it falls in the back half, or
   to W0 if it lands before any back half.
5. Forwards each window independently with ``use_cache=False`` and a
   multi-position forward hook to capture hidden states at the targets.
6. Saves a single stacked ``.pt`` file per session with three per-layer
   feature streams (``hidden`` post-residual, ``attn`` pre-residual
   attention sublayer, ``mlp`` pre-residual MLP sublayer):

       {
           "features":      Tensor(n_targets, n_layers, hidden_dim) bfloat16,
           "features_attn": Tensor(n_targets, n_layers, hidden_dim) bfloat16,
           "features_mlp":  Tensor(n_targets, n_layers, hidden_dim) bfloat16,
           "metadata":      list[dict] per target,
           "windows":       list[dict] per window,
           "session":       dict,
           "provenance":    dict,
       }

No KV-cache threading across windows, no temporal subsampling, no
backwards-compat with the legacy per-level ``.pt`` layout -- this replaces
it.  See ``sliding_window_extraction_plan.md`` for the design.
"""

import gzip
import json
import os
import subprocess
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Ensure repo root is on the path for src.* imports when invoked as
# ``python -m src.llm_eval.human_replay.extract_features`` from any cwd.
_repo_path = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/"
if _repo_path not in sys.path:
    sys.path.insert(0, _repo_path)

from src.llm_eval.human_replay.feature_saver import sanitize_model_id  # noqa: E402
from src.llm_eval.human_replay.prompt_utils import _mt_user_content  # noqa: E402
from src.llm_eval.shared.config import ExtractionConfig  # noqa: E402
from src.llm_eval.shared.replay_codec import load_replay as _load_replay_gz  # noqa: E402


def _model_id(cfg: ExtractionConfig) -> str:
    mid = sanitize_model_id(cfg.model)
    if cfg.ablation_tag:
        mid += f"__ABLATION__{cfg.ablation_tag}"
    return mid


WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_MAIN = RANK == 0


# -- Provenance + logging helpers ---------------------------------------


def _get_provenance() -> dict:
    """Collect git hash/dirty + command + cwd for traceability."""
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    git_hash = r.stdout.strip()
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    git_dirty = len(r.stdout.strip()) > 0
    return {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "command": sys.argv,
        "cwd": os.getcwd(),
        "world_size": WORLD_SIZE,
    }


def _print(*args, **kwargs) -> None:
    if IS_MAIN:
        print(*args, **kwargs, flush=True)


# -- Session loading + conversation rebuild -----------------------------


def _load_session(prompts_file: Path) -> dict:
    """Load a ``.replay.json.gz`` session file."""
    return _load_replay_gz(prompts_file)


def _real_steps(all_steps: list[dict]) -> list[dict]:
    """Drop synthetic marker steps (e.g. ``_level_advance`` bookkeeping)."""
    return [s for s in all_steps if not s["action"].startswith("_")]


def _compress_consecutive_actions(real_steps: list[dict]) -> list[dict]:
    """Collapse runs of identical consecutive actions within a trial.

    Keeps the first step of each ``(level, attempt, action)`` run.  The
    ``(level, attempt)`` guard ensures trial boundaries always survive
    the compression -- a new trial that happens to start with the same
    action as the previous trial ended on is still retained, so
    ``_mt_user_content`` still detects the transition.
    """
    if not real_steps:
        return []
    kept: list[dict] = [real_steps[0]]
    for step in real_steps[1:]:
        last = kept[-1]
        same_trial = (
            step["level"] == last["level"] and step["attempt"] == last["attempt"]
        )
        if not same_trial or step["action"] != last["action"]:
            kept.append(step)
    return kept


def _build_conversation(
    system_prompt: str, real_steps: list[dict], rationale_mode: str
) -> list[dict]:
    """Rebuild the messages list for a session from its real steps.

    Mirrors ``prompt_utils.reconstruct_messages`` -- system prompt then
    interleaved user/assistant turns with per-step trial-transition
    markers on the user side.

    ``rationale_mode`` controls the assistant JSON shape (``action-only``
    vs. full ``{"rationale", "action"}``).
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for i, step in enumerate(real_steps):
        user_content = _mt_user_content(real_steps, i)
        messages.append({"role": "user", "content": user_content})
        response = step["response"]
        if rationale_mode == "action-only":
            assistant_content = json.dumps({"action": response["action"]})
        else:
            assistant_content = json.dumps(response)
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


# -- Tokenisation + per-message boundaries ------------------------------


# -- Per-template char-level boundary detection -------------------------
#
# Supported templates:
#
#   ChatML (Qwen3.5, Mistral, most new OSS models): every message begins
#     with ``<|im_start|>{role}\n...``.  Marker count == message count.
#
#   DeepSeek V3.x: system message is unmarked (directly after BOS);
#     user messages start with ``<｜User｜>``; assistant messages start
#     with ``<｜Assistant｜>``.  Number of ``<｜User｜>`` + number of
#     ``<｜Assistant｜>`` markers == (len(messages) - 1) for a single-
#     system conversation.
#
# Any other template (Llama3, Gemma, ...) raises RuntimeError rather
# than silently mis-aligning targets.
_CHATML_MSG_MARKER = "<|im_start|>"
_DEEPSEEK_USER_MARKER = "<｜User｜>"
_DEEPSEEK_ASSISTANT_MARKER = "<｜Assistant｜>"


def _find_all(haystack: str, needle: str) -> list[int]:
    """All non-overlapping char start positions of ``needle`` in ``haystack``."""
    out: list[int] = []
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx == -1:
            return out
        out.append(idx)
        pos = idx + 1


def _chatml_char_starts(full_text: str, messages: list[dict]) -> list[int]:
    """ChatML: every message begins with ``<|im_start|>``."""
    starts = _find_all(full_text, _CHATML_MSG_MARKER)
    if len(starts) != len(messages):
        raise RuntimeError(
            f"ChatML boundary scan found {len(starts)} {_CHATML_MSG_MARKER!r} "
            f"markers but expected {len(messages)}."
        )
    return starts


def _deepseek_char_starts(full_text: str, messages: list[dict]) -> list[int]:
    """DeepSeek V3 template boundaries.

    The system message starts at char 0 (right after BOS).  Each
    subsequent message begins at its role marker: ``<｜User｜>`` for
    user, ``<｜Assistant｜>`` for assistant.  Walk the messages list
    in order and consume markers left-to-right.

    Raises if the template's marker sequence disagrees with the
    messages' role sequence (e.g. two consecutive assistants, or a
    missing marker).
    """
    user_positions = _find_all(full_text, _DEEPSEEK_USER_MARKER)
    asst_positions = _find_all(full_text, _DEEPSEEK_ASSISTANT_MARKER)

    n_user = sum(1 for m in messages if m["role"] == "user")
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    if len(user_positions) != n_user:
        raise RuntimeError(
            f"DeepSeek boundary scan: found {len(user_positions)} "
            f"{_DEEPSEEK_USER_MARKER!r} markers but {n_user} user messages."
        )
    if len(asst_positions) != n_asst:
        raise RuntimeError(
            f"DeepSeek boundary scan: found {len(asst_positions)} "
            f"{_DEEPSEEK_ASSISTANT_MARKER!r} markers but {n_asst} assistant "
            f"messages."
        )

    if messages[0]["role"] != "system":
        raise RuntimeError(
            "DeepSeek boundary scan requires messages[0]['role'] == 'system'; "
            f"got {messages[0]['role']!r}."
        )

    starts: list[int] = [0]  # system starts at the very beginning (after BOS)
    user_idx = 0
    asst_idx = 0
    for msg in messages[1:]:
        role = msg["role"]
        if role == "user":
            starts.append(user_positions[user_idx])
            user_idx += 1
        elif role == "assistant":
            starts.append(asst_positions[asst_idx])
            asst_idx += 1
        else:
            raise RuntimeError(
                f"DeepSeek template only handles system/user/assistant; got {role!r}."
            )
    # Monotonicity: each start must be strictly greater than the previous.
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            raise RuntimeError(
                f"DeepSeek boundary scan: starts[{i}]={starts[i]} is not "
                f"strictly after starts[{i - 1}]={starts[i - 1]}. The template's "
                "marker sequence disagrees with the message role order."
            )
    return starts


def _detect_template_boundaries(full_text: str, messages: list[dict]) -> list[int]:
    """Return per-message char start positions for the template used.

    Dispatches on which marker the rendered text contains.  Both
    supported templates (ChatML, DeepSeek V3) have distinct marker
    strings with no overlap, so the dispatch is unambiguous.
    """
    has_chatml = _CHATML_MSG_MARKER in full_text
    has_deepseek = (
        _DEEPSEEK_USER_MARKER in full_text or _DEEPSEEK_ASSISTANT_MARKER in full_text
    )
    if has_chatml and has_deepseek:
        raise RuntimeError(
            "Rendered chat template contains BOTH ChatML and DeepSeek "
            "role markers; cannot unambiguously choose a boundary rule."
        )
    if has_chatml:
        return _chatml_char_starts(full_text, messages)
    if has_deepseek:
        return _deepseek_char_starts(full_text, messages)
    raise RuntimeError(
        "Rendered chat template contains no recognised role markers "
        f"({_CHATML_MSG_MARKER!r} for ChatML or {_DEEPSEEK_USER_MARKER!r} / "
        f"{_DEEPSEEK_ASSISTANT_MARKER!r} for DeepSeek V3). Supported "
        "models: Qwen3.5-family and DeepSeek V3.2; add a new boundary "
        "detector for other templates."
    )


def _compute_message_boundaries(
    tokenizer, messages: list[dict]
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Tokenise the conversation and find per-message token boundaries.

    Strategy:
      - Canonical token sequence: ``apply_chat_template(tokenize=True,
        return_tensors='pt', return_dict=False)`` -- forces a 2-D
        ``LongTensor`` regardless of backend quirks and matches what
        the model sees at inference time.
      - Per-message char boundaries: scan the template's string output
        for role markers.  ChatML templates (Qwen) put
        ``<|im_start|>`` at the start of every message; DeepSeek V3.x
        uses ``<｜User｜>`` / ``<｜Assistant｜>`` for non-system turns
        and places the system content right after BOS with no marker.
      - Char->token mapping via the fast tokenizer's ``offset_mapping``.

    Returns:
        full_ids:   1-D long tensor of the full conversation tokens.
        msg_starts: per-message token start offsets (len == len(messages)).
        msg_ends:   per-message exclusive end offsets (last == len(full_ids)).

    Raises:
        RuntimeError: if the tokenizer's template is neither ChatML nor
            DeepSeek V3.  Other templates need their own boundary
            detector -- silent mis-alignment would be worse.
    """
    # Pin the canonical token sequence to a 2-D LongTensor.
    pt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=False,
    )
    if pt_ids.ndim != 2 or pt_ids.shape[0] != 1:
        raise RuntimeError(
            "apply_chat_template(return_tensors='pt', return_dict=False) returned "
            f"unexpected shape {tuple(pt_ids.shape)}; expected (1, seq_len)."
        )
    canonical_ids: list[int] = pt_ids[0].tolist()

    full_text: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    enc_ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]

    # Reconcile: some templates emit a leading BOS only via tokenize=True.
    # DeepSeek's template inlines ``{{ bos_token }}`` into the string so both
    # paths already include it; ChatML tokenisers add BOS only in tokenize=True
    # mode.  Handle both.
    if canonical_ids == enc_ids:
        pass
    elif len(canonical_ids) == len(enc_ids) + 1 and canonical_ids[1:] == enc_ids:
        offsets = [(0, 0)] + offsets
    else:
        raise RuntimeError(
            "Tokenizer skew: apply_chat_template(tokenize=True) "
            f"({len(canonical_ids)} tokens) and "
            "tokenizer(apply_chat_template(tokenize=False), add_special_tokens=False) "
            f"({len(enc_ids)} tokens) disagree beyond a single leading BOS. "
            "Downstream alignment would be wrong; refusing to continue."
        )

    msg_char_starts = _detect_template_boundaries(full_text, messages)
    char_ends_exclusive = msg_char_starts[1:] + [len(full_text)]

    msg_ends: list[int] = []
    tok_idx = 0
    for char_end in char_ends_exclusive:
        while tok_idx < len(offsets) and offsets[tok_idx][1] <= char_end:
            tok_idx += 1
        msg_ends.append(tok_idx)

    if msg_ends[-1] != len(canonical_ids):
        raise RuntimeError(
            f"Per-message boundary derivation ended at {msg_ends[-1]} "
            f"but full tokenisation has {len(canonical_ids)} tokens. "
            "Aborting rather than producing mis-aligned targets."
        )

    msg_starts = [0] + msg_ends[:-1]
    return torch.tensor(canonical_ids, dtype=torch.long), msg_starts, msg_ends


# -- Window planning + target assignment --------------------------------


@dataclass
class Window:
    start: int  # inclusive token offset
    end: int  # exclusive token offset

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def back_half_start(self) -> int:
        """The token offset where the window's back half begins.

        Equivalently, ``front_half_end`` in the plan's terminology.
        Floor-divide keeps off-by-one at bay for odd-length windows.
        """
        return self.start + self.length // 2


def _plan_windows(
    total_tokens: int,
    window_tokens: int,
    stride_tokens: int,
    msg_starts_sorted: list[int],
) -> list[Window]:
    """Plan overlapping windows snapped to message boundaries.

    W0 starts at token 0.  ``W_(k+1).start`` = max message-start offset
    ``<= W_k.start + stride_tokens``.  Stop once the last window reaches
    ``total_tokens``.
    """
    if total_tokens <= 0:
        return []
    if stride_tokens <= 0:
        raise ValueError(
            f"stride_tokens={stride_tokens} must be positive; "
            "check window_fraction and overlap."
        )

    windows: list[Window] = []
    start = 0
    while True:
        end = min(start + window_tokens, total_tokens)
        windows.append(Window(start=start, end=end))
        if end >= total_tokens:
            break

        ideal = start + stride_tokens
        idx = bisect_right(msg_starts_sorted, ideal) - 1
        next_start = msg_starts_sorted[idx] if idx >= 0 else 0
        if next_start <= start:
            # No message boundary in (start, ideal]; advance to the next
            # boundary strictly after ``start`` to guarantee progress.
            idx2 = bisect_right(msg_starts_sorted, start)
            if idx2 >= len(msg_starts_sorted):
                # No more boundaries -- rare; force-advance by one token
                # so the loop doesn't get stuck.  Next window will be the
                # final tail.
                next_start = min(start + 1, total_tokens - 1)
            else:
                next_start = msg_starts_sorted[idx2]
        start = next_start

    return windows


def _assign_targets_to_windows(
    target_positions: list[int], windows: list[Window]
) -> list[int]:
    """Assign each target position to exactly one window.

    Rule (per ``sliding_window_extraction_plan.md`` point 2c):
      - If T < W0.back_half_start: assign to W0 (no earlier window).
      - Else: the first W_k where T in [W_k.back_half_start, W_k.end).

    With 50% overlap and uniform windows the back halves partition the
    token axis, so "first match" is unique.  With snapped boundaries
    the back halves may overlap slightly; first-match still satisfies
    "each T is assigned to exactly one window".
    """
    if not windows:
        raise ValueError("Cannot assign targets: no windows")

    assignments: list[int] = []
    for T in target_positions:
        if T < windows[0].back_half_start:
            assignments.append(0)
            continue
        picked: int | None = None
        for k, w in enumerate(windows):
            if w.back_half_start <= T < w.end:
                picked = k
                break
        if picked is None:
            # Edge: T in the front half of the last window with no earlier
            # back half catching it (can happen when a sparse-boundary
            # region forced the stride large).  Fall back to the latest
            # window that *contains* T.
            for k in range(len(windows) - 1, -1, -1):
                if windows[k].start <= T < windows[k].end:
                    picked = k
                    break
        if picked is None:
            raise ValueError(
                f"Target position T={T} lies outside every window's span "
                f"[{windows[0].start}, {windows[-1].end})."
            )
        assignments.append(picked)
    return assignments


# -- Forward hook: multiple positions per forward -----------------------


class _WindowHookExtractor:
    """Capture three per-layer feature streams at N positions per forward pass.

    For every decoder layer we capture three streams at each target position:

      - ``hidden``: the layer's post-residual output (the residual-stream
        activation flowing into the next layer).
      - ``attn``: the pre-residual attention sublayer output (``self_attn``
        or ``linear_attn`` on hybrid-attn models).
      - ``mlp``: the pre-residual MLP sublayer output.

    Hook capture is gather-at-target only -- memory per layer-stream is
    ``(n_targets, hidden_dim)``, bounded by the decision-point count in
    one window, not the window length.
    """

    _STREAMS = ("hidden", "attn", "mlp")

    def __init__(self, model, num_layers: int, hidden_dim: int, primary_device: str):
        self.model = model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.primary_device = primary_device
        self.target_positions: torch.Tensor | None = None
        self.stream_outputs: dict[str, list[torch.Tensor | None]] = {
            s: [] for s in self._STREAMS
        }
        self.hook_handles: list = []

    def register_hooks(self) -> None:
        self.remove_hooks()
        base = self.model.model if hasattr(self.model, "model") else self.model
        for i, layer in enumerate(base.layers):
            # Post-residual (full layer output) -- the residual stream.
            self.hook_handles.append(
                layer.register_forward_hook(self._make_hook(i, "hidden"))
            )
            # Attention sublayer: ``self_attn`` on standard decoder blocks,
            # ``linear_attn`` on hybrid-attention Qwen3 models.
            if hasattr(layer, "self_attn"):
                attn_mod = layer.self_attn
            elif hasattr(layer, "linear_attn"):
                attn_mod = layer.linear_attn
            else:
                raise AttributeError(
                    f"Layer {i} has no attention submodule; expected "
                    f"``self_attn`` or ``linear_attn``, found "
                    f"{[n for n, _ in layer.named_children()]}."
                )
            self.hook_handles.append(
                attn_mod.register_forward_hook(self._make_hook(i, "attn"))
            )
            if not hasattr(layer, "mlp"):
                raise AttributeError(
                    f"Layer {i} has no ``mlp`` submodule; found "
                    f"{[n for n, _ in layer.named_children()]}."
                )
            self.hook_handles.append(
                layer.mlp.register_forward_hook(self._make_hook(i, "mlp"))
            )

    def _make_hook(self, layer_idx: int, stream: str):
        def hook(_module, _inputs, output):
            if self.target_positions is None:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            # tensor: (batch=1, seq, hidden)
            positions = self.target_positions.to(tensor.device)
            extracted = tensor[0, positions, :]  # (n_targets, hidden)
            self.stream_outputs[stream][layer_idx] = extracted.to(
                dtype=torch.bfloat16, device=self.primary_device
            )

        return hook

    def remove_hooks(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def prepare(self, positions: list[int]) -> None:
        self.target_positions = torch.tensor(
            positions, dtype=torch.long, device=self.primary_device
        )
        for stream in self._STREAMS:
            self.stream_outputs[stream] = [None] * self.num_layers

    def get_results(self) -> dict[str, torch.Tensor]:
        """Return ``{stream: (n_targets, num_layers, hidden) CPU bf16}``."""
        out: dict[str, torch.Tensor] = {}
        for stream in self._STREAMS:
            slots = self.stream_outputs[stream]
            if any(x is None for x in slots):
                missing = [i for i, x in enumerate(slots) if x is None]
                raise RuntimeError(
                    f"Stream {stream!r} hooks did not fire for layers {missing}. "
                    "Forward path diverged from the expected decoder-layer sequence."
                )
            stacked = torch.stack(slots, dim=0).permute(1, 0, 2)
            out[stream] = stacked.to("cpu", non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out


# -- Model loading ------------------------------------------------------


def _load_runtime(cfg: ExtractionConfig):
    """Load HF model + tokenizer, register hooks, then compile.

    Returns ``(model, tokenizer, hook)`` where ``hook`` is a
    ``_WindowHookExtractor`` already bound to ``model``'s decoder layers.

    Hook registration happens BEFORE ``torch.compile`` -- hooks attached
    after the compile wrapper may not fire inside the compiled graph
    (the same ordering rule documented in ``transformers_wrapper``).
    """
    import torch.distributed as dist
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if IS_DISTRIBUTED and not dist.is_initialized():
        dist.init_process_group("nccl")
    if IS_DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)

    # Qwen3.5 MoE has num_key_value_heads=2; HF's default tp_plan colwise-shards
    # k_proj/v_proj which crashes in attention.forward at view(-1, head_dim) when
    # WORLD_SIZE > num_kv_heads. Patch tp_plan + attention forward before load.
    if IS_DISTRIBUTED:
        probe_cfg = AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
        if getattr(probe_cfg, "model_type", "") == "qwen3_5_moe":
            from src.llm_eval.shared import qwen3_5_moe_torchrun_wrapper

            qwen3_5_moe_torchrun_wrapper.apply()

    # Enable TF32 for any residual float32 matmuls (e.g. rotary / norm casts).
    # Torch's inductor warns when this is unset; harmless at "high" for our
    # bf16 feature-extraction forwards.
    # torch.set_float32_matmul_precision("high")

    dtype_arg: object
    if cfg.torch_dtype == "auto":
        dtype_arg = "auto"
    else:
        dtype_arg = getattr(torch, cfg.torch_dtype)

    _print(f"Loading {cfg.model} (dtype={cfg.torch_dtype}, world_size={WORLD_SIZE})")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
        _print("Using flash_attention_2")
    except ImportError:
        attn_impl = "sdpa"
        _print("Using sdpa (flash_attn unavailable)")

    load_kwargs: dict = dict(
        torch_dtype=dtype_arg,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    if IS_DISTRIBUTED:
        load_kwargs["tp_plan"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kwargs)
    if cfg.random_init:
        _print("random_init=True: reinitialising all parameters")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
    model.eval()

    primary_device = (
        f"cuda:{LOCAL_RANK}"
        if IS_DISTRIBUTED
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    num_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    _print(f"Model: num_layers={num_layers}, hidden_dim={hidden_dim}")

    hook = _WindowHookExtractor(model, num_layers, hidden_dim, primary_device)
    hook.register_hooks()

    if not cfg.no_compile:
        _print("Compiling model with torch.compile (mode=default)")
        model = torch.compile(model, mode="default")

    return model, tokenizer, hook


# -- Main extraction loop -----------------------------------------------


def _exp_db_table(cfg: ExtractionConfig) -> str:
    """Return the DynamoDB table name for this extraction run."""
    from src.llm_eval.neurips_exp_db.client import (
        TABLE_ABLATION,
        TABLE_FEATURE_EXTRACTION,
    )

    return TABLE_ABLATION if cfg.ablation_tag else TABLE_FEATURE_EXTRACTION


def _derive_slot_and_s3(
    cfg: ExtractionConfig, session: dict, prompts_file: Path
) -> tuple[str, str, str]:
    """Build (slot_id, s3_pt_key, s3_prompts_key) from a loaded session.

    Standard extractions mirror the slot-id format produced by
    ``populate_feature_extraction``; ablation runs use the ablation table
    with a tag-prefixed slot_id and create-on-claim semantics.
    """
    from src.llm_eval.neurips_exp_db.slot_ids import (
        slot_id_ablation,
        slot_id_feature_extraction,
    )

    subject = session.get("subject") or session.get("meta", {}).get("subject")
    game = session["game"]
    suggestion_level = session.get("meta", {}).get("suggestion_level")
    if not subject:
        raise ValueError(f"{prompts_file}: missing subject in session")
    if not suggestion_level:
        raise ValueError(
            f"{prompts_file}: missing meta.suggestion_level; older replay "
            "files predate the field. Re-run run_replay.py to regenerate, "
            "or pass exp_db.enabled=false."
        )
    prompt_config = f"suggestion-{suggestion_level}"
    if cfg.ablation_tag:
        slot_id = slot_id_ablation(
            ablation_tag=cfg.ablation_tag,
            extraction_model=cfg.model,
            prompt_config=prompt_config,
            subject=subject,
            game=game,
        )
    else:
        slot_id = slot_id_feature_extraction(
            extraction_model=cfg.model,
            prompt_config=prompt_config,
            subject=subject,
            game=game,
        )
    model_id = _model_id(cfg)
    variant_tag = "compressed" if cfg.action_compression else "all"
    prefix = cfg.exp_db.feature_s3_prefix.rstrip("/")
    base = f"{prefix}/model-{model_id}/{variant_tag}/{subject}/{game}"
    return slot_id, f"{base}.pt", f"{base}_prompts.jsonl.gz"


def _derive_slot_id_worker(args: tuple) -> tuple:
    """Process-pool worker for the ExpDB pre-check.

    Loads a single replay JSON and derives its slot_id.  Lifted to module
    level so ``ProcessPoolExecutor`` can pickle it.  Decompressing and
    parsing ~1 MB gzipped (~30 MB JSON) is a CPython-GIL-bound operation
    (~2.3 s per file on this cluster); thread pools don't help.  Forking a
    handful of workers lets the parses actually run in parallel.
    """
    idx, path_str, model, ablation_tag = args
    from src.llm_eval.neurips_exp_db.slot_ids import (
        slot_id_ablation,
        slot_id_feature_extraction,
    )

    pf = Path(path_str)
    try:
        session = _load_session(pf)
        subject = session.get("subject") or session.get("meta", {}).get("subject")
        game = session.get("game")
        suggestion_level = session.get("meta", {}).get("suggestion_level")
        if not subject:
            return idx, None, f"{pf.name}: missing subject; keeping"
        if not suggestion_level:
            return idx, None, f"{pf.name}: missing meta.suggestion_level; keeping"
        prompt_config = f"suggestion-{suggestion_level}"
        if ablation_tag:
            slot_id = slot_id_ablation(
                ablation_tag=ablation_tag,
                extraction_model=model,
                prompt_config=prompt_config,
                subject=subject,
                game=game,
            )
        else:
            slot_id = slot_id_feature_extraction(
                extraction_model=model,
                prompt_config=prompt_config,
                subject=subject,
                game=game,
            )
        return idx, slot_id, None
    except Exception as e:
        return idx, None, f"{pf.name} unparseable ({e!r}); keeping"


def _prefilter_claimable_sessions(
    prompts_files: list[Path],
    cfg: ExtractionConfig,
    exp_db_client,
) -> list[Path]:
    """Drop sessions whose ExpDB slot is not in pending state.

    The in-loop ``ExpDB.claim_slot`` already gracefully skips non-claimable
    slots, but only AFTER ``_load_runtime`` has paid the model-shard load
    and torch.compile cost (minutes for >100B models).  This pre-check
    runs before the runtime is built so an entire torchrun call returns in
    seconds when every (model, subject, game) slot is already done.

    Rank 0 queries DynamoDB; the keep-mask is broadcast over NCCL so all
    ranks filter to the same list.  No-op when ``exp_db.enabled=false``.
    """
    if not cfg.exp_db.enabled or cfg.exp_db.force_reclaim:
        return prompts_files

    import torch.distributed as dist

    if IS_DISTRIBUTED:
        if not dist.is_initialized():
            torch.cuda.set_device(LOCAL_RANK)
            dist.init_process_group("nccl")
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")

    keep = torch.ones(len(prompts_files), dtype=torch.bool, device=device)
    if IS_MAIN and exp_db_client is not None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        table = _exp_db_table(cfg)

        # Two-phase: (1) derive slot_ids from local replay JSONs in a
        # process pool (CPython JSON parsing is GIL-bound, threads don't
        # help -- 16 forked workers cut the parse phase from ~5 min to
        # ~15 s on N=102).  (2) one BatchGetItem to DynamoDB instead of N
        # point-gets, which collapses the network phase from minutes of
        # serial round-trips to ~2 s.
        slot_ids: list[str | None] = [None] * len(prompts_files)
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as ex:
            args_list = [
                (i, str(pf), cfg.model, cfg.ablation_tag)
                for i, pf in enumerate(prompts_files)
            ]
            futures = [ex.submit(_derive_slot_id_worker, a) for a in args_list]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="ExpDB derive slot_ids",
                unit="file",
            ):
                idx, sid, msg = fut.result()
                slot_ids[idx] = sid
                if msg:
                    tqdm.write(f"ExpDB pre-check: {msg}")

        ids_to_query = [sid for sid in slot_ids if sid is not None]
        _print(
            f"ExpDB pre-check: BatchGetItem for {len(ids_to_query)} slot_ids "
            f"(in chunks of 100)"
        )
        items = exp_db_client.batch_get_slots(table, ids_to_query)
        _print(f"ExpDB pre-check: BatchGetItem returned {len(items)} existing slots")

        n_skipped = 0
        for i, sid in enumerate(slot_ids):
            if sid is None:
                continue  # unparseable replay -- already kept
            item = items.get(sid)
            if item is not None and item.get("status") != "pending":
                keep[i] = False
                n_skipped += 1
                _print(f"ExpDB pre-check: skip {sid} (status={item.get('status')})")
        n_keep = int(keep.sum().item())
        _print(
            f"ExpDB pre-check: {n_keep}/{len(prompts_files)} sessions still claimable "
            f"({n_skipped} skipped)"
        )

    if IS_DISTRIBUTED:
        dist.broadcast(keep, src=0)

    return [pf for pf, k in zip(prompts_files, keep.cpu().tolist()) if k]


def extract_for_session(
    prompts_file: Path,
    cfg: ExtractionConfig,
    model,
    tokenizer,
    hook: "_WindowHookExtractor",
    wandb_run=None,
    session_idx: int = 0,
    exp_db_client=None,
    exp_db_worker: str = "",
) -> dict | None:
    """Extract sliding-window features for one ``.replay.json.gz`` session.

    The model, tokenizer, and hook extractor are provided by the caller
    and reused across sessions -- so a multi-session run pays the model
    load / compile-warmup cost once, not per session.

    When ``exp_db_client`` is provided (rank 0, ``cfg.exp_db.enabled``),
    the function claims a slot in ``vgdl-grid-feature-extraction`` before
    extraction begins, uploads the saved ``.pt`` and ``_prompts.jsonl.gz``
    to ``s3://<bucket>/<feature_s3_prefix>/...`` on success, and marks the
    slot ``done`` (or ``failed`` with a phase-tagged error message).
    Non-claimable slots (already running/done elsewhere) cause an early
    ``return None`` so the caller's batch loop can skip on to the next
    session.

    Returns a summary dict on rank 0; ``None`` on non-main ranks (which
    still participate in every forward) and on rank 0 when an opt-in slot
    was not claimable.
    """
    session = _load_session(prompts_file)
    game_name = session["game"]
    subject = session.get("subject") or session.get("meta", {}).get("subject")
    if not subject:
        raise ValueError(f"{prompts_file}: could not determine subject")
    system_prompt = session["system_prompt"]
    rationale_mode = session["meta"]["rationale_mode"]
    all_steps = session["steps"]

    real = _real_steps(all_steps)
    if cfg.action_compression:
        real = _compress_consecutive_actions(real)
    if not real:
        raise ValueError(f"{prompts_file}: no real decision steps")

    _print(
        f"Session {subject} x {game_name}: {len(real)} decision points "
        f"(action_compression={cfg.action_compression}, "
        f"rationale_mode={rationale_mode})"
    )

    # ExpDB claim -- rank 0 only, opt-in.  Outside the try below: a hard
    # claim failure (network, IAM) propagates as-is, since we have no
    # claimed slot to mark failed.  ConditionalCheckFailedException returns
    # False here and we soft-skip with a None return.
    slot_id: str | None = None
    s3_pt_key: str | None = None
    s3_prompts_key: str | None = None
    if exp_db_client is not None and IS_MAIN:
        table = _exp_db_table(cfg)
        slot_id, s3_pt_key, s3_prompts_key = _derive_slot_and_s3(
            cfg, session, prompts_file
        )
        if cfg.ablation_tag:
            exp_db_client.create_slot(
                table,
                slot_id,
                {
                    "ablation_tag": cfg.ablation_tag,
                    "extraction_model": cfg.model,
                    "subject": subject,
                    "game": game_name,
                },
            )
        if cfg.exp_db.force_reclaim:
            exp_db_client.set_status(table, slot_id, "pending")
        claimed = exp_db_client.claim_slot(
            table,
            slot_id,
            worker=exp_db_worker,
            wandb_run_id=wandb_run.id,
            wandb_run_url=wandb_run.url,
            s3_replay_key=s3_pt_key,
        )
        if not claimed:
            _print(f"ExpDB: skipped (slot not claimable: {slot_id})")
            return None
        _print(f"ExpDB: claimed {slot_id}")

    phase = "extract"
    try:
        messages = _build_conversation(system_prompt, real, rationale_mode)
        _print(f"Tokenising {len(messages)} messages...")
        t0 = time.time()
        full_ids, msg_starts, msg_ends = _compute_message_boundaries(
            tokenizer, messages
        )
        total_tokens = int(full_ids.shape[0])
        _print(f"Tokenised in {time.time() - t0:.1f}s: total_tokens={total_tokens:,}")

        # Target positions = last token of each assistant turn.
        # messages = [system, user_0, asst_0, user_1, asst_1, ...]; asst_k at idx 2 + 2k.
        target_msg_indices = [2 + 2 * k for k in range(len(real))]
        target_positions = [msg_ends[i] - 1 for i in target_msg_indices]
        for p in target_positions:
            if p < 0 or p >= total_tokens:
                raise ValueError(f"Target position {p} outside [0, {total_tokens}).")

        max_context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        if max_context <= 0:
            raise ValueError(
                f"Model config has no usable max_position_embeddings "
                f"(got {max_context})."
            )
        window_tokens = max(1, int(cfg.window_fraction * max_context))
        stride_tokens = max(1, int(window_tokens * (1.0 - cfg.overlap)))
        _print(
            f"max_context={max_context:,}  window_tokens={window_tokens:,}  "
            f"stride_tokens={stride_tokens:,}"
        )

        msg_starts_sorted = sorted(set(msg_starts))
        windows = _plan_windows(
            total_tokens, window_tokens, stride_tokens, msg_starts_sorted
        )
        _print(f"Planned {len(windows)} windows")

        boundary_set = set(msg_starts_sorted)
        for w in windows:
            if w.length > window_tokens:
                raise AssertionError(
                    f"Window length {w.length} exceeds window_tokens {window_tokens}"
                )
            if w.start not in boundary_set:
                raise AssertionError(
                    f"Window start {w.start} is not on a message boundary"
                )
        if windows[-1].end < total_tokens:
            raise AssertionError(
                f"Last window ends at {windows[-1].end} but total_tokens={total_tokens}"
            )

        assignments = _assign_targets_to_windows(target_positions, windows)
        if len(assignments) != len(target_positions):
            raise AssertionError("Assignment length mismatch")
        for i, k in enumerate(assignments):
            if not (0 <= k < len(windows)):
                raise AssertionError(f"Target {i} got invalid window idx {k}")

        primary_device = hook.primary_device
        num_layers = hook.num_layers
        hidden_dim = hook.hidden_dim

        # Group targets by window so each window's forward captures all its targets at once.
        targets_per_window: dict[int, list[int]] = defaultdict(list)
        for target_idx, w_idx in enumerate(assignments):
            targets_per_window[w_idx].append(target_idx)

        streams = ("hidden", "attn", "mlp")
        all_features: dict[str, list[torch.Tensor | None]] = {
            s: [None] * len(target_positions) for s in streams
        }

        peak_mem_mb = 0.0
        total_forward_s = 0.0
        session_t0 = time.time()

        pbar = tqdm(
            range(len(windows)),
            desc="Windows",
            disable=not IS_MAIN,
            position=0,
        )
        for w_idx in pbar:
            target_indices_here = targets_per_window.get(w_idx, [])
            if not target_indices_here:
                # No targets fall into this window's back half; skipping the
                # forward is correct -- the window only exists to keep stride
                # continuity with later windows.
                continue
            w = windows[w_idx]
            within = [target_positions[ti] - w.start for ti in target_indices_here]
            for p in within:
                if not (0 <= p < w.length):
                    raise AssertionError(
                        f"Within-window position {p} outside [0, {w.length})"
                    )

            window_ids = full_ids[w.start : w.end].unsqueeze(0).to(primary_device)
            attention_mask = torch.ones_like(window_ids)

            hook.prepare(within)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(primary_device)
            t_fwd = time.time()
            with torch.inference_mode():
                _ = model(
                    input_ids=window_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            window_forward_s = time.time() - t_fwd
            total_forward_s += window_forward_s

            stream_feats = hook.get_results()
            for stream in streams:
                tensor = stream_feats[
                    stream
                ]  # (n_targets_in_window, num_layers, hidden)
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    debug_path = Path(cfg.output_dir) / "debug_nan"
                    debug_path.mkdir(parents=True, exist_ok=True)
                    debug_file = (
                        debug_path / f"{prompts_file.stem}_w{w_idx}_{stream}.pt"
                    )
                    torch.save(
                        {
                            "stream": stream,
                            "tensor": tensor.cpu(),
                            "window_idx": w_idx,
                            "window_start": w.start,
                            "window_end": w.end,
                            "window_length": w.length,
                            "window_ids": window_ids.cpu(),
                            "within_positions": within,
                            "target_indices": target_indices_here,
                            "prompts_file": str(prompts_file),
                            "max_context": max_context,
                            "window_tokens": window_tokens,
                            "total_tokens": total_tokens,
                        },
                        debug_file,
                    )
                    raise RuntimeError(
                        f"NaN/Inf in stream {stream!r} at window {w_idx}/{len(windows)} "
                        f"(len={w.length}). Debug saved to {debug_file}"
                    )
                for local_i, target_idx in enumerate(target_indices_here):
                    all_features[stream][target_idx] = tensor[local_i]

            this_peak_mb = 0.0
            if torch.cuda.is_available():
                this_peak_mb = torch.cuda.max_memory_allocated(primary_device) / (
                    1024**2
                )
                peak_mem_mb = max(peak_mem_mb, this_peak_mb)
                # Forward peak sits ~20 GB under the 192 GB B200 ceiling for the
                # 122B model; without this, allocator fragmentation OOMs window 2.
                torch.cuda.empty_cache()
            pbar.set_postfix_str(
                f"len={w.length} n_tgt={len(within)} peak_mb={peak_mem_mb:.0f}"
            )

            if wandb_run is not None and IS_MAIN:
                import wandb

                wandb.log(
                    {
                        "window/forward_s": window_forward_s,
                        "window/peak_mem_mb": this_peak_mb,
                        "window/length_tokens": w.length,
                        "window/n_targets": len(within),
                        "window/tokens_per_sec": w.length / max(window_forward_s, 1e-9),
                        "session_idx": session_idx,
                        "window_idx": w_idx,
                    }
                )

        session_wall_s = time.time() - session_t0

        if not IS_MAIN:
            return None

        for stream in streams:
            missing = [i for i, x in enumerate(all_features[stream]) if x is None]
            if missing:
                raise RuntimeError(
                    f"Stream {stream!r}: captured no features for {len(missing)} "
                    f"targets (first: {missing[:10]})."
                )

        features_by_stream: dict[str, torch.Tensor] = {
            s: torch.stack(all_features[s], dim=0).contiguous() for s in streams
        }
        expected_shape = (len(target_positions), num_layers, hidden_dim)
        for s, t in features_by_stream.items():
            if tuple(t.shape) != expected_shape:
                raise AssertionError(
                    f"Stream {s} shape {tuple(t.shape)} != expected {expected_shape}"
                )
            if torch.isnan(t).any():
                raise RuntimeError(f"NaN in stream {s}")
            if torch.isinf(t).any():
                raise RuntimeError(f"Inf in stream {s}")
        md: list[dict] = []
        for target_idx, step in enumerate(real):
            w_idx = assignments[target_idx]
            w = windows[w_idx]
            md.append(
                {
                    "play_id": step.get("play_id"),
                    "step_num": step["step"],
                    "level_id": step["level"],
                    "attempt": step.get("attempt"),
                    "trial_idx": step.get("trial_idx"),
                    "action": step["action"],
                    "reward": step.get("reward"),
                    "won": step.get("won"),
                    "realworld_ts": step.get("realworld_ts"),
                    "global_token_offset": target_positions[target_idx],
                    "window_idx": w_idx,
                    "position_within_window": target_positions[target_idx] - w.start,
                }
            )

        model_id = _model_id(cfg)
        variant_tag = "compressed" if cfg.action_compression else "all"
        out_dir = Path(cfg.output_dir) / f"model-{model_id}" / variant_tag / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{game_name}.pt"
        prompts_out_file = out_dir / f"{game_name}_prompts.jsonl.gz"

        saved = {
            "features": features_by_stream["hidden"],
            "features_attn": features_by_stream["attn"],
            "features_mlp": features_by_stream["mlp"],
            "metadata": md,
            "windows": [
                {
                    "start": w.start,
                    "end": w.end,
                    "back_half_start": w.back_half_start,
                    "length": w.length,
                    "n_targets": len(targets_per_window.get(i, [])),
                }
                for i, w in enumerate(windows)
            ],
            "session": {
                "subject": subject,
                "game": game_name,
                "model": cfg.model,
                "model_id": model_id,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "rationale_mode": rationale_mode,
                "suggestion_level": session["meta"].get("suggestion_level"),
                "source": session.get("source"),
                "total_tokens": total_tokens,
                "num_messages": len(messages),
                "n_windows": len(windows),
                "n_targets": len(target_positions),
                "max_context": max_context,
                "window_fraction": cfg.window_fraction,
                "overlap": cfg.overlap,
                "window_tokens": window_tokens,
                "stride_tokens": stride_tokens,
                "action_compression": cfg.action_compression,
                "peak_gpu_mem_mb": peak_mem_mb,
                "total_forward_s": total_forward_s,
                "session_wall_s": session_wall_s,
            },
            "provenance": {
                **_get_provenance(),
                "model": cfg.model,
                "timestamp": datetime.now().isoformat(),
                "prompts_file": str(prompts_file),
            },
        }
        torch.save(saved, out_file)
        out_mb = out_file.stat().st_size / 1024**2
        _print(f"Saved {out_file} ({out_mb:.1f} MB)")

        with gzip.open(prompts_out_file, "wt") as f:
            for step in real:
                f.write(json.dumps(step) + "\n")
        _print(f"Saved {prompts_out_file}")

        # ExpDB upload + complete -- rank 0, opt-in.  Non-main ranks have
        # already returned at the IS_MAIN guard above.
        if exp_db_client is not None:
            bucket = cfg.exp_db.s3_bucket
            phase = "upload_pt"
            exp_db_client._s3.upload_file(str(out_file), bucket, s3_pt_key)
            _print(f"S3: uploaded s3://{bucket}/{s3_pt_key}")
            phase = "upload_prompts"
            exp_db_client._s3.upload_file(str(prompts_out_file), bucket, s3_prompts_key)
            _print(f"S3: uploaded s3://{bucket}/{s3_prompts_key}")
            phase = "complete"
            exp_db_client.complete_slot(table, slot_id, s3_replay_key=s3_pt_key)
            _print(f"ExpDB: completed {slot_id}")

        return {
            "output_file": str(out_file),
            "out_mb": out_mb,
            "n_targets": len(target_positions),
            "n_windows": len(windows),
            "total_tokens": total_tokens,
            "peak_gpu_mem_mb": peak_mem_mb,
            "total_forward_s": total_forward_s,
            "session_wall_s": session_wall_s,
            "slot_id": slot_id,
            "s3_pt_key": s3_pt_key,
        }
    except BaseException as e:
        # Mark the slot failed if we own one.  Phase tag distinguishes
        # extraction crashes from S3-upload / DDB-complete failures.
        if exp_db_client is not None and IS_MAIN and slot_id is not None:
            import traceback

            err_msg = f"[{phase}] {type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            try:
                exp_db_client.fail_slot(table, slot_id, error_msg=err_msg)
                _print(f"ExpDB: failed {slot_id} ({phase})")
            except Exception as fail_err:
                _print(f"ExpDB: fail_slot itself failed: {fail_err}")
        raise


# -- Hydra entry point --------------------------------------------------


@hydra.main(
    config_path="../../../conf/extract_features",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    schema = OmegaConf.structured(ExtractionConfig)
    merged = OmegaConf.merge(schema, cfg)
    typed_cfg: ExtractionConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    if typed_cfg.prompts in (None, "", "???"):
        raise ValueError("prompts=<path-to-.replay.json.gz> is required")
    if typed_cfg.model in (None, "", "???"):
        raise ValueError("model=<hf-model-id> is required")
    if not (0.0 < typed_cfg.window_fraction <= 1.0):
        raise ValueError(
            f"window_fraction={typed_cfg.window_fraction} must be in (0, 1]"
        )
    if not (0.0 <= typed_cfg.overlap < 1.0):
        raise ValueError(f"overlap={typed_cfg.overlap} must be in [0, 1)")

    # prompts may be a single file path or a glob pattern.  ``glob.glob``
    # on a plain non-glob path returns ``[path]`` if it exists, so one
    # code path handles both.
    import csv
    import glob as _glob

    pattern = typed_cfg.prompts
    matches = sorted(_glob.glob(pattern))
    if not matches:
        # Fall back to treating it as a literal path -- gives a clear error.
        lit = Path(pattern)
        if lit.exists():
            matches = [str(lit)]
    if not matches:
        raise FileNotFoundError(f"No files match prompts={pattern!r}")
    prompts_files = [Path(p).resolve() for p in matches]

    if IS_MAIN:
        print("=" * 60)
        print("Sliding-window feature extraction")
        print("=" * 60)
        print(f"Prompts pattern:    {pattern}")
        print(f"Sessions matched:   {len(prompts_files)}")
        for p in prompts_files:
            print(f"  - {p}")
        print(f"Model:              {typed_cfg.model}")
        print(f"Output dir:         {typed_cfg.output_dir}")
        print(f"Window fraction:    {typed_cfg.window_fraction}")
        print(f"Overlap:            {typed_cfg.overlap}")
        print(f"Action compression: {typed_cfg.action_compression}")
        print(f"torch_dtype:        {typed_cfg.torch_dtype}")
        print(f"no_compile:         {typed_cfg.no_compile}")
        print(f"World size:         {WORLD_SIZE}")
        print("=" * 60)

    # ExpDB tracking is rank-0 only.  Every claim/upload/complete call
    # below is gated on ``exp_db_client is not None``, which is itself
    # gated on IS_MAIN.  Built before the prefilter and wandb init so a
    # job whose slots are all already done can exit before paying the
    # model-load / torch.compile cost.
    exp_db_client = None
    exp_db_worker = ""
    if typed_cfg.exp_db.enabled and IS_MAIN:
        import socket

        from src.llm_eval.neurips_exp_db import ExpDB

        if not typed_cfg.wandb_project:
            raise ValueError(
                "exp_db.enabled=true requires wandb_project to be set "
                "(experiment tracking needs wandb run URLs)"
            )
        exp_db_client = ExpDB(typed_cfg.exp_db.region)
        exp_db_worker = socket.gethostname()
        _print(f"ExpDB: tracking enabled (region={typed_cfg.exp_db.region})")

    prompts_files = _prefilter_claimable_sessions(
        prompts_files, typed_cfg, exp_db_client
    )
    if not prompts_files:
        _print("ExpDB pre-check: no claimable sessions; exiting before model load.")
        return

    wandb_run = None
    if typed_cfg.wandb_project and IS_MAIN:
        import wandb

        wandb_run = wandb.init(
            project=typed_cfg.wandb_project,
            config={
                "prompts_pattern": pattern,
                "n_sessions": len(prompts_files),
                "model": typed_cfg.model,
                "window_fraction": typed_cfg.window_fraction,
                "overlap": typed_cfg.overlap,
                "action_compression": typed_cfg.action_compression,
                "torch_dtype": typed_cfg.torch_dtype,
                "no_compile": typed_cfg.no_compile,
                "world_size": WORLD_SIZE,
            },
        )

    # One model + tokenizer + hook extractor for ALL matched sessions --
    # amortises the model-load and torch.compile cost across the batch.
    load_t0 = time.time()
    model, tokenizer, hook = _load_runtime(typed_cfg)
    load_wall_s = time.time() - load_t0
    _print(f"Runtime ready in {load_wall_s:.1f}s")

    summaries: list[dict] = []
    for i, prompts_file in enumerate(prompts_files):
        _print(f"\n### [{i + 1}/{len(prompts_files)}] {prompts_file.name} ###")
        try:
            result = extract_for_session(
                prompts_file,
                typed_cfg,
                model,
                tokenizer,
                hook,
                wandb_run=wandb_run,
                session_idx=i,
                exp_db_client=exp_db_client,
                exp_db_worker=exp_db_worker,
            )
        except BaseException as e:
            # extract_for_session has already called fail_slot when
            # exp_db is enabled.  Loop-level policy: continue to the next
            # session (default) or re-raise to abort the batch.
            if typed_cfg.exp_db.continue_on_session_error:
                _print(
                    f"!!! Session {prompts_file.name} failed "
                    f"({type(e).__name__}: {e!r}) -- continuing"
                )
                continue
            raise
        if result is None:
            continue
        result = {"prompts_file": str(prompts_file), **result}
        summaries.append(result)
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    **{
                        f"session/{k}": v
                        for k, v in result.items()
                        if isinstance(v, (int, float))
                    },
                    "session_idx": i,
                }
            )

    if IS_MAIN:
        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)
        print(
            f"{'session':50s}  {'tokens':>8s}  {'tgts':>5s}  "
            f"{'wins':>4s}  {'peak_MB':>8s}  {'fwd_s':>7s}  {'wall_s':>7s}"
        )
        for s in summaries:
            print(
                f"{Path(s['prompts_file']).name:50s}  "
                f"{s['total_tokens']:>8d}  {s['n_targets']:>5d}  "
                f"{s['n_windows']:>4d}  {s['peak_gpu_mem_mb']:>8.0f}  "
                f"{s['total_forward_s']:>7.2f}  {s['session_wall_s']:>7.2f}"
            )
        total_fwd = sum(s.get("total_forward_s", 0.0) for s in summaries)
        total_wall = sum(s.get("session_wall_s", 0.0) for s in summaries)
        print("-" * 72)
        print(
            f"{'total (excl. load/compile)':50s}  {'':>8s}  {'':>5s}  "
            f"{'':>4s}  {'':>8s}  {total_fwd:>7.2f}  {total_wall:>7.2f}"
        )
        print(
            f"{'runtime load+compile wall_s':50s}  {'':>8s}  {'':>5s}  "
            f"{'':>4s}  {'':>8s}  {'':>7s}  {load_wall_s:>7.2f}"
        )
        print("=" * 72)

        # Write a per-run CSV under output_dir for downstream analysis.
        if summaries:
            variant_tag = "compressed" if typed_cfg.action_compression else "all"
            model_id = _model_id(typed_cfg)
            csv_path = (
                Path(typed_cfg.output_dir)
                / f"model-{model_id}"
                / variant_tag
                / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            cols = [
                "prompts_file",
                "output_file",
                "total_tokens",
                "n_targets",
                "n_windows",
                "peak_gpu_mem_mb",
                "total_forward_s",
                "session_wall_s",
                "out_mb",
            ]
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                writer.writeheader()
                for s in summaries:
                    writer.writerow(s)
            print(f"benchmark CSV: {csv_path}")
            print(
                f"load+compile:  {load_wall_s:.2f}s (amortised across {len(summaries)} sessions)"
            )
            if wandb_run is not None:
                import wandb

                table = wandb.Table(columns=cols)
                for s in summaries:
                    table.add_data(*[s.get(c) for c in cols])
                wandb.log({"sessions_table": table})

    if wandb_run is not None:
        import wandb

        total_fwd_sum = sum(s["total_forward_s"] for s in summaries)
        total_wall_sum = sum(s["session_wall_s"] for s in summaries)
        total_tokens_sum = sum(s["total_tokens"] for s in summaries)
        total_targets_sum = sum(s["n_targets"] for s in summaries)
        total_windows_sum = sum(s["n_windows"] for s in summaries)
        peak_mem_max = max((s["peak_gpu_mem_mb"] for s in summaries), default=0.0)
        out_mb_sum = sum(s.get("out_mb", 0.0) for s in summaries)
        wandb.log(
            {
                "runtime/load_compile_s": load_wall_s,
                "runtime/n_sessions": len(summaries),
                "runtime/total_forward_s_sum": total_fwd_sum,
                "runtime/session_wall_s_sum": total_wall_sum,
                "runtime/total_tokens_sum": total_tokens_sum,
                "runtime/total_targets_sum": total_targets_sum,
                "runtime/total_windows_sum": total_windows_sum,
                "runtime/peak_gpu_mem_mb_max": peak_mem_max,
                "runtime/out_mb_sum": out_mb_sum,
                "runtime/tokens_per_sec_avg": (
                    total_tokens_sum / total_fwd_sum if total_fwd_sum > 0 else 0.0
                ),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
