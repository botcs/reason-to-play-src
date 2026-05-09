# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
DeepSeek V4 multi-turn sliding-window feature extractor.

Standalone entry point that loads V4 via ``deepseek_v4_inference/`` (custom
Transformer + FP8/FP4 TileLang kernels, torchrun MP=N tensor parallel) and
reuses the shared extraction loop from ``extract_features.py``.

DeepSeek V4 cannot be loaded through HF ``AutoModelForCausalLM`` because
the model uses Hyper-Connections, FP4 expert weights, and custom sparse
attention kernels not supported by transformers.  This script uses DeepSeek's
official V4 inference repo (``deepseek_v4_inference/model.py``) with converted
MP=N safetensor shards and TileLang-JIT FP8/FP4 kernels.

Usage::

    torchrun --nproc-per-node 8 --standalone \\
        -m src.llm_eval.human_replay.extract_features_v4 \\
        prompts="path/to/*.replay.json.gz" \\
        model=deepseek-ai/DeepSeek-V4-Pro \\
        ckpt_path=/path/to/converted/shards \\
        ds_config=deepseek_v4_inference/vgdl_DS4.json \\
        output_dir=out/features_v4

Prerequisites:
    1. ``deepseek_v4_inference/`` folder at the repo root (from DeepSeek-V4
       official inference release).
    2. Converted checkpoint shards::
           cd deepseek_v4_inference && python convert.py \\
               --hf-ckpt-path $HF --save-path $SAVE \\
               --n-experts 384 --model-parallel $MP
    3. ``tilelang==0.1.8`` and ``fast_hadamard_transform`` installed.
    4. ``torch>=2.10.0`` (required for float4_e2m1fn_x2 dtype).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

# Ensure repo root is on the path for src.* imports.
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Add deepseek_v4_inference/ to path for its internal imports (kernel, model).
# NOTE: do NOT also add deepseek_inference/ -- both export a ``model`` module
# and having both on sys.path would cause import collisions.
_ds_v4_inference_path = str(Path(_repo_root) / "deepseek_v4_inference")
if _ds_v4_inference_path not in sys.path:
    sys.path.insert(0, _ds_v4_inference_path)

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from load_utils import load_sharded_model  # noqa: E402

from src.llm_eval.human_replay.extract_features import (  # noqa: E402
    IS_DISTRIBUTED,
    IS_MAIN,
    LOCAL_RANK,
    WORLD_SIZE,
    _WindowHookExtractor,
    _print,
    extract_for_session,
)
from src.llm_eval.human_replay.feature_saver import sanitize_model_id  # noqa: E402
from src.llm_eval.shared.config import ExtractionConfig  # noqa: E402

# DeepSeek V3/V4 chat template — fallback when the tokenizer has none.
# V4 uses the same markers as V3 (<｜User｜>, <｜Assistant｜>).
DEEPSEEK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% set ns = namespace(system_prompt='', is_first_sp=true, is_last_user=false) %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% if ns.is_first_sp %}{% set ns.system_prompt = message['content'] %}{% set ns.is_first_sp = false %}"
    "{% else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{{ bos_token }}{{ ns.system_prompt }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{% set ns.is_last_user = true %}{{ '<｜User｜>' + message['content'] }}{% endif %}"
    "{% if message['role'] == 'assistant' %}{% if ns.is_last_user %}{{ '<｜Assistant｜></think>' }}{% endif %}{% set ns.is_last_user = false %}{{ message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and ns.is_last_user %}{{ '<｜Assistant｜></think>' }}{% endif %}"
)


# ---------------------------------------------------------------------------
# V4 model adapter
# ---------------------------------------------------------------------------


class DeepSeekV4Adapter(nn.Module):
    """Wraps the native DeepSeek V4 ``Transformer`` to expose the
    duck-typed surface that ``extract_for_session`` and
    ``_WindowHookExtractor`` need.

    Specifically:
    - ``.config.num_hidden_layers``, ``.config.hidden_size``,
      ``.config.max_position_embeddings``
    - ``.model.layers[i]`` with ``.self_attn`` and ``.mlp`` on each block
    - ``forward(input_ids, attention_mask=..., use_cache=..., return_dict=...)``
    """

    def __init__(self, transformer: nn.Module, args):
        super().__init__()
        self.model = transformer  # _WindowHookExtractor accesses model.model
        self.config = SimpleNamespace(
            num_hidden_layers=args.n_layers,
            hidden_size=args.dim,
            max_position_embeddings=DEEPSEEK_V4_MAX_CONTEXT,
        )
        # _WindowHookExtractor.register_hooks looks for layer.self_attn
        # and layer.mlp.  DeepSeek V4 Block has .attn (sparse MLA) and
        # .ffn (MoE).  Alias so the hook finder works unchanged.
        for block in transformer.layers:
            block.self_attn = block.attn
            block.mlp = block.ffn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        use_cache: bool = False,
        return_dict: bool = True,
        **_,
    ):
        # V4's Transformer.forward builds its own causal mask, manages
        # sliding-window + compressed KV cache internally, and passes
        # input_ids through to each block (for hash-based MoE routing).
        # Feature extraction always runs fresh full-context windows.
        return self.model(input_ids, start_pos=0)


# ---------------------------------------------------------------------------
# V4-specific hook extractor
# ---------------------------------------------------------------------------


class _V4WindowHookExtractor(_WindowHookExtractor):
    """Override the ``hidden`` stream hook for DeepSeek V4's Block.

    V4 uses Hyper-Connections: Block.forward returns a single tensor of
    shape ``[b, s, hc_mult, d]`` where ``hc_mult=4``.  The ``attn`` and
    ``mlp`` submodule hooks return standard ``[b, s, d]`` tensors (the HC
    reduce/expand happens outside those submodules), so they work unchanged.

    For the hidden stream we reduce the HC dimension via mean to produce
    the expected ``[b, s, d]`` shape.
    """

    def _make_hook(self, layer_idx: int, stream: str):
        if stream != "hidden":
            return super()._make_hook(layer_idx, stream)

        def hook(_module, _inputs, output):
            if self.target_positions is None:
                return
            # Block returns [b, s, hc_mult, d]; reduce to [b, s, d].
            tensor = output.mean(dim=2)
            positions = self.target_positions.to(tensor.device)
            extracted = tensor[0, positions, :]  # (n_targets, hidden)
            self.stream_outputs[stream][layer_idx] = extracted.to(
                dtype=torch.bfloat16, device=self.primary_device
            )

        return hook


# ---------------------------------------------------------------------------
# Runtime loader
# ---------------------------------------------------------------------------


DEEPSEEK_V4_MAX_CONTEXT = 1_048_576


def _load_v4_runtime(
    ckpt_path: str,
    ds_config: str,
    model_hf_id: str,
    window_fraction: float = 0.3,
):
    """Load DeepSeek V4 model + tokenizer + hook extractor.

    Returns the same ``(model, tokenizer, hook)`` triple that
    ``extract_features._load_runtime`` returns.
    """
    from model import ModelArgs, Transformer

    if IS_DISTRIBUTED and not dist.is_initialized():
        dist.init_process_group("nccl")
    if IS_DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)

    torch.set_default_dtype(torch.bfloat16)

    with open(ds_config) as f:
        config_dict = json.load(f)
    kv_cache_len = int(window_fraction * DEEPSEEK_V4_MAX_CONTEXT)
    config_dict["max_seq_len"] = kv_cache_len
    args = ModelArgs(**config_dict)

    _print(
        f"Loading DeepSeek V4: {args.n_layers} layers, "
        f"dim={args.dim}, dtype={args.dtype}, world_size={WORLD_SIZE}, "
        f"hc_mult={args.hc_mult}, experts={args.n_routed_experts}x{args.n_activated_experts}, "
        f"kv_cache={kv_cache_len} (window_fraction={window_fraction})"
    )

    with torch.device("cuda"):
        transformer = Transformer(args)

    load_sharded_model(transformer, ckpt_path, LOCAL_RANK, WORLD_SIZE)
    transformer.eval()
    _print(f"Loaded shards for rank {LOCAL_RANK}")

    # V4's helper functions (get_window_topk_idxs, get_compress_topk_idxs)
    # create tensors via torch.arange without specifying a device.  Setting
    # the default device ensures they land on the correct rank's GPU.
    torch.set_default_device("cuda")

    model = DeepSeekV4Adapter(transformer, args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        _print("No chat template found, using DeepSeek chat template")
        tokenizer.chat_template = DEEPSEEK_CHAT_TEMPLATE

    primary_device = f"cuda:{LOCAL_RANK}" if IS_DISTRIBUTED else "cuda:0"
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    hook = _V4WindowHookExtractor(model, num_layers, hidden_dim, primary_device)
    hook.register_hooks()

    _print(
        f"DeepSeek V4 ready ({num_layers} layers, {3 * num_layers} hooks registered)"
    )
    return model, tokenizer, hook


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../../../conf/extract_features",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Pop V4-specific keys before merging into ExtractionConfig.
    raw = OmegaConf.to_container(cfg, resolve=True)
    ckpt_path = raw.pop("ckpt_path", None)
    ds_config = raw.pop("ds_config", None)
    if not ckpt_path or not ds_config:
        raise ValueError(
            "ckpt_path=<converted-shards-dir> and "
            "ds_config=<path-to-vgdl_DS4.json> are required"
        )

    schema = OmegaConf.structured(ExtractionConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    typed_cfg: ExtractionConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    if typed_cfg.prompts in (None, "", "???"):
        raise ValueError("prompts=<path-to-.replay.json.gz> is required")
    if typed_cfg.model in (None, "", "???"):
        raise ValueError("model=<hf-model-id> is required (for output-dir naming)")

    import csv  # noqa: F401
    import glob as _glob

    pattern = typed_cfg.prompts
    matches = sorted(_glob.glob(pattern))
    if not matches:
        lit = Path(pattern)
        if lit.exists():
            matches = [str(lit)]
    if not matches:
        raise FileNotFoundError(f"No files match prompts={pattern!r}")
    prompts_files = [Path(p).resolve() for p in matches]

    if IS_MAIN:
        print("=" * 60)
        print("DeepSeek V4 sliding-window feature extraction")
        print("=" * 60)
        print(f"Prompts pattern:    {pattern}")
        print(f"Sessions matched:   {len(prompts_files)}")
        for p in prompts_files:
            print(f"  - {p}")
        print(f"Model (naming):     {typed_cfg.model}")
        print(f"Checkpoint:         {ckpt_path}")
        print(f"DS config:          {ds_config}")
        print(f"Output dir:         {typed_cfg.output_dir}")
        print(f"Window fraction:    {typed_cfg.window_fraction}")
        print(f"Overlap:            {typed_cfg.overlap}")
        print(f"Action compression: {typed_cfg.action_compression}")
        print(f"World size:         {WORLD_SIZE}")
        print("=" * 60)

    exp_db_client = None
    exp_db_worker = ""
    if typed_cfg.exp_db.enabled and IS_MAIN:
        import socket

        from src.llm_eval.neurips_exp_db import ExpDB

        if not typed_cfg.wandb_project:
            raise ValueError("exp_db.enabled=true requires wandb_project to be set")
        exp_db_client = ExpDB(typed_cfg.exp_db.region)
        exp_db_worker = socket.gethostname()
        _print(f"ExpDB: tracking enabled (region={typed_cfg.exp_db.region})")

    if typed_cfg.exp_db.enabled:
        from src.llm_eval.human_replay.extract_features import (
            _prefilter_claimable_sessions,
        )

        prompts_files = _prefilter_claimable_sessions(
            prompts_files, typed_cfg, exp_db_client
        )
        if not prompts_files:
            _print("No claimable sessions; exiting before model load.")
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
                "backend": "deepseek_v4_torchrun",
                "ckpt_path": ckpt_path,
                "ds_config": ds_config,
                "window_fraction": typed_cfg.window_fraction,
                "overlap": typed_cfg.overlap,
                "action_compression": typed_cfg.action_compression,
                "world_size": WORLD_SIZE,
            },
        )

    load_t0 = time.time()
    model, tokenizer, hook = _load_v4_runtime(
        ckpt_path, ds_config, typed_cfg.model, typed_cfg.window_fraction
    )
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
        from datetime import datetime

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
            f"{'total (excl. load)':50s}  {'':>8s}  {'':>5s}  "
            f"{'':>4s}  {'':>8s}  {total_fwd:>7.2f}  {total_wall:>7.2f}"
        )
        print(
            f"{'runtime load wall_s':50s}  {'':>8s}  {'':>5s}  "
            f"{'':>4s}  {'':>8s}  {'':>7s}  {load_wall_s:>7.2f}"
        )
        print("=" * 72)

        if summaries:
            variant_tag = "compressed" if typed_cfg.action_compression else "all"
            model_id = sanitize_model_id(typed_cfg.model)
            csv_path = (
                Path(typed_cfg.output_dir)
                / f"model-{model_id}"
                / variant_tag
                / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            import csv

            cols = sorted({k for s in summaries for k in s})
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(summaries)
            print(f"Benchmark CSV: {csv_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
