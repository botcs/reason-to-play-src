# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""DeepSeek V3.2 multi-turn sliding-window feature extraction.

Uses the bundled ``agents.lrm.backends.deepseek_v32`` runtime with converted
checkpoint shards and the same windowing and alignment as ``extract.py``.

Run with torchrun and Hydra key=value overrides::

    torchrun --nproc-per-node 8 --standalone -m agents.lrm.features.extract_deepseek_v32 \
        prompts="path/to/*.replay.json.gz" model=deepseek-ai/DeepSeek-V3.2 \
        +ckpt_path=/path/to/converted/shards +ds_config=/path/to/vgdl_DS32.json output_dir=out/features

Supply a runtime JSON configuration matching the converted checkpoint. Convert checkpoints with
``python -m agents.lrm.backends.deepseek_v32.convert --help``.
Install the runtime's GPU dependencies before running extraction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn


import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from agents.lrm.backends.deepseek_v32.load_utils import load_sharded_model

from agents.lrm.features.extract import (
    IS_DISTRIBUTED,
    IS_MAIN,
    LOCAL_RANK,
    WORLD_SIZE,
    _WindowHookExtractor,
    _finish_extraction_batch,
    _print,
    extract_for_session,
)
from agents.lrm.features.feature_saver import sanitize_model_id
from agents.lrm.config import ExtractionConfig

# DeepSeek V3 chat template — fallback when the tokenizer has none.
DEEPSEEK_V3_CHAT_TEMPLATE = (
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
# V3.2 model adapter
# ---------------------------------------------------------------------------


class DeepSeekV32Adapter(nn.Module):
    """Wraps the native DeepSeek V3.2 ``Transformer`` to expose the
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
            max_position_embeddings=DEEPSEEK_V32_MAX_CONTEXT,
        )
        # _WindowHookExtractor.register_hooks looks for layer.self_attn
        # and layer.mlp.  DeepSeek V3.2 Block has .attn (MLA) and
        # .ffn (MLP/MoE).  Alias so the hook finder works unchanged.
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
        # V3.2's Transformer.forward builds its own causal mask from seqlen
        # and has start_pos for KV-cache offset.  Feature extraction always
        # runs fresh full-context windows, so start_pos=0.
        return self.model(input_ids, start_pos=0)


# ---------------------------------------------------------------------------
# V3.2-specific hook extractor
# ---------------------------------------------------------------------------


class _V32WindowHookExtractor(_WindowHookExtractor):
    """Override the ``hidden`` stream hook for DeepSeek V3.2's Block.

    V3.2 Block.forward returns ``(x, residual)`` where ``x`` is the FFN
    delta and ``residual`` is the running sum through attention.  The true
    post-layer hidden state (analogous to HF's layer output) is their sum:
    ``x + residual``.  The ``attn`` and ``mlp`` submodule hooks return plain
    tensors, so they work unchanged.
    """

    def _make_hook(self, layer_idx: int, stream: str):
        if stream != "hidden":
            return super()._make_hook(layer_idx, stream)

        def hook(_module, _inputs, output):
            if self.target_positions is None:
                return
            # Block returns (x, residual); true hidden = x + residual.
            x, residual = output
            tensor = x + residual
            positions = self.target_positions.to(tensor.device)
            extracted = tensor[0, positions, :]  # (n_targets, hidden)
            self.stream_outputs[stream][layer_idx] = extracted.to(
                dtype=torch.bfloat16, device=self.primary_device
            )

        return hook


# ---------------------------------------------------------------------------
# Runtime loader
# ---------------------------------------------------------------------------


DEEPSEEK_V32_MAX_CONTEXT = 163_840


def _load_v32_runtime(
    ckpt_path: str,
    ds_config: str,
    model_hf_id: str,
    window_fraction: float = 0.3,
):
    """Load DeepSeek V3.2 model + tokenizer + hook extractor.

    Returns the same ``(model, tokenizer, hook)`` triple that
    ``extract_features._load_runtime`` returns.
    """
    from agents.lrm.backends.deepseek_v32.model import ModelArgs, Transformer

    if IS_DISTRIBUTED and not dist.is_initialized():
        dist.init_process_group("nccl")
    if IS_DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)

    torch.set_default_dtype(torch.bfloat16)

    with open(ds_config) as f:
        config_dict = json.load(f)
    kv_cache_len = int(window_fraction * DEEPSEEK_V32_MAX_CONTEXT)
    config_dict["max_seq_len"] = kv_cache_len
    args = ModelArgs(**config_dict)

    _print(
        f"Loading DeepSeek V3.2: {args.n_layers} layers, "
        f"dim={args.dim}, dtype={args.dtype}, world_size={WORLD_SIZE}, "
        f"kv_cache={kv_cache_len} (window_fraction={window_fraction})"
    )

    with torch.device("cuda"):
        transformer = Transformer(args)

    load_sharded_model(transformer, ckpt_path, LOCAL_RANK, WORLD_SIZE)
    transformer.eval()
    _print(f"Loaded shards for rank {LOCAL_RANK}")

    model = DeepSeekV32Adapter(transformer, args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        _print("No chat template found, using DeepSeek V3 template")
        tokenizer.chat_template = DEEPSEEK_V3_CHAT_TEMPLATE

    primary_device = f"cuda:{LOCAL_RANK}" if IS_DISTRIBUTED else "cuda:0"
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    hook = _V32WindowHookExtractor(model, num_layers, hidden_dim, primary_device)
    hook.register_hooks()

    _print(
        f"DeepSeek V3.2 ready ({num_layers} layers, {3 * num_layers} hooks registered)"
    )
    return model, tokenizer, hook


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../configs/extract_features",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Pop V3.2-specific keys before merging into ExtractionConfig.
    raw = OmegaConf.to_container(cfg, resolve=True)
    ckpt_path = raw.pop("ckpt_path", None)
    ds_config = raw.pop("ds_config", None)
    if not ckpt_path or not ds_config:
        raise ValueError(
            "+ckpt_path=<converted-shards-dir> and "
            "+ds_config=<runtime-config.json> are required"
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
        print("DeepSeek V3.2 sliding-window feature extraction")
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

    wandb_run = None
    if typed_cfg.wandb_project and IS_MAIN:
        import wandb

        wandb_run = wandb.init(
            project=typed_cfg.wandb_project,
            config={
                "prompts_pattern": pattern,
                "n_sessions": len(prompts_files),
                "model": typed_cfg.model,
                "backend": "deepseek_v32_torchrun",
                "ckpt_path": ckpt_path,
                "ds_config": ds_config,
                "window_fraction": typed_cfg.window_fraction,
                "overlap": typed_cfg.overlap,
                "action_compression": typed_cfg.action_compression,
                "world_size": WORLD_SIZE,
            },
        )

    load_t0 = time.time()
    model, tokenizer, hook = _load_v32_runtime(
        ckpt_path, ds_config, typed_cfg.model, typed_cfg.window_fraction
    )
    load_wall_s = time.time() - load_t0
    _print(f"Runtime ready in {load_wall_s:.1f}s")

    summaries: list[dict] = []
    failed_sessions: list[str] = []
    skipped_sessions = 0
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
            )
        except Exception as e:
            if typed_cfg.continue_on_session_error:
                failed_sessions.append(str(prompts_file))
                _print(
                    f"!!! Session {prompts_file.name} failed "
                    f"({type(e).__name__}: {e!r}) -- continuing"
                )
                continue
            raise
        if result is None:
            if IS_MAIN:
                skipped_sessions += 1
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

    _finish_extraction_batch(failed_sessions, len(summaries), skipped_sessions)


if __name__ == "__main__":
    main()
