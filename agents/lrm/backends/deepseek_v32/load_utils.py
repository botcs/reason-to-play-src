from pathlib import Path

from safetensors.torch import load_model


def load_sharded_model(model, ckpt_path, rank, world_size):
    """Load model weights from chunked or single safetensors files.

    Tries chunked format (model{rank}-mp{world_size}-NNNNN-of-NNNNN.safetensors)
    first, falls back to single file (model{rank}-mp{world_size}.safetensors).
    """
    base = Path(ckpt_path)
    pattern = f"model{rank}-mp{world_size}-*-of-*.safetensors"
    chunks = sorted(base.glob(pattern))

    if chunks:
        for chunk_path in chunks:
            load_model(model, str(chunk_path), strict=False)
        return

    single = base / f"model{rank}-mp{world_size}.safetensors"
    if single.exists():
        load_model(model, str(single))
        return

    raise FileNotFoundError(
        f"No shards found at {base} for rank={rank}, world_size={world_size}\n"
        f"Looked for: {pattern} or {single.name}\n"
        f"Run: python -m agents.lrm.backends.deepseek_v32.convert "
        f"--hf-ckpt-path $HF --save-path {ckpt_path} "
        f"--n-experts 256 --model-parallel {world_size}"
    )
