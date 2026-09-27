# DeepSeek V3.2 runtime

This is the custom inference runtime used by
`src.llm_eval.human_replay.extract_features_v32`, including local loading and
feature-hook adaptations. The [source provenance](../docs/source-provenance.md)
records the development snapshot and upstream source. The upstream MIT notice
is retained in [LICENSE](LICENSE). Model weights are not included.

From the repository root, convert the downloaded Hugging Face checkpoint into
shards for the number of GPUs you will use:

```bash
python deepseek_inference/convert.py \
    --hf-ckpt-path /absolute/path/to/DeepSeek-V3.2 \
    --save-path /absolute/path/to/converted-v32 \
    --n-experts 256 --model-parallel 8
```

Use the [extraction recipe](../docs/reproducibility.md#optional-deepseek-runtimes)
for `torchrun`, the custom model config, dependencies, and output layout.
`requirements.txt` in this directory is the historical upstream runtime list;
the repository's `requirements-full.txt` records the later CUDA/B200 research
environment. Choose a compatible kernel stack for your hardware rather than
installing both conflicting TileLang pins.
