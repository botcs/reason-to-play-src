# DeepSeek V4 inference runtime

This is the runtime used by `src.llm_eval.human_replay.extract_features_v4`.
The feature extractor uses the shared research extraction loop with V4's
custom tensor-parallel Transformer and TileLang kernels. Checkpoint weights
are not bundled.

## Provenance and license

The upstream implementation and encoder are available in the official
[DeepSeek-V4-Flash model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)
at revision `60d8d70770c6776ff598c94bb586a859a38244f1`. That repository's
[MIT license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/LICENSE)
is preserved verbatim in [LICENSE](LICENSE), including DeepSeek's copyright
notice. The primary source is the official Hugging Face repository; a missing
GitHub mirror does not remove the published source or its notice.

The research runtime was restored from the development repository's
`deepseek_v4_inference/` directory. `model.py` matches the pinned upstream
`inference/model.py` byte for byte. The research copy retains these existing
changes instead of silently updating numerical behavior:

- `kernel.py` differs at the in-place quantization cast (`out_dtype` in the
  research copy versus `FP8` in the pinned upstream file).
- `convert.py` writes bounded safetensors chunks and handles the research
  checkpoint variants; `load_utils.py` loads those chunks or a single shard.
- `vgdl_DS4*.json` contains the research extraction configurations, including
  Flash/Pro and base variants. Dependencies match the research environment.

For the public restoration, `encoding_dsv4.py` is copied unmodified from that
same upstream revision. `generate.py` imports it locally and uses the research
chunk loader; these packaging changes let the optional generation example
resolve its dependencies and read the output of the bundled converter.

## Setup and extraction

Use a GPU environment compatible with the runtime dependencies in
[requirements.txt](requirements.txt). This backend requires PyTorch 2.10 or
later, the pinned TileLang version, `fast_hadamard_transform`, and enough GPU
memory for the chosen model and tensor parallel degree. Install the public
research repository's core requirements first; those include conversion
utilities such as `tqdm`.

```bash
python -m pip install -r requirements.txt
python -m pip install -r deepseek_v4_inference/requirements.txt
```

Download the exact model revision used for the experiment separately. Convert
its checkpoints with `convert.py`, choosing the expert count, model parallel
degree and expert dtype from the corresponding model/config. For example,
V4 Pro uses 384 routed experts:

```bash
python deepseek_v4_inference/convert.py \
  --hf-ckpt-path /path/to/pinned/model-snapshot \
  --save-path /path/to/converted-checkpoints \
  --n-experts 384 --model-parallel 8

torchrun --nproc-per-node 8 --standalone \
  -m src.llm_eval.human_replay.extract_features_v4 \
  'prompts=/path/to/replays/*.replay.json.gz' \
  model=deepseek-ai/DeepSeek-V4-Pro \
  +ckpt_path=/path/to/converted-checkpoints \
  +ds_config=deepseek_v4_inference/vgdl_DS4.json \
  output_dir=out/features_v4
```

Use a fresh process for V3.2 versus V4 extraction because both runtimes expose
modules named `model` and `kernel`. Preserve the selected model revision,
configuration, converter flags and extraction code commit with each output.

The release checks parse the Python/config files and exercise the standalone
encoder. Full checkpoint conversion and multi-GPU inference require model
weights and GPU hardware and have not been rerun as part of this restoration.
