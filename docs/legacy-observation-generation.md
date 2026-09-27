# Legacy observation generation

`src/llm_eval/human_replay/generate_observations.py` is an optional utility for
the older, independently prompted text-observation experiments. It accepts
archived JSONL prompt records and writes `observations.jsonl` plus metadata.
These outputs are not replay sessions and are not accepted by the current
`extract_features.py` workflow. Use `run_replay.py` for the selected human
replay/imputation pipeline described in [reproducibility.md](reproducibility.md).

The missing batch wrapper was recovered from development commit
`0cfe0aff2ebc4efe0807cd2287a3dc1aff7166cb` (28 March 2026). The original file hash,
Git blob and small interface adaptations are recorded in
`src/llm_eval/shared/vllm_wrapper.py`. It preserves the historical chat-template
handling and sampling settings: temperature 0.5, top-k 50, top-p 0.95, seed 0.
The old dependency file did not pin a vLLM version; this recovery does not
establish an identical historical GPU runtime or identical generated text.

Use a **separate GPU environment**, following the
[vLLM GPU installation instructions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
Install a vLLM build matching the machine's GPU/CUDA stack and let that build
select its compatible Torch dependencies. Do not combine it with the fixed
Torch/kernel pins in `requirements-full.txt`. This standalone utility requires
vLLM and Torch; `wandb` is needed only when `--wandb-project` is supplied.
Importing the wrapper or viewing CLI help does not load either GPU package.

Run from the repository root in that environment:

```bash
python -m src.llm_eval.human_replay.generate_observations --help
python -m src.llm_eval.human_replay.generate_observations \
    --prompts /absolute/path/to/sub-01_vgfmri3_bait_first_person_all.jsonl \
    --model /absolute/path/to/a/pinned/model/snapshot \
    --max-model-len 8192 --batch-size 32 --n-gpus 1 \
    --output-dir out/legacy_observations
python -m pip freeze > out/legacy_observations/environment.txt
```

The basename convention is `{subject}_{game}_{prompt_variant}.jsonl`, with a
three-part variant such as `first_person_all`. Each line contains `messages`
(chat role/content dictionaries), `game_name`, `level_id`, and optional trial,
run, play and step identifiers. A sibling `.meta.json` supplies source metadata.
Existing outputs are skipped unless `--overwrite` is supplied, so use a separate
output directory for each model snapshot and environment.

Validation covers lazy imports, command-line help, and a fake vLLM batch
through the complete JSONL-to-output path, including batches spanning files.
No real vLLM GPU generation was run during this source recovery.
