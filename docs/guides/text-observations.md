# Text-observation experiments

`src/llm_eval/human_replay/generate_observations.py` is an optional utility for
independently prompted text-observation experiments. It accepts
archived JSONL prompt records and writes `observations.jsonl` plus metadata.
These outputs are not replay sessions and are not accepted by the current
`extract_features.py` workflow. Use `run_replay.py` for the selected human
replay/imputation pipeline described in [workflow guide](../reproducibility.md).

The batch wrapper uses temperature 0.5, top-k 50, top-p 0.95 and seed 0.
Record the model snapshot and vLLM environment with each experiment.

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
    --output-dir out/text_observations
python -m pip freeze > out/text_observations/environment.txt
```

The basename convention is `{subject}_{game}_{prompt_variant}.jsonl`, with a
three-part variant such as `first_person_all`. Each line contains `messages`
(chat role/content dictionaries), `game_name`, `level_id`, and optional trial,
run, play and step identifiers. A sibling `.meta.json` supplies source metadata.
Existing outputs are skipped unless `--overwrite` is supplied, so use a separate
output directory for each model snapshot and environment.

Validation covers lazy imports, command-line help, and a fake vLLM batch
through the complete JSONL-to-output path, including batches spanning files.
These checks do not validate real vLLM GPU generation.
