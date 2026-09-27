# Baseline reproduction

This directory unifies the DDQN, EfficientZero V2, and EMPA entry points. Different historical RC_RL revisions have different roles; `sources.json` records exact commits. A recovered source revision is evidence of available code, not proof of which run produced every paper result.

## Included source and optional training checkouts

The public code includes the essential RC_RL engine, DDQN modules, sweep configurations, and text game definitions under `vendor/rc_rl/`. The three roles retain separate snapshots: `extraction` is the historical extractor pin, `ez` is the environment with EfficientZero level transformations and warmup levels, and `current` includes later DDQN replay fixes. Each has per-file SHA256, original Git blob IDs, the exact source commit, and a license-unspecified notice. No author GitHub credentials are needed for the included inference and training entry points.

The source bootstrap is optional when the full original repositories are needed:

```bash
python baselines/setup.py --list
python baselines/setup.py efficientzero
# Optional full historical RC_RL checkouts require authorized access:
python baselines/setup.py rc-rl rc-rl-current rc-rl-ez --transport ssh
```

Full checkouts live under `baselines/checkouts/` and are excluded from Git. The bootstrap fetches exact commits, does not install packages or initialize unrelated submodules, and refuses to overwrite an existing changed checkout. The two recovered text-game directories preserve their original definitions; they are not substituted with the separately converted LLM game rules.

## DDQN

`vendor/rc_rl/current/runDDQN.py`, `VGDLEnv.py`, `rl_models.py`, and the committed sweep YAML files provide training and behavioral generation. Install the public analysis requirements, a hardware-appropriate Torch build, and `baselines/requirements-inference.txt`. A tiny CPU training check was run under Python 3.12; full historical performance still depends on the original run configuration. For example, from `baselines/vendor/rc_rl/current/`:

```bash
python runDDQN.py --game_name vgfmri4_bait --random_seed 7 --no_wandb
```

This is an entry-point example, not a declaration of the paper's exact hyperparameters. The historical S3 checkpoints and sweep/run configurations determine the scientific selection. The neural extraction command is in `analysis/tomov23/stages/extract_model_features_to_npz.py`; pass `--rc-rl-dir` when using a nondefault checkout. Its default is the included, checksum-verified `baselines/vendor/rc_rl/extraction/` snapshot.

Export full behavioral records from the historical W&B sweeps without sampling:

```bash
python scripts/analysis/export_ddqn_history.py --output data/baselines/ddqn.json
```

This reads W&B and requires access to `dpag-rl/ddqn-vgdl`. Existing archived JSON can instead be supplied directly to the offline importer below. The old internal exporter called the sampled `history()` endpoint; successful parsing of an old cache does not establish that it contains every episode. The new exporter uses `scan_history()`.

## EfficientZero hidden features

The complete recovered extraction scripts live in `recovered/efficientzero/`. Their SHA256/source records and publication-permission gap are in `PROVENANCE.json`; they have **not** been assigned this repository's MIT license. The compact model implementation in `vendor/efficientzero/` retains the upstream GPL-3.0 license. Training-only eager imports were removed or made lazy. The action plane now follows the state tensor’s device rather than forcing CUDA. Architecture, state-dict loading, action encoding, normalization, and support math remain unchanged. The recovered trace script was modified relative to its recorded Git HEAD, and the layer list was untracked; the S3 file hashes, rather than that commit alone, identify this recovered snapshot.

Use the public Torch/analysis environment and install `baselines/requirements-inference.txt`. The bundled `vendor/rc_rl/ez/` engine is selected automatically. Inference does not need Ray, MuJoCo, Atari, or TorchRL. The wrapper verifies source checksums and sets `RC_RL_ROOT`, `RC_RL_PATH`, and `EZ_ROOT` explicitly, keeping this baseline engine separate from the LLM engine. An optional `--rc-rl-dir` can point at a clean external checkout of the same locked revision.

Preserve each checkpoint's `models/` directory and sibling `logs/Train.log`; the extractor reconstructs the exact architecture from the recorded training configuration. The recovered six-game selection map is in `recovered/efficientzero/checkpoint_paths.json`, with paths relative to the original `EfficientZeroV2/results/` tree. The same file’s `release_models` gives exact normalized checkpoint and sibling log paths below the staged HF dataset root; retain literal percent escapes when locating files. Map entries alone do not prove association with the final paper arrays.

```bash
python baselines/run_efficientzero.py traces -- \
  --model /absolute/data/ez-run/models/model_180000.p \
  --dataset-root /absolute/data/behavior \
  --subj-id 12 --run-id 6 --play-id 0 --game-name vgfmri4_bait \
  --play-key 606debc9b4f366ca0cba5fda \
  --heads all --device cpu \
  --trace-layers-json /absolute/reason-to-play-src/baselines/recovered/efficientzero/trace_layers.json \
  --trace-output /absolute/data/ez-features/vgfmri4_bait/subj12/run6/play0_key606debc9b4f366ca0cba5fda/traces.pt
```

Use absolute file paths because the wrapper runs from the recovered source directory. The exact Mongo `_id` prevents ambiguity when `play_id` repeats. The example is a verified short recorded human play, not a new synthetic fixture. Each trace also gets a `.pt.provenance.json` sidecar with checkpoint/configuration/source/output SHA256 values. This records new extraction provenance without retroactively assigning a producing checkpoint to historical traces. The neural aligner's `--ez-features-dir /absolute/data/ez-features` consumes this `GAME/subjN/runN/playN_keyID/traces.pt` layout and writes the aligned EZ sidecar.

For representation activations and metrics rather than the selected layer traces:

```bash
python baselines/run_efficientzero.py features -- --help
```

For full training, fetch `efficientzero` and use its separate historical environment (`environment.yaml` records Python 3.8 and Ray 1.0). Its old conda file contains a machine-specific `prefix:`; omit that field when creating a local environment. Training is deliberately not claimed to be validated by an inference smoke test. The configuration helper changes only the private machine-specific game directory to the bundled game path; all scientific training settings are retained. The old training CLI can discard absolute-path overrides when re-merging its experiment file, so supply the generated config file rather than an `env.game_folder` override alone:

```bash
python baselines/prepare_efficientzero_config.py --output data/efficientzero-training.yaml
python baselines/run_efficientzero.py train -- exp_config=/absolute/reason-to-play-src/data/efficientzero-training.yaml
```

## Offline behavioral analysis

The importer accepts the original EfficientZero `GAME/self_play_episodes.csv`, DDQN JSON, and EMPA1 episode-summary JSON. EMPA2 is excluded from the paper comparison.

```bash
python scripts/analysis/convert_empa_summaries.py \
  --input /absolute/data/EMPA1-behaviour --output data/empa-json --trusted-pickle
python scripts/analysis/build_episodes.py \
  --human-data /absolute/data/prepare_behavioral_data \
  --replays /absolute/data/replay.generative.replay.json.gz \
  --efficientzero /absolute/data/EZV2-behaviour \
  --ddqn data/baselines/ddqn.json --empa data/empa-json \
  --output data/episodes.csv
python scripts/analysis/plot_behavioural.py all --csv data/episodes.csv
```

The default cohort comparison retains levels 0–8, excludes practice run 00, uses EMPA's one-based filename conversion, and keeps episode order. Human outcomes come from the nullable play-document `win`, with avatar deaths identified before the incomplete branch. Human non-idle keypress frames, LLM decisions, and baseline recorded steps are explicitly distinct units; unknown baseline frame counts remain null. The plotting implementation preserves the submission's optional retrospective blocked-curriculum transformation and its default censoring convention. These are analytical transformations, not evidence that the original human and model advancement protocols matched; use `--no-forced-blocked-curricula` to inspect the original reached-level records. Plot defaults and unplayed-level censor budgets must be stated when reporting a comparison.

## Remaining source provenance

The original human experiment / EMPA reference is `tsividis/vgdl` at the `empa-reference` pin, not the current RC_RL DDQN branch. This recovered EMPA reference does not establish the revision that produced the received EMPA1 behavioral pickles. Exact feature/checkpoint-to-result provenance and collaborator source publication permission are tracked as release gaps rather than silently inferred.
