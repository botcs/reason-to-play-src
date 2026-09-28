# DDQN, EfficientZero and EMPA

For analysis of recorded baseline behaviour, use the
[dataset analysis guide](dataset-analysis.md#behavioral-analysis).
This guide covers baseline input formats, feature extraction and optional
training. [sources.json](../../baselines/sources.json) records source commits; a source pin
alone does not identify the run that produced an archived result.

Directory paths in the source table below are relative to `baselines/`.
Run shell commands from the checkout root unless another directory is stated.

## Environment and included sources

Use the checkout with the [analysis environment](dataset-analysis.md#install),
a hardware-appropriate PyTorch build and:

```bash
python -m pip install -r baselines/requirements-inference.txt
```

The baseline sources are included for inference without author GitHub
credentials. Keep their engine revisions separate:

| Directory | Role |
| --- | --- |
| `vendor/rc_rl/extraction/` | DDQN feature extraction |
| `vendor/rc_rl/current/` | DDQN training and behavioural generation |
| `vendor/rc_rl/ez/` | EfficientZero's environment, level transformations and warmup levels |
| `vendor/efficientzero/` | EfficientZero model inference |
| `extraction/efficientzero/` | Activation and trace extraction scripts |

The source records include commits and per-file hashes. Upstream license and
publication-permission limits are listed in [THIRD_PARTY.md](../../THIRD_PARTY.md)
and [EfficientZero provenance](../../baselines/extraction/efficientzero/PROVENANCE.json).
The repository's MIT license does not replace those terms.

## Offline behavioural inputs

Pass the following optional arguments to the episode exporter in the
[dataset guide](dataset-analysis.md#behavioral-analysis):

| Argument | Input |
| --- | --- |
| `--ddqn` | DDQN episode-history JSON |
| `--efficientzero` | Directory containing `GAME/self_play_episodes.csv` |
| `--empa` | Directory containing EMPA1 episode-summary JSON |

EMPA2 is excluded from the paper comparison. No cloud access is required to
read these files. The exporter retains levels 0–8, excludes practice, preserves
episode order and converts EMPA's one-based level filenames. Human keypress
frames, LLM decisions and baseline recorded steps remain separate units;
unknown frame counts remain null.

The plotter's retrospective blocked-curriculum transformation is an analysis
choice. `--no-forced-blocked-curricula` retains the original reached-level
records. State the censoring budget and whether solve-rate denominators count
available or reached levels; the human and model advancement protocols differ.

Only when collecting additional upstream records, use the source exporters:

```bash
python tools/export_ddqn_history.py --output data/baselines/ddqn.json
python tools/convert_empa_summaries.py \
  --input /absolute/data/EMPA1-behaviour --output data/empa-json --trusted-pickle
```

The DDQN exporter requires W&B access to `dpag-rl/ddqn-vgdl` and uses
`scan_history()` for complete histories. A sampled older cache may omit
records. The EMPA converter is for trusted source pickles; the ordinary analysis
input is already JSON.

## DDQN features and training

The [fMRI guide](fmri-preprocessing.md#process-bold-and-align-baseline-features)
contains the DDQN extraction command and local checkpoint-map format.
`reason_to_play.features.ddqn` selects the checksum-verified
`vendor/rc_rl/extraction/` snapshot in an editable checkout; use `--rc-rl-dir`
for an explicit compatible source directory. Checkpoint downloads require
`--allow-checkpoint-download`; a local checkpoint map supports offline use.
The original `trial1-sequential` checkpoint mapping does not establish the
weights behind every archived DDQN feature family.

For a new training run, from `baselines/vendor/rc_rl/current/`:

```bash
python runDDQN.py --game_name vgfmri4_bait --random_seed 7 --no_wandb
```

This is an entry-point example, not the paper's full training configuration.
Use the recorded sweep/run settings when comparing reported results. A small CPU
training check under Python 3.12 validates execution, not historical performance.

## EfficientZero hidden features

The runner verifies source checksums and selects the bundled baseline engine
without Ray, MuJoCo, Atari or TorchRL. An optional `--rc-rl-dir` accepts a clean
checkout of the locked engine revision. Human input images use recorded
rectangles and colours; the replay's single translated game definition supplies
draw order. This does not simulate a new human trajectory.

Each checkpoint must retain its `models/` directory and sibling
`logs/Train.log`: the extractor reads the recorded architecture from that log.
[checkpoint_paths.json](../../baselines/extraction/efficientzero/checkpoint_paths.json) records
the six-game source selection. Choose the checkpoint/log pair from the dataset
manifest or your own run, preserving their relative layout. Its `release_models`
entries identify the indexed source snapshot and do not establish an association
with every final paper array.

From the checkout root:

```bash
python baselines/run_efficientzero.py traces -- \
  --model /absolute/data/ez-run/models/model_180000.p \
  --dataset-root /absolute/data/behavior/human \
  --subj-id 12 --run-id 6 --play-id 0 --game-name vgfmri4_bait \
  --play-key 606debc9b4f366ca0cba5fda \
  --heads all --device cpu \
  --trace-layers-json /absolute/reason-to-play-src/baselines/extraction/efficientzero/trace_layers.json \
  --trace-output /absolute/data/ez-features/vgfmri4_bait/subj12/run6/play0_key606debc9b4f366ca0cba5fda/traces.pt
```

`--dataset-root` accepts the human directory or a standalone self-contained
human file. `--play-key` is the original play ID and resolves repeated
`play_id` values. The participant/run/play selection above identifies a recorded
short play. Each new trace has a `.pt.provenance.json` sidecar with checkpoint,
configuration, source and output hashes. This identifies the new extraction;
it does not retroactively identify the checkpoint behind an archived trace.

The base aligner's `--ez-features-dir` expects
`GAME/subjN/runN/playN_keyID/traces.pt` below that directory and writes the
aligned EZ feature sidecar. For representation activations and metrics instead
of selected layer traces, inspect:

```bash
python baselines/run_efficientzero.py features -- --help
```

## Optional full training checkouts

The source bootstrap fetches exact commits into ignored `baselines/checkouts/`
directories. It does not install dependencies and refuses to overwrite a
changed checkout:

```bash
python baselines/setup.py --list
python baselines/setup.py efficientzero
# Full RC_RL repositories require authorized access:
python baselines/setup.py rc-rl rc-rl-current rc-rl-ez --transport ssh
```

EfficientZero training uses its separate Python 3.8/Ray 1.0 environment.
Omit the machine-specific `prefix:` from its original `environment.yaml`.
The configuration helper sets the bundled game directory while preserving
training settings. Supply that config directly, since the training CLI can
replace standalone absolute-path overrides when merging its experiment file:

```bash
python baselines/prepare_efficientzero_config.py --output data/efficientzero-training.yaml
python baselines/run_efficientzero.py train -- exp_config=/absolute/reason-to-play-src/data/efficientzero-training.yaml
```

Inference and small CPU checks do not validate a full training run. The EMPA
reference is `tsividis/vgdl` at the `empa-reference` pin; it does not establish
the exact revision behind received EMPA1 episode summaries. Remaining
checkpoint/layer/result associations are listed in
[reproduction limits](../reproduction-limits.md).
