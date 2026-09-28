# DDQN, EfficientZero and EMPA

For analysis of recorded baseline behaviour, use the
[dataset analysis guide](dataset-analysis.md#behavioral-analysis).
This guide covers baseline input formats, feature extraction and optional
training. The [baseline source pins](../../baselines/sources.json) and
[EfficientZero source pins](../../agents/efficientzero/sources.json) record source
commits; a source pin alone does not identify the run that produced an archived
result.

Directory paths in the source table below are relative to the checkout root.
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
| `baselines/vendor/rc_rl/extraction/` | DDQN feature extraction |
| `baselines/vendor/rc_rl/current/` | DDQN training and behavioural generation |
| `agents/efficientzero/environment/` | EfficientZero's environment, level transformations and warmup levels |
| `agents/efficientzero/inference/` | EfficientZero model inference |
| `agents/efficientzero/extract_features.py`, `extract_traces.py` | Activation and trace extraction |
| `agents/efficientzero/training/` | Optional pinned upstream training submodule |

The source records include commits and per-file hashes. Upstream license and
publication-permission limits are listed in [THIRD_PARTY.md](../../THIRD_PARTY.md)
and [EfficientZero provenance](../../agents/efficientzero/PROVENANCE.json).
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

Feature extraction uses the included `inference/` model and `observations.py`
renderer without initializing the training submodule or requiring Ray, MuJoCo,
Atari or TorchRL. Human input images use recorded rectangles and colours; the
translated VGDL parser supplies draw order from the replay's single game
definition. The `environment/` engine supports optional training. Extracting
recorded frames does not simulate a new human trajectory.

Each checkpoint must retain its `models/` directory and sibling
`logs/Train.log`: the extractor reads the recorded architecture from that log.
[checkpoint_paths.json](../../agents/efficientzero/checkpoint_paths.json) records
the six-game source selection. Choose the checkpoint/log pair from the dataset
manifest or your own run, preserving their relative layout. Its `release_models`
entries identify the indexed source snapshot and do not establish an association
with every final paper array.

From the checkout root:

```bash
python -m agents.efficientzero.extract_traces \
  --model /absolute/data/ez-run/models/model_180000.p \
  --dataset-root /absolute/data/behavior/human \
  --subj-id 12 --run-id 6 --play-id 0 --game-name vgfmri4_bait \
  --play-key 606debc9b4f366ca0cba5fda \
  --heads all --device cpu \
  --trace-layers-json /absolute/reason-to-play-src/agents/efficientzero/trace_layers.json \
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
python -m agents.efficientzero.extract_features --help
```

## EfficientZero training

The optional `agents/efficientzero/training/` submodule is Austin Andrews's
[EfficientZeroV2 fork](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490),
pinned to commit `29157d4892afd9467b1bd0994de1355086145490`. Initialize it only
when you want to train a model:

```bash
git submodule update --init -- agents/efficientzero/training
```

Analysis and feature extraction use the included code and do not require this
submodule. Training uses a separate upstream environment: follow the pinned
[installation instructions](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/INSTALL.md)
and [environment specification](https://github.com/A-Andrews/EfficientZeroV2/blob/29157d4892afd9467b1bd0994de1355086145490/environment.yaml).
Use a local environment location rather than the source file's machine-specific
`prefix:`, and build the upstream MCTS extensions as instructed. The
analysis/inference environment is not a full training install.

From the code checkout and its Python environment, create an experiment
configuration using:

```bash
python -m agents.efficientzero.prepare_training_config \
  --output /absolute/work/efficientzero-training.yaml
```

The helper reads `training/ez/config/exp/vgdl.yaml` and changes only
`env.game_folder` to the absolute bundled `environment/all_games_recovered`
directory. It writes a separate file and preserves the upstream training
settings. Pass that file as `exp_config`; the upstream CLI can replace individual
path overrides while merging its experiment configuration.

In the upstream training environment, run from the submodule directory. Replace
the absolute paths with your checkout and the configuration created above:

```bash
cd /absolute/reason-to-play-src/agents/efficientzero/training
RC_RL_PATH=/absolute/reason-to-play-src/agents/efficientzero/environment \
PYTHONPATH=/absolute/reason-to-play-src/agents/efficientzero/environment:/absolute/reason-to-play-src/agents/efficientzero/training \
python -m ez.train exp_config=/absolute/work/efficientzero-training.yaml
```

These are direct upstream training commands. The source pin and configuration
helper do not establish which revision produced each released checkpoint, and
full retraining has not been validated. The recorded RGB inputs, warmup levels,
checkpoint configurations and unresolved result-layer mapping are described in
[reproduction limits](../reproduction-limits.md#baseline-checkpoints-and-layer-mappings).

## Additional baseline sources

Optional full RC_RL checkouts require authorized source access:

```bash
python baselines/setup.py rc-rl rc-rl-current --transport ssh
```

The EMPA reference is `tsividis/vgdl` at the `empa-reference` pin; it does not
establish the exact revision behind received EMPA1 episode summaries.
