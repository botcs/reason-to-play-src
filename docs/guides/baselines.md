# DDQN, EfficientZero and EMPA

For analysis of recorded baseline behaviour, use the
[dataset analysis guide](dataset-analysis.md#behavioral-analysis).
This guide covers baseline input formats, feature extraction and optional
training. The [baseline source pins](../../agents/ddqn/sources.json) and
[EfficientZero source pins](../../agents/efficientzero/sources.json) record source
commits; a source pin alone does not identify the run that produced an archived
result.

Directory paths in the source table below are relative to the checkout root.
Run shell commands from the checkout root unless another directory is stated.

## Environment and included sources

For EfficientZero feature extraction, install the checkout with its dedicated
extra and a hardware-appropriate PyTorch build:

```bash
python -m pip install -e '.[efficientzero]'
```

For DDQN, use the [analysis environment](dataset-analysis.md#install), a
hardware-appropriate PyTorch build and:

```bash
python -m pip install -e '.[ddqn]'
```

EfficientZero training uses the separate upstream environment described below.

The baseline sources are included for inference without author GitHub
credentials. Keep their engine revisions separate:

| Directory | Role |
| --- | --- |
| `agents/ddqn/environment/extraction/` | DDQN feature extraction |
| `agents/ddqn/environment/training/` | DDQN training and behavioural generation |
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
| `--ddqn` | `behavior/ddqn/episode-history.json` |
| `--efficientzero` | `behavior/efficientzero/`, containing `GAME/episodes.csv` |
| `--empa` | `behavior/empa/`, containing `GAME/trial-NN/level-NN.json` |

EMPA2 is excluded from the paper comparison. No cloud access is required to
read these files. The exporter retains levels 0–8, excludes practice, preserves
episode order. Released EMPA files use zero-based level IDs and retain their
original trial IDs; the source converter handles upstream one-based levels. Human keypress
frames, LLM decisions and baseline recorded steps remain separate units;
unknown frame counts remain null.

The plotter's retrospective blocked-curriculum transformation is an analysis
choice. `--no-forced-blocked-curricula` retains the original reached-level
records. State the censoring budget and whether solve-rate denominators count
available or reached levels; the human and model advancement protocols differ.

Only when collecting additional upstream records, use the source exporters:

```bash
python -m agents.ddqn.export_history --workers 32 --output /absolute/data/behavior/ddqn/episode-history.json
python -m agents.empa.import_results \
  --input /absolute/data/EMPA1-behaviour --output /absolute/data/behavior/empa --trusted-pickle
```

The DDQN exporter requires W&B access to `dpag-rl/ddqn-vgdl` and uses
`scan_history()` for complete histories. A sampled older cache may omit
records. The EMPA converter is for trusted source pickles; the ordinary analysis
input is already JSON.

## DDQN features and training

The [fMRI guide](fmri-preprocessing.md#process-bold-and-align-baseline-features)
contains the DDQN extraction command and local checkpoint-map format.
`agents.ddqn.extract_features` selects the checksum-verified
`agents/ddqn/environment/extraction/` implementation. Checkpoint downloads require
`--allow-checkpoint-download`; a local checkpoint map supports offline use.
The original `trial1-sequential` checkpoint mapping does not establish the
weights behind every archived DDQN feature family.

For a new training run:

```bash
python -m agents.ddqn.train --game_name vgfmri4_bait --random_seed 7 --no_wandb
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
  --trace-output /absolute/data/ez-features/bait_vgfmri4/sub-12/run-06/play-606debc9b4f366ca0cba5fda/traces.pt
```

`--dataset-root` accepts the human directory or a standalone self-contained
human file. `--play-key` is the original play ID and resolves repeated
`play_id` values. The participant/run/play selection above identifies a recorded
short play. Each new trace has a `.pt.provenance.json` sidecar with checkpoint,
configuration, source and output hashes. This identifies the new extraction;
it does not retroactively identify the checkpoint behind an archived trace.

To sample existing traces at the released BOLD samples, use the human JSONs
and that participant's directory containing `bold.npz` and `samples.npz`:

```bash
python -m analysis.neural.align_efficientzero \
  --subject-dir /absolute/data/neural/sub-13 \
  --behavior-dir /absolute/data/behavior/human \
  --trace-dir /absolute/data/features/efficientzero \
  --output /absolute/data/neural/sub-13/model-features/efficientzero.npz \
  --workers 32
```

The original play ID joins each trace to its human frames. Embedded participant,
run and game metadata must agree with the recording; a directory label alone
does not establish identity. `samples.npz` fixes the sample order and retained
play lengths. Frame timestamps are rounded to the nearest scanner TR, with the
same AR(1) offset as BOLD, and activations are averaged within each retained
sample. No raw MRI or source behavioural archive is needed.

The default output contains four representation and seven initial value/policy
hooks under their full names. `--include-dynamics` additionally includes the
four sparse dynamics/reward hooks, averaging entries at each engine timestep.
Missing plays or hooks have explicit coverage records. The output binds to the
BOLD and samples archives' checksums and ordered sample identities; the adjacent
`.npz.alignment.json` records per-layer coverage. Keep the two files together.
The [dataset guide](dataset-analysis.md#neural-encoding-from-processed-inputs) describes fitting
these features. Their association with archived paper results is covered in
[reproduction limits](../reproduction-limits.md#baseline-checkpoints-and-layer-mappings).

The base aligner's `--ez-features-dir` remains available when preparing new BOLD
inputs. For representation activations and metrics instead of selected layer
traces, inspect:

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

The [EMPA source record](../../agents/empa/sources.json) identifies the preceding
`tsividis/vgdl` implementation. It does not establish the exact revision behind
received EMPA1 episode summaries.
