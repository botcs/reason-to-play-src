# Human behaviour and model prompts

A human replay is one self-contained JSON file for one participant, one game
and one prompt condition. It contains that participant's levels and attempts
across scanner runs, the recorded trajectory, and the exact saved conversation.

```text
behavior/human/sub-13/helper_vgfmri4/
  elaborate.human.replay.json.gz
  minimal.human.replay.json.gz
  oracle.human.replay.json.gz
```

The three conditions repeat the same measured behaviour and carry different
prompts. Readers select `elaborate` by default so an analysis counts each
participant's behaviour once. Pass `condition="minimal"` or `condition="oracle"`
to select another condition, or open one file directly. Each file can be read
without other human files, BSON, scanner sidecars or game-definition downloads.

Each file declares `schema: "reason-to-play/human-replay"`, `schema_version: 1`
and `source: "human"`. Its fields include `steps`,
`states`, `system_prompt`, `meta`, colour mapping and a single translated
`game_description`. The description uses the Colas interpreter's dialect shared
with the browser interpreter. The [game translation history](sources/README.md#game-translation)
includes the original translator and explains the source dialect.

## Plays and scanner timing

`plays` partitions the embedded trajectory. Each entry has `state_start` and
`state_count` into the expanded `states` array, its original `_id`, participant,
run, game and level IDs, nullable `win`, score, start/end times and original
input/event streams. `source_document_index` is the zero-based ordinal in the
original scanner-run file; it is distinct from `play_id`, which can repeat.
The original `_id` identifies a play across prompt conditions.

`scanner` inside each play retains the original run identity and timing needed
for neural alignment, including `scan_start_ts`. Repeating this small record
makes the file independently usable. It does not embed source game rules or
level layouts. Participant 11/run 05 has no separate source scanner record.
Its start time is derived from the nine play records, which all agree on that time. The
embedded scanner entry records this derivation and has `clock_only: true`.
The other selected runs use their original scanner clocks.

Top-level `started_at` and `finished_at` describe the replay export, not the
participant's session. Use play timestamps, frame `realworld_ts` and
`scanner.scan_start_ts` for measurements and neural alignment.

Human play outcomes are **true, false, or null**. `outcome` distinguishes `win`,
`avatar_died`, `loss` and `incomplete`. Avatar-death events are checked before
classifying a null result as incomplete. A frame's terminal `win=-1` or a
viewer's `won` flag is not the original play outcome.

The original inputs remain separate measurements:

- `actions` retains the original timestamped action sequence.
- `keydowns` and `keyups` retain original onset and release timestamps;
  `keyholds` retains onset/duration pairs.
- `events` retains the structured event records, including their timestamps,
  action, resource and interaction information where present.
- Each recorded frame retains its `keystate`, `keyPressType`, interaction lists,
  sprite creation/removal IDs and `dt` where supplied by the source.

Event times need not coincide with observed frame times. Do not pair sparse
actions with frames by array index or replace button onsets with frame times.
The archived full engine/control dump is outside this selected analysis
contract; [the historical source reference](sources/tomov23-behavior-notes.md)
documents how to obtain additional fields for a different study.

## Frames and geometry

A frame is one recorded engine state, including the initial state and states
with no action. `time` is the original engine tick, resetting to zero for each
play; `realworld_ts` is that frame's original wall-clock timestamp. A play with
N states has N−1 engine updates. Neural alignment uses timestamps, not an
assumed fixed frame rate. No frames are synthesized in gaps between plays.
Each frame also carries its original `source_play_id`.

Files use sprite-group delta encoding: the first state stores all sprite groups;
subsequent states store only groups that changed. `delta_encoded: true` marks
this representation. Standard gzip compresses the JSON. Expanding these deltas
copies recorded values; it does not run or simulate the game.

A sprite's `col` and `row` are its original drawing-rectangle coordinates divided
by the play's `block_size`. They retain fractional values. The web viewer renders
these positions directly, without shifting them to the next frame. Original
logical pixel coordinates `x` and `y` are also retained because the baseline model
inputs use them and they can differ from the drawing rectangle. Sprite IDs,
source keys, RGB colours, optional colour names and recorded resources remain
available. Empty sprite groups are retained.

Each play stores its numeric `grid_size`. Initial frames carry the observed
layout; there are no per-frame screenshots, ASCII copies, RNG dumps or a second
game definition. Prompt observations and baseline grids have their own
quantization and alignment rules; those do not replace recorded geometry.

## Conversation steps and feature identities

`steps` contains the saved prompt text, responses, actions, order and
metadata. A step is a selected observation/action pair, not every engine tick.
Idle frames and plays without selected prompt turns remain in `states` and
`plays`. `state_index` points into the embedded expanded trajectory.
`meta.action_frames_only` describes which observations were selected for prompt
steps; it does not remove idle frames from the recorded trajectory.

Each real step also identifies its source measurement:

| Field | Meaning |
| --- | --- |
| `source_play_id` | Original play `_id`. |
| `source_document_index` | Original play ordinal within its scanner run. |
| `source_frame_index` | Original frame index within that play. |
| `source_recording` | This file's canonical path relative to `behavior/human/`. |
| `realworld_ts` | Exact timestamp of that observation. |

These fields let tensor metadata identify its observation. They do not require
an external trajectory lookup to open the replay. Synthetic bookkeeping steps
whose actions begin with `_` are not additional observations. The extraction
pipeline preserves observation identities with the resulting activations.

## Reading a file

The reader decodes recorded fields into dictionaries using the Python
standard library; it does not import BSON or an engine. Source datetimes and
binary values in metadata/events use explicit `$rtp` encodings in JSON.

```python
from human.behavior import iter_plays, load_runs, play_states

root = "/data/reason-to-play/behavior/human"
runs = load_runs(root)
for play in iter_plays(root, subject="sub-13"):
    frames = play_states(play)
    scanner = runs[(int(play["subj_id"]), int(play["run_id"]))]
    times = [frame["ts"] - scanner["scan_start_ts"] for frame in frames]
```

The same functions accept one `.human.replay.json.gz` file as their input.
Use `data.replay_codec.load_replay` for the full replay dictionary
with expanded viewer states and unchanged conversations. Inspect files with the
[web replay viewer](https://botcs.github.io/reason-to-play/replay.html).

EMPA theory regressors are a separate model input at
`features/empa/theory-regressors.json.gz`; they are not human behaviour.

## Replay source roles

| Source | Meaning | Included in the paper release |
| --- | --- | --- |
| `human` | Recorded participant actions with action-only prompts. | Yes |
| `generative` | Model gameplay with its observations, actions and conversation. | Yes |
| `imputed` | Human actions with generated rationales in gameplay format. | No |
| `narration` | Imputation conversation, including action hints. | No |

The imputation code remains available for new experiments. Model-generated
rationale is not a measurement of a participant's thoughts.

## Neural inputs

`neural/sub-XX/` contains `bold.npz` (voxel-by-sample values, mask and affine),
`samples.npz` (play identities, scanner runs, timing and ordered boundaries),
`nuisance.npz` (buttons, scores and time variables) and sample-by-feature arrays
under `model-features/`. DDQN, EfficientZero, LRM and EMPA have separate files.
NPZ keys describe the arrays; no pickle is needed to read these inputs.

Each feature/nuisance `.npz.alignment.json` uses
`reason-to-play/alignment-binding`, version 2. Its `feature_sha256`,
`bold_sha256` and `samples_sha256` hash the exact compressed file bytes;
`sample_order_sha256` hashes the recorded sample identities and timing fields.
The BOLD association uses `reason-to-play/bold-samples`, version 1, with the
same BOLD/sample hashes. Readers reject missing or conflicting associations.

`feature_coverage` lists missing play IDs and sample intervals, or explicitly
states unknown coverage. `feature_coverage_by_layer` preserves distinct layer
availability where required, including EfficientZero hooks. Coverage does not
change the fitting mask. See the [analysis guide](guides/dataset-analysis.md)
for direct Python commands and the numerical policy.
