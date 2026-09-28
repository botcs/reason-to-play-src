# Tomov et al. (2023): behavioral data notes

The document below is preserved verbatim from
[`botcs/RC_RL` at commit `6186f36313be956fe2ea35f85a0a70ea31c5b387`](https://github.com/botcs/RC_RL/blob/6186f36313be956fe2ea35f85a0a70ea31c5b387/VGFMRI_DB_README.md).
That repository derives from the original study's
[`tomov/RC_RL`](https://github.com/tomov/RC_RL) code. This reference describes the
upstream MongoDB/BSON records associated with
[OpenNeuro ds004323 v1.0.0](https://openneuro.org/datasets/ds004323/versions/1.0.0).
It is not the specification of the public Reason to Play replay format.
The unchanged source document has SHA-256
`ffa22e92ac1faa04f6d7b8fdb89d7884e4ddad71f5f998a3ff1a2544550c166b`.
Paths and example commands in its body refer to the source repository.

The public human artifact is one self-contained JSON file per participant, game
and prompt condition, spanning that game's levels, attempts and scanner runs.
It contains the trajectory, exact prompts and identities/timing needed for the
supported analyses. See the [data-format guide](../data-format.md) for the
current schema. Public analysis reads these files without BSON or MongoDB.

This historical reference supports independent source-data archaeology. Raw
MRI preprocessing is documented separately in the
[fMRI guide](../guides/fmri-preprocessing.md).

## Corrections to use when reading the historical document

The source body is retained unchanged, including statements superseded by
inspection of the recorded data:

- **Control and frame clocks:** decoded `zkeystates` is a per-frame control
  stream, not one entry per play or a sparse action list. Its entries include
  `gt`, timestamps and RNG state, and may include null entries. The claims that
  `gt` is always 1 and timestamps are unique only per play are incorrect.
  Observed state frames use `states[i]["gt"] == i`; control timestamps can
  differ from state timestamps. Sparse `actions` cannot be paired with frames
  by array index.
- **Outcomes:** the original play's `win` is three-valued: true, false or null.
  Null includes avatar deaths as well as incomplete plays. Check the recorded
  avatar-death event before classifying null as incomplete. A terminal state's
  `win=-1` is not a substitute for the play outcome. The quoted 92.9% win rate
  uses only the non-null outcomes, not all 7,034 plays.
- **Seeds and replay:** the source includes `subjects.seed` and recorded
  `RNG_state` values in the control stream. Claims that no seed was stored or
  that exact reconstruction is categorically impossible are too strong.
  Their presence alone does not prove exact replay in another engine version;
  that requires a separate validation. Recorded trajectories and play outcomes
  remain the evidence for what participants experienced.

---

The original document begins below; its bytes have not been edited.

# VGFMRI MongoDB Database Schema Documentation

## Overview

The `vgfmri` MongoDB database contains behavioral and neural data from an fMRI experiment where subjects played VGDL (Video Game Description Language) video games. The database stores game definitions, subject information, gameplay episodes, scanner run metadata, and model-based regressors for fMRI analysis.

**Database**: `vgfmri`
**MongoDB URI**: `mongodb://localhost:27017/vgfmri`

---

## Collections Summary

| Collection | Count | Purpose |
|------------|-------|---------|
| **subjects** | 32 | Subject registry and experimental structure |
| **runs** | 204 | Scanner run sessions (avg 6.4 per subject) |
| **games** | 229 | Game definitions in VGDL format |
| **plays** | 7,034 | Individual gameplay episodes |
| **plays_post** | 6,652 | Detailed post-processed gameplay states |
| **regressors** | 5,241 | EMPA model regressors for fMRI analysis |
| **dqn_regressors_25M** | 6,653 | DQN model regressors for fMRI analysis |
| **sim_results** | 693 | Simulation/agent performance results |

---

## Detailed Schema

### 1. `subjects` Collection

Registry of experimental subjects with their complete experimental structure.

**Count**: 32 subjects

**Key Fields**:
- `_id`: ObjectId - Primary key
- `subj_id`: String - Subject identifier ("1", "2", ..., "32")
- **`seed`**: Number - **Subject-specific random seed** (used for color/symbol assignment and RNG initialization)
- `ts`: Number - Unix timestamp when subject record was created
- `dt`: Date - Datetime when subject record was created
- `games`: Array - Game definitions with subject-specific customizations (colors, symbols, fake names)
  - Same structure as `games` collection, but with additional fields:
  - `fake_name`: String - Subject-specific anonymized game name
  - `alphabet`: Array[Number] - Subject-specific symbol encoding
  - `bg_color`: Array[Number] - Subject-specific background color [r, g, b]
- `runs`: Array - Complete experimental design for this subject
  - `run_id`: Number - Run number (0-7)
  - `prerun_interval`: Number - Pre-run rest period (seconds)
  - `postrun_interval`: Number - Post-run rest period (seconds)
  - `blocks`: Array - Game blocks within this run
    - `block_id`: Number - Block number
    - `game_id`: Number - Game identifier
    - `game`: Object - Embedded game definition
      - `name`: String - Game name (e.g., 'vgfmri3_sokoban')
      - `descs`: Array[String] - VGDL game descriptions
      - `levels`: Array[String] - Level layouts
      - `alphabet`: Array[Number] - Character encoding for display
      - `fake_name`: String - Anonymized game name shown to subjects
      - `bg_color`: Array[Number] - Background color [r, g, b]
    - `interblock_interval`: Number - Rest between blocks
    - `instances`: Array - Level instances to play
      - `instance_id`: Number
      - `duration`: Number - Time limit (seconds)
      - `desc_id`: Number - Which game description to use
      - `level_id`: Number - Which level to play
      - `interplay_interval`: Number - Rest between plays

**Purpose**: Defines the complete experimental structure. Each subject document contains their full experimental design including all runs, blocks, games, and levels they will encounter.

**Example Query**:
```javascript
// Get subject 1's experimental structure
db.subjects.findOne({}, {runs: 1})
```

---

### 2. `runs` Collection

Metadata for individual scanner run sessions.

**Count**: 204 runs (avg 6.4 per subject, range: 5-8)

**Key Fields**:
- `_id`: ObjectId - Primary key
- `subj_id`: String - Subject identifier ("1", "2", ...)
- `run_id`: Number - Run number for this subject (0-7)
- `scan_start_ts`: Number - Unix timestamp when scan started
- `scan_start_dt`: Date - Datetime when scan started
- `end_time`: Number - Unix timestamp when run ended
- `prerun_interval`: Number - Pre-run rest period (seconds)
- `postrun_interval`: Number - Post-run rest period (seconds)
- `postrun_interval_start_time`: Number - When post-run rest started
- `blocks`: Array - Game blocks in this run (same structure as in subjects)
  - Each block also includes:
  - `start_time`: Number - When this block started
  - `instances`: Array - Contains actual play references
    - `play_keys`: Array[ObjectId] - References to `plays._id` for this instance
    - `start_time`: Number - When this instance started
    - `end_time`: Number - When this instance ended
- `subj_key`: ObjectId - Reference to subjects collection

**Purpose**: Records actual scanner session timing and structure.

**Relationships**:
- Links to `subjects` via `subj_id`
- Links to `plays` via `(subj_id, run_id)` composite key

**Example Query**:
```javascript
// Get all runs for subject 1
db.runs.find({subj_id: "1"})

// Get plays for a specific run
db.plays.find({subj_id: "1", run_id: 0})
```

---

### 3. `games` Collection

Game definitions in VGDL format.

**Count**: 229 game definitions (13 unique games actually played in experiment)

**Key Fields**:
- `_id`: ObjectId - Primary key
- `name`: String - Game name (e.g., "vgfmri3_sokoban", "vgfmri4_lemmings")
- `descs`: Array[String] - VGDL game descriptions (usually 1 per game, defines rules)
- `levels`: Array[String] - Level layouts (ASCII art maps)
- `__v`: Number - Version number

**Purpose**: Stores all game definitions. Each game has:
- **Game descriptions** (`descs`): VGDL code defining game mechanics, sprites, interactions
- **Levels** (`levels`): ASCII layouts defining sprite positions

**Important Notes**:
- Many games have **stochastic elements** (RandomNPC, Chaser, Missile sprites)
- Games with `Chaser`, `RandomNPC`, or `Missile` are **non-deterministic**
- Most games have 1 game description for all levels (rules stay constant)
- Number of levels varies by game (e.g., sokoban has 12 levels)

**Games Played in Experiment**:
- vgfmri3_bait (deterministic)
- vgfmri3_chase (stochastic: Chaser)
- vgfmri3_helper (stochastic: Chaser)
- vgfmri3_lemmings (stochastic: Chaser)
- vgfmri3_plaqueAttack (stochastic: Chaser)
- vgfmri3_sokoban (deterministic)
- vgfmri3_zelda (stochastic: RandomNPC)
- vgfmri4_avoidgeorge (stochastic: RandomNPC, Chaser)
- vgfmri4_bait (deterministic)
- vgfmri4_chase (stochastic: Chaser)
- vgfmri4_helper (stochastic: Chaser)
- vgfmri4_lemmings (stochastic: Chaser)
- vgfmri4_zelda (stochastic: RandomNPC)

**Example Query**:
```javascript
// Get game definition
db.games.findOne({name: "vgfmri3_sokoban"})

// Count levels in a game
db.games.findOne({name: "vgfmri3_sokoban"}, {levels: 1}).levels.length
```

---

### 4. `plays` Collection

**PRIMARY DATA**: Individual gameplay episodes with actions and outcomes.

**Count**: 7,034 plays
- **Wins**: 3,793 (92.9%)
- **Losses**: 291 (7.1%)
- **Incomplete/Timeout**: Remaining plays

**Key Fields** (complete list):
- `_id`: ObjectId - Primary key
- `subj_id`: String - Subject identifier ("1", "2", ...)
- `run_id`: Number - Run number (0-7)
- `block_id`: Number - Block number within run
- `play_id`: Number - Play number within block (0-indexed within instance)
- `instance_id`: Number - Instance identifier (0-2 per block)
- `game_id`: Number - Game identifier (0-6)
- `game_name`: String - Game name (e.g., "vgfmri3_sokoban")
- `desc_id`: Number - Which game description was used (usually 0)
- `level_id`: Number - Which level was played (0-11)
- `level_str`: String - Level layout (ASCII)
- `game_str`: String - VGDL game description used
- `fake_name`: String - Anonymized game name shown to subject
- **`win`**: Boolean - **TRUE if won, FALSE if lost** (ground truth)
- **`score`**: Number - **Final score** (ground truth)
- **`actions`**: Array[[String, Number]] - **List of [action_name, timestamp] pairs**
  - Action names: "spacebar", "up", "down", "left", "right"
  - Timestamps are Unix time with millisecond precision
- `keydowns`: Object - Map of key names to arrays of press timestamps
- `keyups`: Object - Map of key names to arrays of release timestamps
- `keyholds`: Object - Map of key names to arrays of [start_time, duration] pairs
- `start_time`: Number - Unix timestamp when play started
- `end_time`: Number - Unix timestamp when play ended
- `run_start_ts`: Number - Unix timestamp when run started
- `run_start_dt`: Date - Datetime when run started
- `events`: Array - Game events (interactions, effects) with timestamps
  - Each event: `{agentAction, agentState, effectList, ts, dt}`
- **`zstates`**: Binary - **Compressed game states** (see [Decoding zstates and zkeystates](#decoding-zstates-and-zkeystates))
  ```
  Binary (zlib compressed) → BSON decoded → {
    "states": [                    # List of states, one per timestep (1.67M total across all 7K plays)
      {
        "collision_effLen": int,   # Unique values: 9
        "dt": datetime,            # Datetime object (1.67M unique timestamps)
        "effectList": list|list[list], # List of effects that occurred
        "effectListByClass": list|list[list],
        "effectListByColor": list|list[list],
        "effectListLen": int,      # Unique values: 235
        "ended": bool,             # Whether game has ended (True/False)
        "entropy": null,           # Always null
        "gt": int,                 # Game tick 0-1734 (timestep number within play)
        "keyPressType": null|str,  # 'spacebar'|'up'|'down'|'left'|'right' or null
        "keystate": [int],         # Length: 323 (pygame 1.x key codes 0-322)
                                   # Index = pygame key code, value = pressed state
        "kill_listLen": int,       # Unique values: 142
        "kill_list_ID": list|list[Binary], # UUIDs of killed sprites
        "list": [str],             # Human-readable sprite list
        "new_spritesLen": int,     # Unique values: 84
        "new_sprites_ID": list|list[Binary], # UUIDs of newly created sprites
        "objects": {               # Sprite objects by type
          "sprite_type": {         # e.g., "avatar", "wall", "enemy"
            "(x, y)": {            # Position tuple as string key
              "ID": Binary,        # UUID v1 (693K unique across all states)
              "ID2": Binary,       # Copy of ID (693K unique)
              "_age": int,         # 0-20 (RandomNPC/Chaser only)
              "autotiling": bool|str, # True or 'true'
              "color": [int, int, int],
              "colorName": str,    # 16 colors: RED, DARKGRAY, GREEN, DARKBLUE, etc.
              "cons": int,         # 1,2,6,8,12 (text sprites only)
              "counter": int,      # 0-6 (some sprites)
              "direction": null,   # Always null
              "fontsize": int,     # 14 or 24 (text sprites only)
              "frameRate": int,    # Always 8
              "img_path": null,    # Always null
              "lastdisplacement": int, # 0-1733 (timesteps since last move)
              "limit": int,        # 1 or 5 (resource sprites only)
              "offset": [int, int],
              "orientation": [int, int]|[float, float], # Direction vector
              "rect": {
                "pos": [int, int],
                "size": [int, int]
              },
              "resources": dict,   # Resource counters (key sprite types)
              "rotateInPlace": bool, # Always False (missile sprites only)
              "spawnCooldown": int, # Always 1 (spawner sprites only)
              "speed": null|int|float, # null, 0, 1, 0.15, 0.25
              "symbol": str,       # 151 unique symbols (Unicode game icons)
              "x": int,            # 0-479 (pixel x-coordinate)
              "y": int             # 0-304 (pixel y-coordinate)
            }
          }
        },
        "observe_state": bool,     # Always False
        "score": int|float,        # Current score (4519 unique values)
        "sprite_groupsLen": int,   # 4-11 sprite type groups
        "ts": float,               # Unix timestamp (1.67M unique)
        "win": null|bool|int       # null, True, False, or -1 (loss)
      }
    ]
  }
  ```
- **`zkeystates`**: Binary - **Compressed key states** (see [Decoding zstates and zkeystates](#decoding-zstates-and-zkeystates))
  ```
  Binary (zlib compressed) -> BSON decoded -> {
    "keystates": [                # List of keystates, one per play (7034 total)
      null,                       # First keystate is usually null
      {
        "RNG_state": [            # Python random.getstate() - Mersenne Twister MT19937
          int,                    # Version (always 3)
          [int],                  # Length: 625 (MT19937: 624 state + 1 position)
          null                    # Gauss next (always null)
        ],
        "dt": datetime,           # Datetime object (7034 unique, one per play)
        "gt": int,                # Game tick (always 1 - captured at play start)
        "keyPressType": null|str, # 'spacebar'|'up'|'down'|'left'|'right' or null
        "keydowns": dict,         # Key down events
        "keystate": [int],        # Length: 323 (pygame 1.x key codes 0-322)
                                  # Index = pygame key code, value = pressed state
        "keyups": dict,           # Key up events
        "ts": float               # Unix timestamp (7034 unique, one per play)
      }
    ]
  }
  ```

**Purpose**: **Ground truth data** for human gameplay. Each document represents one play-through of a level.

**CRITICAL NOTES**:
- **`win` and `score` are reliable** - recorded during actual gameplay
- **`actions` cannot be reliably replayed** for stochastic games due to randomness
- For games with RandomNPC/Chaser/Missile, replay will produce different outcomes
- Random seed was not stored, making exact replay impossible for stochastic games

**Relationships**:
- Links to `runs` via `(subj_id, run_id)`
- Links to `games` via `game_name`
- Referenced by `regressors`, `dqn_regressors_25M`, `plays_post` via `_id`

**Example Queries**:
```javascript
// Get all plays for subject 2 playing lemmings
db.plays.find({subj_id: "2", game_name: "vgfmri3_lemmings"})

// Count wins/losses per game
db.plays.aggregate([
  {$group: {
    _id: {game: "$game_name", win: "$win"},
    count: {$sum: 1}
  }}
])

// Get subject's performance on a specific level
db.plays.find({
  subj_id: "2",
  game_name: "vgfmri3_sokoban",
  level_id: 0
}, {win: 1, score: 1, actions: 1})
```

---

### 5. `plays_post` Collection

Detailed post-processed gameplay data with per-timestep information.

**Count**: 6,652 plays (subset of `plays` with post-processing)

**Key Fields**:
- `_id`: ObjectId - Primary key
- `play_key`: ObjectId - **Reference to `plays._id`**
- `subj_id`: String - Subject identifier
- `block_id`, `run_id`, `play_id`, `level_id`, `instance_id`: Numbers - Identifiers
- `game_name`: String - Game name
- `type`: String - Processing type
- `play_post_ts`: Number - Processing timestamp
- `play_post_dts`: String - Processing datetime
- `grid_size`: Number - Game grid size
- Per-timestep arrays (all same length, one entry per game tick):
  - `timestamps`: Array[Number] - Timestamp for each game state
  - `state_timestamps`: Array[Number] - State recording times
  - `keystate_timestamps`: Array[Number] - Key state times
  - `keypresses`, `keyholds`, `keydowns`, `keyups`: Objects - Input data
  - `score`: Array[Number] - Score at each timestep
  - `win`: Array[Boolean] - Win state at each timestep
  - `ended`: Array[Boolean] - Whether game ended
  - `sprites`: Array - Sprite information per timestep
  - `sprite_groups`: Array - Sprite groupings
  - `movable`, `moved`, `changed`: Array - Sprite state changes
  - `new_sprites`, `killed_sprites`: Array - Sprite creation/destruction
  - `non_walls`: Array - Non-wall sprites
  - `collisions`: Array - Collision events
  - `effects`, `effectsByCol`: Array - Interaction effects
  - `avatar_moved`, `avatar_collision_flag`: Array - Avatar-specific flags
- Model-based metrics (EMPA theory):
  - `I_len`, `Ip_len`, `Igen_len`: Array - Information theory metrics
  - `T_len`, `Tnov_len`: Array - Termination theory metrics
  - `S_len`: Array - State metrics
  - `dI_len`, `dT_len`, `dS_len`: Array - Deltas
  - `dIgen_len`, `dIp_len`, `dTnov_len`: Array - Generative deltas
  - `likelihood`, `sum_lik_play`: Array - Likelihood metrics
  - `surprise`: Array - Surprise signal
  - `newTimeStep_flag`, `termination_change_flag`, `interaction_change_flag`: Array - Change detection

**Purpose**: Frame-by-frame analysis of gameplay for computational modeling and fMRI regressor generation.

**Relationships**:
- **Links to `plays`** via `play_key = plays._id`

**Example Query**:
```javascript
// Get detailed post-processing for a specific play
var play = db.plays.findOne({subj_id: "2", play_id: 0});
db.plays_post.findOne({play_key: play._id})
```

---

### 6. `regressors` Collection

EMPA (Episodic Model-based Planning Agent) model regressors for fMRI analysis.

**Count**: 5,241 regressors

**Key Fields**:
- `_id`: ObjectId - Primary key
- `play_key`: ObjectId - **Reference to `plays._id`**
- `subj_id`: String - Subject identifier
- `run_id`, `block_id`, `play_id`, `instance_id`, `level_id`: Numbers - Identifiers
- `game_name`: String - Game name
- `type`: String - Regressor type
- `ts`: Number - Timestamp
- `reg_ts`: Number - Regressor timestamp
- `dt`: Date - Datetime
- `reg_dts`: String - Regressor datetime string
- `regressors`: Object - Model-based regressor values
  - Contains theory-based signals (e.g., information gain, prediction error, value, etc.)

**Purpose**: Model-based fMRI regressors from EMPA theory for neural analysis.

**Relationships**:
- **Links to `plays`** via `play_key = plays._id`

**Example Query**:
```javascript
// Get regressors for a specific play
var play = db.plays.findOne({subj_id: "2", play_id: 0});
db.regressors.find({play_key: play._id})
```

---

### 7. `dqn_regressors_25M` Collection

DQN (Deep Q-Network) model regressors for fMRI analysis.

**Count**: 6,653 regressors

**Key Fields**: (Same structure as `regressors`)
- `_id`: ObjectId - Primary key
- `play_key`: ObjectId - **Reference to `plays._id`**
- `subj_id`, `run_id`, `block_id`, `play_id`, `instance_id`, `level_id`: Identifiers
- `game_name`: String - Game name
- `type`: String - Regressor type
- `regressors`: Object - DQN model regressor values
  - Contains RL-based signals (Q-values, RPE, value, etc.)

**Purpose**: RL model-based fMRI regressors from DQN agent (trained for 25M steps) for neural analysis.

**Relationships**:
- **Links to `plays`** via `play_key = plays._id`

**Example Query**:
```javascript
// Get DQN regressors for a specific play
var play = db.plays.findOne({subj_id: "2", play_id: 0});
db.dqn_regressors_25M.find({play_key: play._id})
```

---

### 8. `sim_results` Collection

Agent simulation performance results.

**Count**: 693 simulation runs

**Key Fields**:
- `_id`: ObjectId - Primary key
- `agent_name`: String - Name of agent (e.g., "DQN", "EMPA")
- `subj_id`: String - Subject being compared to
- `game_name`: String - Game played
- `tag`: String - Experimental tag/version
- `ts`: Number - Timestamp
- `dt`: Date - Datetime
- `results`: Array - Performance results
  - Contains metrics like wins, scores, steps per level

**Purpose**: Stores agent performance for comparison with human behavior.

**Example Query**:
```javascript
// Get DQN agent results for lemmings game
db.sim_results.find({agent_name: "DQN", game_name: "vgfmri3_lemmings"})
```

---

## Key Relationships

```
subjects (32)
    └─> runs (204)
           ├─> (subj_id, run_id) link
           └─> plays (7,034)
                  ├─> game_name link to games (229)
                  ├─> _id = play_key in plays_post (6,652)
                  ├─> _id = play_key in regressors (5,241)
                  └─> _id = play_key in dqn_regressors_25M (6,653)

games (229)
    └─> name matches plays.game_name
```

**Primary Keys**:
- Each collection uses MongoDB ObjectId (`_id`)

**Foreign Key Relationships**:
- `subjects` → `runs`: via `subj_id`
- `runs` → `plays`: via `(subj_id, run_id)` composite
- `plays` → `games`: via `game_name`
- `plays` → `plays_post`: via `plays._id = plays_post.play_key`
- `plays` → `regressors`: via `plays._id = regressors.play_key`
- `plays` → `dqn_regressors_25M`: via `plays._id = dqn_regressors_25M.play_key`

---

## Data Statistics

### Subjects
- **Total**: 32 subjects
- **Subject IDs**: "1" through "32" (stored as strings, but range 1-9 are single digit)
- **Runs per subject**: 5-8 runs (average: 6.4)

### Games
- **Total game definitions**: 229
- **Unique games in experiment**: 13
- **Game types**:
  - Deterministic: bait, sokoban
  - Stochastic: chase, helper, lemmings, plaqueAttack, zelda, avoidgeorge
- **Levels per game**: Varies (e.g., sokoban has 12 levels)

### Plays
- **Total episodes**: 7,034
- **Win rate**: 92.9% (3,793 wins)
- **Loss rate**: 7.1% (291 losses)
- **Actions per play**: Varies (stored in `actions` array)

### Processing
- **plays_post**: 6,652 / 7,034 plays processed (94.6%)
- **regressors**: 5,241 plays with EMPA regressors
- **dqn_regressors_25M**: 6,653 plays with DQN regressors

---

## Important Implementation Notes

### 1. **Stochastic Games Cannot Be Reliably Replayed**

Many games contain stochastic elements (RandomNPC, Chaser, Missile sprites) that move randomly. These games include:
- All "chase", "helper", "lemmings", "zelda", "avoidgeorge" variants
- Any game with enemy AI

**Problem**: Although a subject-level seed exists in `subjects.seed`, this seed is used ONLY for initializing color and symbol assignments. The random state for stochastic sprite behavior (RandomNPC, Chaser, Missile) is NOT reset per play - it evolves continuously across all plays in a session. Therefore, to replay play #N accurately, you would need to replay ALL previous plays #1 through #N-1 first to restore the exact random state. This makes isolated play replay unreliable.

**Solution**: For analysis, always use the **stored values** in the database:
- Use `plays.win` (boolean) for win/loss
- Use `plays.score` (number) for score
- Use `len(plays.actions)` for step count

**DO NOT** replay actions and expect to get the same outcome for stochastic games.

### 2. **Subject ID Format**

Subject IDs are stored as **strings** ("1", "2", ..., "32"), NOT integers.

```javascript
// CORRECT
db.plays.find({subj_id: "2"})

// WRONG
db.plays.find({subj_id: 2})
```

### 4. **Action Format**

Actions are stored as arrays of `[action_name, timestamp]` pairs:
```javascript
actions: [
  ["spacebar", 1583937483.789612],
  ["up", 1583937484.019059],
  ["spacebar", 1583937484.480746]
]
```

Valid action names: `"spacebar"`, `"up"`, `"down"`, `"left"`, `"right"`

### 4b. **Simultaneous Key Priority (keystate decoding)**

Humans frequently hold two arrow keys simultaneously (e.g. DOWN+RIGHT during
a diagonal traverse). The VGDL engine resolves simultaneous presses in
`MovingAvatar._readMultiActions` (ontology.py) with two independent if/elif
blocks -- horizontal first, then vertical:

```python
if   keystate[K_RIGHT]: res += [RIGHT]
elif keystate[K_LEFT]:  res += [LEFT]
if   keystate[K_UP]:    res += [UP]
elif keystate[K_DOWN]:  res += [DOWN]
return res[0]   # _readAction takes the first element
```

Effective priority: **RIGHT > LEFT > UP > DOWN**.
Horizontal axis is checked before vertical; `res[0]` picks horizontal when
both axes are active.

Cross-referenced against 2956 multi-key frames across 5 subjects with
0 real mismatches (remaining ~20% are wall-blocked moves where the avatar
could not move in the predicted direction).

**WARNING**: `keyPressType` in zstates is derived via `keystate.index(1)`
which returns the *lowest pygame key code* -- a different priority
(UP > DOWN > RIGHT > LEFT). Do NOT use `keyPressType` to determine the
action the engine actually processed; decode from `keystate` using the
priority above.

### 5. **Game Name Conventions**

- Format: `vgfmri{VERSION}_{GAME_NAME}`
- Version 3: `vgfmri3_*` (earlier experiment)
- Version 4: `vgfmri4_*` (later experiment)
- Base game names can be extracted by removing the prefix

### 6. **Missing Data**

Not all plays have complete post-processing:
- `plays`: 7,034 total
- `plays_post`: 6,652 (382 missing)
- `regressors`: 5,241 (1,793 missing)
- `dqn_regressors_25M`: 6,653 (381 missing)

---

## Common Query Patterns

### Get all plays for a subject and game
```javascript
db.plays.find({
  subj_id: "2",
  game_name: "vgfmri3_lemmings"
}).sort({start_time: 1})
```

### Get win/loss breakdown by level
```javascript
db.plays.aggregate([
  {$match: {game_name: "vgfmri3_sokoban"}},
  {$group: {
    _id: {level: "$level_id", win: "$win"},
    count: {$sum: 1}
  }},
  {$sort: {"_id.level": 1}}
])
```

### Get complete data for a play (with regressors)
```javascript
var play = db.plays.findOne({subj_id: "2", play_id: 0});
var play_post = db.plays_post.findOne({play_key: play._id});
var regressors = db.regressors.find({play_key: play._id}).toArray();
var dqn_regressors = db.dqn_regressors_25M.find({play_key: play._id}).toArray();
```

### List all games a subject played
```javascript
db.plays.distinct("game_name", {subj_id: "2"})
```

### Get subject's learning curve (wins over time)
```javascript
db.plays.find(
  {subj_id: "2", game_name: "vgfmri3_sokoban"},
  {level_id: 1, win: 1, score: 1, start_time: 1}
).sort({start_time: 1})
```

---

## Analysis Scripts

### Query outcomes for a subject/game
```bash
python scripts/query_game_outcomes.py --subject 2 --game vgfmri3_lemmings
```

### Plot population metrics (from DB stats)
```bash
python scripts/plot_population_metrics_from_stats.py \
  --out plots/population-curriculum/from-stats/
```

### Plot population metrics (from simulation - NOT recommended for stochastic games)
```bash
python scripts/plot_population_metrics.py \
  --games-source db \
  --out plots/population-curriculum/from-simulation/
```

---

## Data Collection Details

### Experimental Design
- **32 subjects** participated
- Each subject completed **5-8 scanner runs**
- Each run contained multiple **game blocks**
- Each block had multiple **level instances**
- **~220 plays per subject** on average

### Games
- **13 unique games** from VGDL game library
- Mix of **puzzle** (sokoban, bait), **chase** (chase, lemmings), and **exploration** (zelda) games
- Games selected to vary in complexity and mechanics

### Data Storage
- All gameplay recorded at high temporal resolution
- Actions captured with millisecond precision
- Game states compressed in `zstates` field
- Post-processing generates per-frame metrics

---

## References

For more information about the VGDL framework and games:
- VGDL framework: See `vgdl/` directory
- Game definitions: See `all_games/` directory or `games` collection
- Original dataset: OpenNeuro ds004323

---

## Decoding zstates and zkeystates

The `plays` collection stores two important binary fields that contain the complete game state trajectory and key press information for each play: `zstates` and `zkeystates`. These fields are compressed using zlib and encoded using BSON.

### What are zstates?

`zstates` contains the **complete game state at every timestep** during gameplay. This includes:
- Positions and attributes of all sprites (avatar, enemies, objects, walls, etc.)
- Game score at each timestep
- Whether the game has ended
- Effect list (interactions that occurred)
- Key state (which keys were pressed)

**Important**: For **stochastic games** (those with RandomNPC, Chaser, Missile sprites), `zstates` provides the **actual state sequence** that occurred during human gameplay. This is the **ground truth** trajectory, which cannot be reliably reproduced by replaying actions due to missing random seeds.

### What are zkeystates?

`zkeystates` contains the **key press state at every timestep**, synchronized with the game states in `zstates`. This provides precise timing information about which keys were held down at each game tick.

### Decoding Method

Both `zstates` and `zkeystates` use the same compression scheme:

1. **BSON encoding** - The Python dict/object is first encoded to BSON format
2. **zlib compression** - The BSON bytes are then compressed with zlib

To decode them, reverse this process:

```python
import zlib
import bson

# Decode zstates
def decompress_zstates(zstates_binary):
    """
    Decompress zstates from the database.

    Args:
        zstates_binary: Binary data from plays.zstates field

    Returns:
        dict with 'states' key containing list of game states
    """
    decompressed = zlib.decompress(zstates_binary)
    data = bson.decode(decompressed)
    states = data['states']  # List of state dicts, one per timestep
    return states

# Decode zkeystates
def decompress_zkeystates(zkeystates_binary):
    """
    Decompress zkeystates from the database.

    Args:
        zkeystates_binary: Binary data from plays.zkeystates field

    Returns:
        dict with 'keystates' key containing list of keystate dicts
    """
    decompressed = zlib.decompress(zkeystates_binary)
    data = bson.decode(decompressed)
    keystates = data['keystates']  # List of keystate dicts, one per timestep
    return keystates
```

### State Structure

Each state in the decompressed `states` list contains:

```python
{
    'objects': {
        'sprite_type_name': {
            'position_tuple': {
                'attribute_name': value,
                # Sprite attributes like:
                # - 'rect': [left, top, width, height]
                # - 'orientation': [dx, dy]
                # - 'resources': {...}
                # - 'color': [r, g, b, alpha]
                # - Various game-specific attributes
            }
        }
    },
    'keystate': {key_code: boolean},  # Which keys are pressed
    'score': float,
    'ended': boolean,
    'effectList': [...],
    'num': int  # timestep number
}
```

### Keystate Structure

Each keystate in the decompressed `keystates` list is a dict mapping pygame key codes to boolean values:

```python
{
    K_UP: False,
    K_DOWN: False,
    K_LEFT: False,
    K_RIGHT: True,  # Right arrow is pressed
    K_SPACE: False
}
```

### Example Usage

```python
from pymongo import MongoClient
import zlib
import bson

# Connect to database
client = MongoClient('mongodb://localhost:27017')
db = client['vgfmri']

# Get a play
play = db.plays.findOne({'subj_id': '1', 'game_name': 'vgfmri3_sokoban'})

# Decompress states
zstates = play['zstates']
decompressed = zlib.decompress(zstates)
data = bson.decode(decompressed)
states = data['states']

print(f"Number of timesteps: {len(states)}")
print(f"First state: {states[0]}")

# Decompress keystates
zkeystates = play['zkeystates']
decompressed_keys = zlib.decompress(zkeystates)
data_keys = bson.decode(decompressed_keys)
keystates = data_keys['keystates']

print(f"Number of keystate timesteps: {len(keystates)}")
```

### Replaying from States (Correct Method)

**For stochastic games**, to get the actual trajectory the human experienced, you MUST use the states from `zstates`, NOT replay the actions:

```python
from vgdl.core import VGDLParser
from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings

# Load game
game_str = play['game_str']
level_str = play['level_str']
rle = createRLInputGameFromStrings(game_str, level_str)

# Decompress states
states = decompress_zstates(play['zstates'])

# Replay by setting each state
for state in states:
    rle.setFullState(state)
    rle._game._drawAll()  # Visualize if needed
    # Process/analyze state...
```

This is the **only reliable way** to replay stochastic games and get the exact trajectory that occurred during human gameplay.

### Reference Implementation

The reference implementation for decoding and replaying these fields can be found in:
- [py-vgdl-reference/vgdl/core.py](https://github.com/tsividis/vgdl/blob/refactor_fMRI_cannon/vgdl/core.py#L140-145) - `compress()` and `decompress()` methods
- [py-vgdl-reference/fmri_replay.py](https://github.com/tsividis/vgdl/blob/refactor_fMRI_cannon/fmri_replay.py) - Example replay script
- [py-vgdl-reference/fmri_agentReplay.py](https://github.com/tsividis/vgdl/blob/refactor_fMRI_cannon/fmri_agentReplay.py#L185-192) - Agent replay using states

A local copy of this reference repository is available in `py-vgdl-reference/` (cloned from the `refactor_fMRI_cannon` branch).

---

## Version History

**Current Schema**: As of 2020-03-11 (earliest play timestamp)

**Notes**:
- Database was populated during fMRI data collection in 2020
- Post-processing (`plays_post`, `regressors`) added later
- Schema has remained stable since collection
