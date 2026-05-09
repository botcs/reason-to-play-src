# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
ZstateAdapter: Convert Tomov 2023 zstate dicts to game engine format.

Bridges the gap between human replay BSON data and the unified Harness pipeline.
The adapter converts zstate frames (from decompressed plays.zstates) into the
same 2D grid + event tuples that the game engine produces via env.step().

This allows ObservationFormatter and EventLogger to operate identically on both
generative gameplay and human replay data.
"""

# Action index constants matching the game engine
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NOOP = 4
ACTION_SPACE = 5

# Map keyPressType strings to action indices
_KEY_TO_ACTION = {
    "up": ACTION_UP,
    "down": ACTION_DOWN,
    "left": ACTION_LEFT,
    "right": ACTION_RIGHT,
    "spacebar": ACTION_SPACE,
}

# Pygame 1.x key codes.
_K_UP = 273
_K_DOWN = 274
_K_LEFT = 276
_K_RIGHT = 275
_K_SPACE = 32

# All action key codes for is_action_frame().
_ACTION_KEY_CODES = {_K_UP, _K_DOWN, _K_LEFT, _K_RIGHT, _K_SPACE}


def action_from_keystate(keystate: list[int]) -> int:
    """Derive action index from pygame keystate array.

    Matches the priority of the original Tomov VGDL engine's
    MovingAvatar._readMultiActions (ontology.py):

        if   keystate[K_RIGHT]: res += [RIGHT]
        elif keystate[K_LEFT]:  res += [LEFT]
        if   keystate[K_UP]:    res += [UP]
        elif keystate[K_DOWN]:  res += [DOWN]
        return res[0]

    Two independent if/elif blocks: horizontal (RIGHT > LEFT) is checked
    before vertical (UP > DOWN). _readAction returns res[0], so horizontal
    wins when both axes are pressed. SPACE is checked last.

    Validated against 2956 multi-key frames across 5 subjects: 0 mismatches
    (the remaining 20.8% are wall-blocked moves where actual=NONE).
    """
    # Horizontal axis (checked first -> wins on simultaneous press)
    if _K_RIGHT < len(keystate) and keystate[_K_RIGHT]:
        return ACTION_RIGHT
    if _K_LEFT < len(keystate) and keystate[_K_LEFT]:
        return ACTION_LEFT
    # Vertical axis
    if _K_UP < len(keystate) and keystate[_K_UP]:
        return ACTION_UP
    if _K_DOWN < len(keystate) and keystate[_K_DOWN]:
        return ACTION_DOWN
    # Space (fire / use)
    if _K_SPACE < len(keystate) and keystate[_K_SPACE]:
        return ACTION_SPACE
    return ACTION_NOOP


def is_action_frame(zstate: dict) -> bool:
    """Check if a zstate frame has any game-relevant key pressed."""
    keystate = zstate.get("keystate", [])
    return action_from_keystate(keystate) != ACTION_NOOP


# Action index to name (for logging)
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "ACTION"]


class ZstateAdapter:
    """Convert Tomov 2023 zstate dicts to game engine format.

    The engine produces:
      - info['state']: 2D grid obs[x][y] = list of object dicts
      - info['events_triggered']: list of (effect_name, (name1, id1), (name2, id2))

    This adapter converts zstate objects/effects to the same format.
    """

    def __init__(self, block_size: int):
        """Initialize adapter.

        Args:
            block_size: Pixel size of one grid cell (vgfmri3=20, vgfmri4=35).
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._block_size = block_size
        # Persistent UUID -> synthetic obj_id mapping
        self._uuid_to_objid: dict[str, str] = {}
        # Per-sprite-type counters for generating IDs like 'goal.1'
        self._type_counters: dict[str, int] = {}
        # Reverse mapping: color -> list of (obj_id, sprite_type) for event resolution
        # Built fresh each frame by adapt_observation()
        self._color_to_objects: dict[str, list[tuple[str, str, tuple[int, int]]]] = {}

    def reset(self) -> None:
        """Reset UUID registry for new level/play."""
        self._uuid_to_objid.clear()
        self._type_counters.clear()
        self._color_to_objects.clear()

    @staticmethod
    def extract_color_mapping(zstates: list[dict]) -> dict[str, str]:
        """Extract sprite_type -> colorName mapping from zstates.

        Tomov et al. 2023 randomized color assignments per subject+game.
        Scans multiple zstates to find all sprite types across levels
        (not all types appear in every level).

        Args:
            zstates: List of zstate dicts from decompressed behavioral data.

        Returns:
            Dict mapping sprite type names to their colorName strings.
            Example: {"goal": "BROWN", "key": "RED", "wall": "ORANGE"}
        """
        color_map = {}
        for zstate in zstates:
            objects = zstate.get("objects", {})
            for sprite_type, positions in objects.items():
                if sprite_type in color_map:
                    continue  # already have this type
                for _pos_key, obj_data in positions.items():
                    color_name = obj_data.get("colorName")
                    if color_name is not None:
                        color_map[sprite_type] = color_name
                    break  # one sample per type is enough
        return color_map

    def detect_block_size(self, zstate: dict) -> int:
        """Detect block_size from a zstate's rect.size field.

        Args:
            zstate: A zstate dict with objects.

        Returns:
            Block size in pixels.
        """
        objects = zstate.get("objects", {})
        for sprite_type, positions in objects.items():
            for pos_key, obj_data in positions.items():
                rect = obj_data.get("rect", {})
                size = rect.get("size")
                if size and len(size) >= 1:
                    return size[0]
        raise ValueError("Could not determine block_size from zstate objects")

    # ------------------------------------------------------------------
    # adapt_observation
    # ------------------------------------------------------------------

    def adapt_observation(self, zstate: dict) -> list[list[list[dict]]]:
        """Convert zstate to 2D grid matching info['state'] format.

        The returned grid is indexed as obs[x][y] where each cell is a list
        of object dicts with keys: name, color, obj_id, pos, resources, resources_max.

        Args:
            zstate: A single state dict from decompressed zstates.

        Returns:
            2D grid matching the engine's info['state'] format.
        """
        objects = zstate.get("objects", {})

        # 1. Determine grid dimensions from max pixel coordinates
        width, height = self._compute_grid_dims(objects)

        # 2. Create empty grid
        grid: list[list[list[dict]]] = [
            [[] for _ in range(height)] for _ in range(width)
        ]

        # 3. Reset per-frame color->objects mapping
        self._color_to_objects.clear()

        # 4. Place each sprite into the grid
        for sprite_type, positions in objects.items():
            for pos_key, obj_data in positions.items():
                gx = obj_data["x"] // self._block_size
                gy = obj_data["y"] // self._block_size

                # Objects can be off-screen in VGDL (pushed/spawned outside
                # the visible grid).  They are not observable by the player,
                # so exclude them from the observation.
                if gx < 0 or gx >= width or gy < 0 or gy >= height:
                    continue

                # Get or create persistent obj_id for this UUID
                obj_id = self._get_or_create_objid(obj_data, sprite_type)

                # Build object dict matching engine format
                color_name = obj_data.get("colorName", "UNKNOWN")
                resources = obj_data.get("resources", {})

                obj_dict = {
                    "name": sprite_type,
                    "color": color_name,
                    "obj_id": obj_id,
                    "pos": (gx, gy),
                    "resources": dict(resources) if resources else {},
                    "resources_max": {},
                }

                grid[gx][gy].append(obj_dict)

                # Track color -> objects for event resolution
                if sprite_type not in ("wall", "floor", "background"):
                    if color_name not in self._color_to_objects:
                        self._color_to_objects[color_name] = []
                    self._color_to_objects[color_name].append(
                        (obj_id, sprite_type, (gx, gy))
                    )

        return grid

    def _compute_grid_dims(self, objects: dict) -> tuple[int, int]:
        """Compute grid dimensions from max pixel coordinates of all objects.

        Returns:
            (width, height) in grid cells.
        """
        max_gx = 0
        max_gy = 0
        has_objects = False

        for sprite_type, positions in objects.items():
            for pos_key, obj_data in positions.items():
                has_objects = True
                gx = obj_data["x"] // self._block_size
                gy = obj_data["y"] // self._block_size
                max_gx = max(max_gx, gx)
                max_gy = max(max_gy, gy)

        if not has_objects:
            raise ValueError("No objects in zstate -- cannot determine grid dimensions")

        return (max_gx + 1, max_gy + 1)

    def _get_uuid_hex(self, obj_data: dict) -> str:
        """Extract hex string from UUID bytes in object data."""
        obj_id_bytes = obj_data.get("ID")
        if obj_id_bytes is None:
            raise ValueError(f"Object missing 'ID' field: {list(obj_data.keys())}")
        return bytes(obj_id_bytes).hex()

    def _get_or_create_objid(self, obj_data: dict, sprite_type: str) -> str:
        """Get or create a persistent synthetic obj_id for an object UUID.

        Produces IDs in the same format as the engine: 'avatar.0', 'goal.1', etc.
        Walls always get type-based IDs without persistent tracking.

        Args:
            obj_data: Object data dict from zstate.
            sprite_type: VGDL sprite type name.

        Returns:
            Synthetic obj_id string like 'goal.1'.
        """
        uuid_hex = self._get_uuid_hex(obj_data)

        # Check if already registered
        if uuid_hex in self._uuid_to_objid:
            return self._uuid_to_objid[uuid_hex]

        # Create new ID
        counter = self._type_counters.get(sprite_type, 0)
        obj_id = f"{sprite_type}.{counter}"
        self._type_counters[sprite_type] = counter + 1

        # Persist mapping (walls too -- needed for event resolution)
        self._uuid_to_objid[uuid_hex] = obj_id
        return obj_id

    # ------------------------------------------------------------------
    # register_objects
    # ------------------------------------------------------------------

    def register_objects(self, zstate: dict) -> None:
        """Pre-register all objects from a zstate for consistent ID ordering.

        Should be called on the first frame of a play to ensure deterministic
        obj_id assignment (sorted by grid position).

        Args:
            zstate: First zstate dict of a play.
        """
        objects = zstate.get("objects", {})
        all_objs: list[tuple[str, str, int, int]] = []

        for sprite_type, positions in objects.items():
            for pos_key, obj_data in positions.items():
                uuid_hex = self._get_uuid_hex(obj_data)
                gx = obj_data["x"] // self._block_size
                gy = obj_data["y"] // self._block_size
                all_objs.append((sprite_type, uuid_hex, gx, gy))

        # Sort by position then type for deterministic ordering
        all_objs.sort(key=lambda o: (o[2], o[3], o[0]))

        for sprite_type, uuid_hex, _, _ in all_objs:
            if uuid_hex not in self._uuid_to_objid:
                counter = self._type_counters.get(sprite_type, 0)
                self._uuid_to_objid[uuid_hex] = f"{sprite_type}.{counter}"
                self._type_counters[sprite_type] = counter + 1

    # ------------------------------------------------------------------
    # adapt_events
    # ------------------------------------------------------------------

    def adapt_events(
        self,
        effect_list: list,
        prev_obs: list[list[list[dict]]],
        curr_obs: list[list[list[dict]]],
    ) -> list[tuple]:
        """Convert effectListByColor to events_triggered tuples.

        The engine produces events as:
            (effect_name, (name1, obj_id1), (name2, obj_id2))

        effectListByColor from zstates is:
            [effect_name, color1, color2, ...] or [[e1], [e2]]

        Args:
            effect_list: effectListByColor from zstate.
            prev_obs: Adapted 2D grid from previous frame.
            curr_obs: Adapted 2D grid from current frame.

        Returns:
            List of event tuples in engine format.
        """
        if not effect_list:
            return []

        # Flatten nested format: [[e1], [e2]] -> [e1, e2]
        flat_effects = self._flatten_effects(effect_list)

        # Build lookup tables for both observations
        prev_by_color = self._objects_by_color(prev_obs)
        curr_by_color = self._objects_by_color(curr_obs)

        events = []
        for effect in flat_effects:
            if not effect or len(effect) < 3:
                continue

            effect_name = effect[0]
            color1 = effect[1]
            color2 = effect[2]

            # Resolve which specific objects are involved
            # sprite1 = the affected object (color1), sprite2 = the cause (color2)
            sprite1 = self._resolve_sprite(color1, prev_by_color, curr_by_color)
            sprite2 = self._resolve_sprite(color2, prev_by_color, curr_by_color)

            if sprite1 is not None and sprite2 is not None:
                events.append((effect_name, sprite1, sprite2))

        return events

    def _flatten_effects(self, effect_list: list) -> list[list]:
        """Flatten effectListByColor which can be nested or flat.

        Formats seen in data:
        - Single effect: ['stepBack', 'YELLOW', 'DARKGRAY']
        - Multiple effects: [['stepBack', 'YELLOW', 'DARKGRAY'], ['killSprite', ...]]
        - Already flat list of effects
        """
        if not effect_list:
            return []

        # Check if it's a single flat effect (first element is a string = effect name)
        if isinstance(effect_list[0], str):
            return [effect_list]

        # It's a list of effects (could be nested one more level)
        result = []
        for item in effect_list:
            if isinstance(item, list):
                if item and isinstance(item[0], str):
                    # It's a single effect: ['stepBack', 'YELLOW', 'DARKGRAY']
                    result.append(item)
                elif item and isinstance(item[0], list):
                    # Nested one more level: [['stepBack', ...], ...]
                    result.extend(item)
                # Empty list -- skip
            # Non-list items are unexpected -- skip
        return result

    def _objects_by_color(
        self, obs: list[list[list[dict]]]
    ) -> dict[str, list[tuple[str, str]]]:
        """Build color -> [(name, obj_id)] mapping from adapted observation grid."""
        result: dict[str, list[tuple[str, str]]] = {}
        if not obs:
            return result

        for x, column in enumerate(obs):
            for y, cell in enumerate(column):
                for obj in cell:
                    color = obj.get("color", "")
                    name = obj["name"]
                    obj_id = obj["obj_id"]
                    if color not in result:
                        result[color] = []
                    result[color].append((name, obj_id))

        return result

    def _resolve_sprite(
        self,
        color: str,
        prev_by_color: dict[str, list[tuple[str, str]]],
        curr_by_color: dict[str, list[tuple[str, str]]],
    ) -> tuple[str, str] | None:
        """Resolve a color to a specific (name, obj_id) tuple.

        Checks prev_obs first (the object may have been destroyed), falls back
        to curr_obs.

        Returns:
            (name, obj_id) or None if no object found with that color.
        """
        # Try prev state first (destroyed objects only exist there)
        objs = prev_by_color.get(color, [])
        if objs:
            return objs[0]

        # Fall back to curr state
        objs = curr_by_color.get(color, [])
        if objs:
            return objs[0]

        return None

    # ------------------------------------------------------------------
    # adapt_action
    # ------------------------------------------------------------------

    def adapt_action(self, zstate: dict) -> int:
        """Derive action index from a zstate.

        Checks keystate array first (reliable), falls back to keyPressType.

        Args:
            zstate: Full zstate dict with keystate and/or keyPressType fields.

        Returns:
            Action index 0-5 matching the game engine convention.
        """
        keystate = zstate.get("keystate", [])
        if keystate:
            action = action_from_keystate(keystate)
            if action != ACTION_NOOP:
                return action
        key_press = zstate.get("keyPressType")
        if key_press is not None:
            return _KEY_TO_ACTION.get(key_press, ACTION_NOOP)
        return ACTION_NOOP

    # ------------------------------------------------------------------
    # adapt_frame (convenience)
    # ------------------------------------------------------------------

    def adapt_frame(
        self,
        prev_zstate: dict | None,
        curr_zstate: dict,
    ) -> dict:
        """Adapt a single frame, returning all engine-equivalent fields.

        Convenience method that calls adapt_observation, adapt_events, adapt_action
        and computes reward/won/lose.

        Expects zstates that have already been passed through
        ``realign_zstate_positions()`` so that sprite positions, effects,
        and scores are all consistently post-action within each frame.

        Args:
            prev_zstate: Previous zstate dict (None for first frame).
            curr_zstate: Current zstate dict.

        Returns:
            Dict with keys: observation, prev_observation, events,
            action_idx, action_name, reward, won, lose, score, prev_score.
        """
        curr_obs = self.adapt_observation(curr_zstate)

        # Events and prev_obs
        if prev_zstate is not None:
            prev_obs = self.adapt_observation(prev_zstate)
            effect_list = curr_zstate.get("effectListByColor", [])
            events = self.adapt_events(effect_list, prev_obs, curr_obs)
        else:
            prev_obs = curr_obs
            events = []

        # Action
        action_idx = self.adapt_action(curr_zstate)

        # Score / reward
        curr_score = curr_zstate.get("score", 0)
        prev_score = prev_zstate.get("score", 0) if prev_zstate else 0
        # Score can be float in some states, normalize to int
        curr_score = int(curr_score) if curr_score is not None else 0
        prev_score = int(prev_score) if prev_score is not None else 0
        reward = curr_score - prev_score

        # Terminal outcome (mutually exclusive, all False if ended=False).
        # Per VGFMRI_DB_README.md:174-176, plays fall into three categories:
        #   - WIN: win=True / win=1         (goal achieved in-game)
        #   - LOSE: win=False / win=-1      (avatar died or explicit loss)
        #   - TIMEOUT: win=None             (level's `duration` budget ran
        #                                    out with no win or loss)
        # Collapsing TIMEOUT into LOSE (which the old code did via the
        # `ended and not won -> lose` fallback) misrepresents the data --
        # ~42% of plays in the dataset are timeouts, not losses.
        win_value = curr_zstate.get("win")
        ended = curr_zstate.get("ended", False)
        won = ended and (win_value is True or win_value == 1)
        lose = ended and (win_value is False or win_value == -1)
        timeout = ended and win_value is None

        return {
            "observation": curr_obs,
            "prev_observation": prev_obs,
            "events": events,
            "action_idx": action_idx,
            "action_name": ACTION_NAMES[action_idx],
            "reward": reward,
            "won": won,
            "lose": lose,
            "timeout": timeout,
            "score": curr_score,
            "prev_score": prev_score,
        }

    # ------------------------------------------------------------------
    # get_action_name (static helper)
    # ------------------------------------------------------------------

    @staticmethod
    def get_action_name(zstate: dict) -> str:
        """Derive action name from a zstate.

        Args:
            zstate: Full zstate dict.

        Returns:
            Action name: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', or 'ACTION'.
        """
        keystate = zstate.get("keystate", [])
        if keystate:
            action_idx = action_from_keystate(keystate)
            if action_idx != ACTION_NOOP:
                return ACTION_NAMES[action_idx]
        key_press = zstate.get("keyPressType")
        if key_press is not None:
            name_map = {
                "up": "UP",
                "down": "DOWN",
                "left": "LEFT",
                "right": "RIGHT",
                "spacebar": "ACTION",
            }
            return name_map.get(key_press, "WAIT")
        return "WAIT"


def realign_zstate_positions(zstates: list[dict], play_win) -> list[dict]:
    """Fix the Tomov VGDL engine's split-brain timing in zstate recordings.

    Two separate recording artifacts are corrected here.

    1. **Position lag across consecutive frames.**  The Tomov engine
       snapshots each frame with keystate, effects, and score reflecting
       the current tick's action, but sprite positions still at their
       *previous*-tick locations.  Each frame's ``objects`` are patched
       from the next frame so positions describe the same tick as the
       other fields.  The last frame has no successor and keeps its
       original positions.

       ``ended`` / ``win`` similarly lag by one frame (they flip to True
       on the frame AFTER the terminal action).  They are left raw:
       per-frame action logs surface the terminal marker on the frame
       where the engine actually set it.  Consumers that want the
       terminal outcome attributed to the action that caused it do the
       one-frame lookahead themselves.

    2. **Play-level outcome mismatch at the terminal frame.**  The raw
       terminal zstate's ``win`` field is the engine's internal signal
       ("avatar did / did not achieve the goal" -> ``True`` or ``-1``),
       but the authoritative categorization of the play is
       ``play_doc["win"]`` per VGFMRI_DB_README.md:174-176, which
       distinguishes WIN (``True``) / LOSS (``False``/``-1``) /
       TIMEOUT (``None``).  ~42% of plays end in timeouts where the
       engine still wrote ``win=-1`` on the terminal zstate, which
       would cascade downstream as a spurious ``[LOSE]`` marker.  When
       the caller supplies ``play_win`` we overwrite the last frame's
       ``win`` so the adapter classifies the outcome correctly.

    Call this once on the raw zstate list before passing frames to
    ``ZstateAdapter.adapt_frame`` or ``convert_zstate_to_viewer``.

    Args:
        zstates: Raw decompressed zstate list from a single play.
        play_win: Play-level ``win`` value from the BSON play document
            (``True`` / ``False`` / ``-1`` / ``None``).  Overrides the
            raw terminal-frame ``win``.  Required -- callers without a
            play document can pass ``zstates[-1].get("win")`` to opt
            out of the override.

    Returns:
        New list of zstate dicts with patched sprite positions and
        terminal outcome matching ``play_win``.
    """
    if len(zstates) <= 1:
        return list(zstates)

    result: list[dict] = []
    for i in range(len(zstates)):
        corrected = dict(zstates[i])
        if i + 1 < len(zstates):
            nxt = zstates[i + 1]
            corrected["objects"] = _patch_sprite_positions(
                zstates[i]["objects"], nxt["objects"]
            )
        result.append(corrected)

    # Artifact (2): override the terminal frame's `win` with the
    # play-level outcome so TIMEOUT plays are not mis-classified as
    # LOSE (the engine writes `win=-1` on the last zstate for both).
    result[-1] = {**result[-1], "win": play_win}

    return result


def _sprite_uuid(obj_data: dict) -> str:
    """Extract a comparable UUID string from a sprite's ID field."""
    raw = obj_data.get("ID", b"")
    if isinstance(raw, bytes):
        return raw.hex()
    return str(raw)


def _patch_sprite_positions(curr_objects: dict, next_objects: dict) -> dict:
    """Replace sprite positions in *curr_objects* with those from *next_objects*.

    Sprites are matched by UUID.  A sprite present in curr but absent in
    next (killed during this tick) keeps its current position.
    """
    # Build UUID -> obj_data lookup for the next frame
    next_by_uuid: dict[str, dict[str, dict]] = {}
    for stype, positions in next_objects.items():
        by_uuid: dict[str, dict] = {}
        for _pos_key, obj_data in positions.items():
            by_uuid[_sprite_uuid(obj_data)] = obj_data
        next_by_uuid[stype] = by_uuid

    patched: dict[str, dict] = {}
    for stype, positions in curr_objects.items():
        new_positions: dict[str, dict] = {}
        next_lookup = next_by_uuid.get(stype, {})

        for pos_key, obj_data in positions.items():
            uuid = _sprite_uuid(obj_data)
            next_obj = next_lookup.get(uuid)

            if next_obj is not None:
                # Sprite survived -- adopt position from next frame
                updated = dict(obj_data)
                updated["x"] = next_obj["x"]
                updated["y"] = next_obj["y"]
                if "rect" in next_obj and "rect" in updated:
                    updated["rect"] = dict(updated["rect"])
                    updated["rect"]["pos"] = list(next_obj["rect"]["pos"])
                new_key = f"({next_obj['x']}, {next_obj['y']})"
                new_positions[new_key] = updated
            else:
                # Sprite killed this tick -- keep current position
                new_positions[pos_key] = obj_data

        patched[stype] = new_positions
    return patched


def convert_zstate_to_viewer(state: dict, block_size: int) -> dict:
    """Convert a BSON zstate to vgdl-js replay viewer format.

    Transforms sprite objects from pixel coordinates to grid coordinates
    and normalizes the state structure for the vgdl-js replay viewer.

    Args:
        state: Raw zstate dict from behavioral data (Tomov 2023).
        block_size: Pixel size of one grid cell.

    Returns:
        Dict with keys: score, time, ended, won, sprites.
        Sprites are keyed by type, each a list of dicts with
        id, key, col, row, alive, resources, _uuid.
    """
    sprites = {}
    sprite_counter = {}

    for sprite_type, positions in state["objects"].items():
        sprite_list = []
        for _pos_key, obj_data in positions.items():
            obj_id_bytes = obj_data.get("ID", b"")
            if isinstance(obj_id_bytes, bytes):
                obj_id_hex = obj_id_bytes.hex()
            else:
                obj_id_hex = str(obj_id_bytes)

            if sprite_type not in sprite_counter:
                sprite_counter[sprite_type] = 0
            numeric_id = sprite_counter[sprite_type]
            sprite_counter[sprite_type] += 1

            col = obj_data["x"] // block_size
            row = obj_data["y"] // block_size

            resources = obj_data.get("resources", {})
            if resources:
                resources = {k: v for k, v in resources.items() if v}

            sprite_data = {
                "id": numeric_id,
                "key": sprite_type,
                "col": col,
                "row": row,
                "alive": True,
                "resources": resources,
                "_uuid": obj_id_hex,
            }
            sprite_list.append(sprite_data)

        if sprite_list:
            sprites[sprite_type] = sprite_list

    # Decode the three mutually exclusive terminal outcomes so the viewer
    # does not have to re-infer them from ended/won (the old `lose =
    # ended && !won` convention collapsed timeouts into losses).
    ended = bool(state.get("ended", False))
    win_value = state.get("win")
    won = ended and (win_value is True or win_value == 1)
    lose = ended and (win_value is False or win_value == -1)
    timeout = ended and win_value is None
    return {
        "score": state.get("score", 0),
        "time": state.get("time", 0),
        "ended": ended,
        "won": won,
        "lose": lose,
        "timeout": timeout,
        "sprites": sprites,
    }
