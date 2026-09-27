"""
Tests for ZstateAdapter: converting zstate dicts to engine format.

Phase 1: State conversion tests
Phase 2: Event conversion tests
"""

import os
import sys

import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_eval.human_replay.zstate_adapter import (
    ZstateAdapter,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_NOOP,
    ACTION_SPACE,
    realign_zstate_positions,
)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic zstate data
# ---------------------------------------------------------------------------


def _make_uuid(n: int) -> bytes:
    """Create a 16-byte UUID-like value from an integer."""
    return n.to_bytes(16, "big")


def _make_obj(uuid_n: int, color: str, px: int, py: int, **extra) -> dict:
    """Create a minimal zstate object dict."""
    obj = {
        "ID": _make_uuid(uuid_n),
        "colorName": color,
        "x": px,
        "y": py,
        "rect": {"pos": [px, py], "size": [35, 35]},
        "resources": {},
    }
    obj.update(extra)
    return obj


def _make_zstate(
    objects: dict,
    score: int = 0,
    win=None,
    ended: bool = False,
    key_press: str | None = None,
    effects: list | None = None,
) -> dict:
    """Create a minimal zstate dict."""
    return {
        "objects": objects,
        "score": score,
        "win": win,
        "ended": ended,
        "keyPressType": key_press,
        # Explicit None-check so an empty list stays an empty list rather
        # than collapsing via `effects or []` (which treats [] and None
        # identically, masking caller intent).
        "effectListByColor": [] if effects is None else effects,
    }


# A simple 7x5 grid (block_size=35, pixels 0-210 x 0-140)
# Avatar at (4,2) = pixel (140, 70)
# Wall at (0,0) = pixel (0, 0)
# Goal (PURPLE) at (5,3) = pixel (175, 105)
# Key (ORANGE) at (2,1) = pixel (70, 35)
SIMPLE_OBJECTS = {
    "avatar": {
        "(140, 70)": _make_obj(1, "YELLOW", 140, 70),
    },
    "wall": {
        "(0, 0)": _make_obj(100, "DARKGRAY", 0, 0),
        "(210, 0)": _make_obj(101, "DARKGRAY", 210, 0),
        "(0, 140)": _make_obj(102, "DARKGRAY", 0, 140),
        "(210, 140)": _make_obj(103, "DARKGRAY", 210, 140),
    },
    "goal": {
        "(175, 105)": _make_obj(10, "PURPLE", 175, 105),
    },
    "key": {
        "(70, 35)": _make_obj(20, "ORANGE", 70, 35),
    },
}


# ===================================================================
# Phase 1: State conversion tests
# ===================================================================


class TestAdaptObservation:
    """Tests for adapt_observation(): pixel-to-grid conversion, ID persistence, etc."""

    def test_pixel_to_grid_conversion(self):
        """Avatar at pixel (140, 70) with block_size=35 -> grid (4, 2)."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        # Find avatar in grid
        avatar_found = False
        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    if obj["name"] == "avatar":
                        assert (x, y) == (4, 2), f"Avatar at ({x},{y}), expected (4,2)"
                        avatar_found = True
        assert avatar_found, "Avatar not found in grid"

    def test_grid_dimensions(self):
        """Grid dimensions should be derived from max object positions."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        # Walls at (0,0) and (210,140) -> grid (0,0) and (6,4) -> dims 7x5
        width = len(obs)
        height = len(obs[0])
        assert width == 7, f"Width {width}, expected 7"
        assert height == 5, f"Height {height}, expected 5"

    def test_object_dict_format(self):
        """Each object dict should have all required keys matching engine format."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        required_keys = {"name", "color", "obj_id", "pos", "resources", "resources_max"}

        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    missing = required_keys - set(obj.keys())
                    assert not missing, f"Object {obj} missing keys: {missing}"
                    assert obj["pos"] == (x, y), (
                        f"Object pos {obj['pos']} != grid ({x},{y})"
                    )

    def test_object_id_persistence_across_frames(self):
        """Same UUID should get the same obj_id across consecutive frames."""
        adapter = ZstateAdapter(block_size=35)

        # Frame 1: avatar at (4,2)
        frame1_objs = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "npc": {"(105, 70)": _make_obj(50, "GREEN", 105, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        obs1 = adapter.adapt_observation(_make_zstate(frame1_objs))

        # Frame 2: NPC moved from (3,2) to (3,3) -- same UUID 50
        frame2_objs = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "npc": {"(105, 105)": _make_obj(50, "GREEN", 105, 105)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        obs2 = adapter.adapt_observation(_make_zstate(frame2_objs))

        # Find NPC obj_id in both frames
        def find_npc(grid):
            for x, col in enumerate(grid):
                for y, cell in enumerate(col):
                    for obj in cell:
                        if obj["name"] == "npc":
                            return obj["obj_id"]
            return None

        npc_id_1 = find_npc(obs1)
        npc_id_2 = find_npc(obs2)

        assert npc_id_1 is not None, "NPC not found in frame 1"
        assert npc_id_2 is not None, "NPC not found in frame 2"
        assert npc_id_1 == npc_id_2, f"NPC ID changed: {npc_id_1} -> {npc_id_2}"

    def test_object_id_format(self):
        """Synthetic obj_ids should match engine format: 'sprite_type.N'."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    obj_id = obj["obj_id"]
                    parts = obj_id.rsplit(".", 1)
                    assert len(parts) == 2, f"obj_id {obj_id!r} not in 'type.N' format"
                    assert parts[0] == obj["name"], (
                        f"obj_id prefix {parts[0]!r} != name {obj['name']!r}"
                    )
                    assert parts[1].isdigit(), (
                        f"obj_id suffix {parts[1]!r} not a number"
                    )

    def test_wall_detection(self):
        """Walls should appear in the grid with name='wall'."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        wall_positions = set()
        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    if obj["name"] == "wall":
                        wall_positions.add((x, y))

        # We placed walls at pixels (0,0), (210,0), (0,140), (210,140)
        # -> grid (0,0), (6,0), (0,4), (6,4)
        expected = {(0, 0), (6, 0), (0, 4), (6, 4)}
        assert wall_positions == expected, (
            f"Walls at {wall_positions}, expected {expected}"
        )

    def test_resource_extraction(self):
        """Avatar resources should be preserved in adapted observation."""
        adapter = ZstateAdapter(block_size=35)

        objects = {
            "avatar": {
                "(140, 70)": _make_obj(1, "YELLOW", 140, 70, resources={"key": 1}),
            },
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        obs = adapter.adapt_observation(_make_zstate(objects))

        # Find avatar
        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    if obj["name"] == "avatar":
                        assert obj["resources"] == {"key": 1}, (
                            f"Avatar resources: {obj['resources']}"
                        )
                        return

        raise AssertionError("Avatar not found")

    def test_color_preserved(self):
        """Object colors from zstate should be preserved verbatim."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS)
        obs = adapter.adapt_observation(zstate)

        colors_found = {}
        for x, col in enumerate(obs):
            for y, cell in enumerate(col):
                for obj in cell:
                    colors_found[obj["name"]] = obj["color"]

        assert colors_found["avatar"] == "YELLOW"
        assert colors_found["wall"] == "DARKGRAY"
        assert colors_found["goal"] == "PURPLE"
        assert colors_found["key"] == "ORANGE"

    def test_register_objects_deterministic_ordering(self):
        """register_objects should assign IDs in sorted position order."""
        adapter = ZstateAdapter(block_size=35)

        objects = {
            "goal": {
                "(175, 105)": _make_obj(10, "PURPLE", 175, 105),
                "(35, 35)": _make_obj(11, "PURPLE", 35, 35),
            },
            "key": {
                "(70, 35)": _make_obj(20, "ORANGE", 70, 35),
            },
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
        }
        zstate = _make_zstate(objects)
        adapter.register_objects(zstate)

        obs = adapter.adapt_observation(zstate)

        # Sorted by position (gx, gy, type):
        # (0,0) wall.0
        # (1,1) goal.0  (pixel 35,35 -> grid 1,1)
        # (2,1) key.0   (pixel 70,35 -> grid 2,1)
        # (4,2) avatar.0
        # (5,3) goal.1  (pixel 175,105 -> grid 5,3)

        def find_obj(grid, name, color, pos):
            for obj in grid[pos[0]][pos[1]]:
                if obj["name"] == name and obj["color"] == color:
                    return obj["obj_id"]
            return None

        goal_at_1_1 = find_obj(obs, "goal", "PURPLE", (1, 1))
        key_at_2_1 = find_obj(obs, "key", "ORANGE", (2, 1))
        goal_at_5_3 = find_obj(obs, "goal", "PURPLE", (5, 3))

        assert goal_at_1_1 == "goal.0", f"First goal ID: {goal_at_1_1}"
        assert key_at_2_1 == "key.0", f"Key ID: {key_at_2_1}"
        assert goal_at_5_3 == "goal.1", f"Second goal ID: {goal_at_5_3}"

    def test_empty_objects_raises(self):
        """adapt_observation should raise on empty objects dict."""
        adapter = ZstateAdapter(block_size=35)
        with pytest.raises(ValueError, match="No objects"):
            adapter.adapt_observation(_make_zstate({}))

    def test_invalid_block_size_raises(self):
        """block_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            ZstateAdapter(block_size=0)
        with pytest.raises(ValueError, match="block_size must be positive"):
            ZstateAdapter(block_size=-1)

    def test_detect_block_size(self):
        """detect_block_size should read from rect.size."""
        adapter = ZstateAdapter(block_size=1)  # placeholder
        zstate = _make_zstate(SIMPLE_OBJECTS)
        assert adapter.detect_block_size(zstate) == 35


# ===================================================================
# Phase 2: Event conversion tests
# ===================================================================


class TestAdaptEvents:
    """Tests for adapt_events(): effectListByColor -> engine event tuples."""

    def _make_two_frame_scenario(self, prev_objects, curr_objects, effects):
        """Helper: create adapted prev/curr observations and return events."""
        adapter = ZstateAdapter(block_size=35)
        # Register objects from first frame for consistent IDs
        adapter.register_objects(_make_zstate(prev_objects))
        prev_obs = adapter.adapt_observation(_make_zstate(prev_objects))
        curr_obs = adapter.adapt_observation(_make_zstate(curr_objects))
        events = adapter.adapt_events(effects, prev_obs, curr_obs)
        return events, adapter

    def test_blocking_event(self):
        """stepBack effect should produce blocking event tuple."""
        prev_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "wall": {"(175, 70)": _make_obj(100, "DARKGRAY", 175, 70)},
        }
        # Avatar didn't move (blocked)
        curr_objects = prev_objects

        effects = ["stepBack", "YELLOW", "DARKGRAY"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        effect_name, sprite1, sprite2 = events[0]
        assert effect_name == "stepBack"
        assert sprite1[0] == "avatar"  # name
        assert sprite2[0] == "wall"  # name

    def test_collection_event(self):
        """killSprite on non-avatar should produce collection event."""
        prev_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "key": {"(175, 70)": _make_obj(20, "ORANGE", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        # Avatar moved to key position, key disappeared
        curr_objects = {
            "avatar": {"(175, 70)": _make_obj(1, "YELLOW", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        effects = ["killSprite", "ORANGE", "YELLOW"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        effect_name, sprite1, sprite2 = events[0]
        assert effect_name == "killSprite"
        assert sprite1[0] == "key"  # ORANGE = key
        assert sprite2[0] == "avatar"  # YELLOW = avatar

    def test_push_event(self):
        """bounceForward should produce push event."""
        prev_objects = {
            "avatar": {"(105, 70)": _make_obj(1, "YELLOW", 105, 70)},
            "box": {"(140, 70)": _make_obj(30, "BROWN", 140, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        # Avatar moved right, box pushed right
        curr_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "box": {"(175, 70)": _make_obj(30, "BROWN", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        effects = ["bounceForward", "BROWN", "YELLOW"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        effect_name, sprite1, sprite2 = events[0]
        assert effect_name == "bounceForward"
        assert sprite1[0] == "box"  # BROWN = box
        assert sprite2[0] == "avatar"  # YELLOW = avatar

    def test_transform_event(self):
        """transformTo should produce transform event."""
        prev_objects = {
            "avatar": {"(105, 70)": _make_obj(1, "YELLOW", 105, 70)},
            "portal": {"(140, 70)": _make_obj(40, "RED", 140, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        # Avatar moved to portal, portal transformed to BLACK
        curr_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "exit": {"(140, 70)": _make_obj(41, "BLACK", 140, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        effects = ["transformTo", "RED", "YELLOW"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        effect_name, sprite1, sprite2 = events[0]
        assert effect_name == "transformTo"
        assert sprite1[0] == "portal"  # RED = portal (from prev state)
        assert sprite2[0] == "avatar"  # YELLOW = avatar

    def test_nested_effects(self):
        """Nested effectListByColor [[e1], [e2]] should produce two events."""
        prev_objects = {
            "avatar": {"(105, 70)": _make_obj(1, "YELLOW", 105, 70)},
            "box": {"(140, 70)": _make_obj(30, "BROWN", 140, 70)},
            "hole": {"(175, 70)": _make_obj(40, "BLUE", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        curr_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            # Box and hole both destroyed
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        # Nested format: push + box killed by hole
        effects = [
            ["bounceForward", "BROWN", "YELLOW"],
            ["killSprite", "BROWN", "BLUE"],
        ]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 2
        assert events[0][0] == "bounceForward"
        assert events[1][0] == "killSprite"

    def test_idle_frame_no_events(self):
        """Frame with no keyPress and empty effects -> empty events list."""
        adapter = ZstateAdapter(block_size=35)
        objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        obs = adapter.adapt_observation(_make_zstate(objects))
        events = adapter.adapt_events([], obs, obs)
        assert events == []

    def test_empty_effect_list(self):
        """Empty effectListByColor should return empty events."""
        adapter = ZstateAdapter(block_size=35)
        objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        obs = adapter.adapt_observation(_make_zstate(objects))

        assert adapter.adapt_events([], obs, obs) == []
        assert adapter.adapt_events(None, obs, obs) == []

    def test_death_event(self):
        """killSprite on avatar should produce death event."""
        prev_objects = {
            "avatar": {"(140, 70)": _make_obj(1, "YELLOW", 140, 70)},
            "enemy": {"(175, 70)": _make_obj(50, "RED", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        # Avatar moved into enemy and died
        curr_objects = {
            "enemy": {"(175, 70)": _make_obj(50, "RED", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        effects = ["killSprite", "YELLOW", "RED"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        effect_name, sprite1, sprite2 = events[0]
        assert effect_name == "killSprite"
        assert sprite1[0] == "avatar"  # YELLOW = avatar died
        assert sprite2[0] == "enemy"  # RED = enemy killed it

    def test_conditional_remove_event(self):
        """killIfOtherHasMore should produce event tuple."""
        prev_objects = {
            "avatar": {
                "(140, 70)": _make_obj(1, "YELLOW", 140, 70, resources={"key": 1})
            },
            "goal": {"(175, 70)": _make_obj(10, "PURPLE", 175, 70)},
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }
        # Avatar reached goal with key, goal removed
        curr_objects = {
            "avatar": {
                "(175, 70)": _make_obj(1, "YELLOW", 175, 70, resources={"key": 1})
            },
            "wall": {"(0, 0)": _make_obj(100, "DARKGRAY", 0, 0)},
        }

        effects = ["killIfOtherHasMore", "PURPLE", "YELLOW"]
        events, adapter = self._make_two_frame_scenario(
            prev_objects, curr_objects, effects
        )

        assert len(events) == 1
        assert events[0][0] == "killIfOtherHasMore"
        assert events[0][1][0] == "goal"  # PURPLE = goal
        assert events[0][2][0] == "avatar"  # YELLOW = avatar


# ===================================================================
# Action conversion tests
# ===================================================================


def _zstate_with_key(key_name: str | None) -> dict:
    """Build a minimal zstate dict with the given key pressed in keystate."""
    keystate = [0] * 323
    pygame_codes = {"up": 273, "down": 274, "left": 276, "right": 275, "spacebar": 32}
    if key_name is not None and key_name in pygame_codes:
        keystate[pygame_codes[key_name]] = 1
    return {"keyPressType": key_name, "keystate": keystate}


class TestAdaptAction:
    """Tests for adapt_action(): zstate -> action index."""

    def test_directional_keys(self):
        adapter = ZstateAdapter(block_size=35)
        assert adapter.adapt_action(_zstate_with_key("up")) == ACTION_UP
        assert adapter.adapt_action(_zstate_with_key("down")) == ACTION_DOWN
        assert adapter.adapt_action(_zstate_with_key("left")) == ACTION_LEFT
        assert adapter.adapt_action(_zstate_with_key("right")) == ACTION_RIGHT

    def test_spacebar(self):
        adapter = ZstateAdapter(block_size=35)
        assert adapter.adapt_action(_zstate_with_key("spacebar")) == ACTION_SPACE

    def test_none_is_noop(self):
        adapter = ZstateAdapter(block_size=35)
        assert adapter.adapt_action(_zstate_with_key(None)) == ACTION_NOOP

    def test_unknown_key_is_noop(self):
        adapter = ZstateAdapter(block_size=35)
        assert (
            adapter.adapt_action({"keyPressType": "unknown", "keystate": [0] * 323})
            == ACTION_NOOP
        )


# ===================================================================
# adapt_frame convenience method tests
# ===================================================================


class TestAdaptFrame:
    """Tests for adapt_frame() convenience method."""

    def test_first_frame(self):
        """First frame (prev=None) should work without errors."""
        adapter = ZstateAdapter(block_size=35)
        zstate = _make_zstate(SIMPLE_OBJECTS, score=0, key_press=None)
        result = adapter.adapt_frame(None, zstate)

        assert result["action_idx"] == ACTION_NOOP
        assert result["action_name"] == "WAIT"
        assert result["reward"] == 0
        assert result["won"] is False
        assert result["lose"] is False
        assert result["score"] == 0

    def test_score_delta(self):
        """Reward should be curr_score - prev_score."""
        adapter = ZstateAdapter(block_size=35)

        prev = _make_zstate(SIMPLE_OBJECTS, score=5)
        curr = _make_zstate(SIMPLE_OBJECTS, score=15, key_press="right")
        result = adapter.adapt_frame(prev, curr)

        assert result["reward"] == 10
        assert result["score"] == 15
        assert result["prev_score"] == 5

    def test_win_detection(self):
        """win=True should set won=True."""
        adapter = ZstateAdapter(block_size=35)
        prev = _make_zstate(SIMPLE_OBJECTS, score=0)
        curr = _make_zstate(SIMPLE_OBJECTS, score=10, win=True, ended=True)
        result = adapter.adapt_frame(prev, curr)

        assert result["won"] is True
        assert result["lose"] is False

    def test_lose_detection(self):
        """win=-1 should set lose=True."""
        adapter = ZstateAdapter(block_size=35)
        prev = _make_zstate(SIMPLE_OBJECTS, score=0)
        curr = _make_zstate(SIMPLE_OBJECTS, score=0, win=-1, ended=True)
        result = adapter.adapt_frame(prev, curr)

        assert result["won"] is False
        assert result["lose"] is True

    def test_timeout_detection(self):
        """ended=True with win=None should be treated as timeout, not lose.

        Per VGFMRI_DB_README.md:174-176 the dataset has three outcomes:
        WIN, LOSS, and Incomplete/Timeout (~42% of plays).  Collapsing
        timeout into lose is misleading -- the level's time budget
        expired without the engine resolving the play.
        """
        adapter = ZstateAdapter(block_size=35)
        prev = _make_zstate(SIMPLE_OBJECTS, score=0)
        curr = _make_zstate(SIMPLE_OBJECTS, score=0, win=None, ended=True)
        result = adapter.adapt_frame(prev, curr)

        assert result["won"] is False
        assert result["lose"] is False
        assert result["timeout"] is True


# ===================================================================
# realign_zstate_positions
# ===================================================================


class TestRealignZstatePositions:
    """Terminal ended/win must NOT be backfilled onto the triggering frame.

    The engine writes ended=True / win=True on the frame AFTER the
    triggering action, and the per-frame action log surfaces the marker
    on that natural frame.  Positions ARE backfilled (separate artifact).
    """

    def test_ended_win_not_backfilled(self):
        #  F0: pre-terminal event (engine wrote ended=False, win=None)
        #  F1: engine flipped ended=True, win=True (terminal marker)
        f0 = _make_zstate(SIMPLE_OBJECTS, score=0, ended=False)
        f0["win"] = None
        f1 = _make_zstate(SIMPLE_OBJECTS, score=10, ended=True)
        f1["win"] = True

        patched = realign_zstate_positions([f0, f1], play_win=True)

        assert patched[0]["ended"] is False, (
            "terminal ended flag leaked backward onto the pre-terminal frame"
        )
        assert patched[0]["win"] is None, (
            "terminal win flag leaked backward onto the pre-terminal frame"
        )
        assert patched[0]["score"] == 0, (
            "terminal score leaked backward onto the pre-terminal frame"
        )
        assert patched[1]["ended"] is True
        assert patched[1]["win"] is True

    def test_play_win_overrides_only_last_frame(self):
        #  Timeout: engine wrote win=-1 on terminal frame, play_doc says None.
        f0 = _make_zstate(SIMPLE_OBJECTS, score=0, ended=False)
        f0["win"] = False
        f1 = _make_zstate(SIMPLE_OBJECTS, score=0, ended=True)
        f1["win"] = -1

        patched = realign_zstate_positions([f0, f1], play_win=None)

        assert patched[-1]["win"] is None, "last frame's win must match play_win"
        assert patched[0]["win"] is False, "pre-terminal frame's win was mangled"

    def test_positions_still_patched(self):
        #  Sanity check: position backfill (separate artifact) still works.
        f0_objects = {
            "avatar": {"(0, 0)": _make_obj(1, "YELLOW", 0, 0)},
        }
        f1_objects = {
            "avatar": {"(35, 0)": _make_obj(1, "YELLOW", 35, 0)},
        }
        f0 = _make_zstate(f0_objects, score=0)
        f1 = _make_zstate(f1_objects, score=0)

        patched = realign_zstate_positions([f0, f1], play_win=None)

        # f0's avatar position should now match f1's (35, 0)
        avatar_entries = list(patched[0]["objects"]["avatar"].items())
        assert len(avatar_entries) == 1
        _pos_key, obj_data = avatar_entries[0]
        assert (obj_data["x"], obj_data["y"]) == (35, 0)


# ===================================================================
# Integration test with real BSON data (conditional)
# ===================================================================

BSON_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "workdir",
    "prepare_behavioral_data",
)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(BSON_DATA_DIR, "plays")),
    reason="Behavioral data not available locally",
)
class TestWithRealData:
    """Integration tests using actual BSON behavioral data."""

    def _load_first_bait_play(self):
        """Load the first bait play from any subject (deterministic game)."""
        from src.llm_eval.human_replay.data_loader import HumanPlayLoader

        loader = HumanPlayLoader(BSON_DATA_DIR)
        # Search across subjects for any bait play
        for subject in loader.list_subjects():
            for run in loader.list_runs(subject):
                plays = loader.list_plays(subject, run)
                for play_info in plays:
                    if "bait" in play_info["game_name"]:
                        play_doc, states = loader.load_play(
                            subject, run, play_info["play_idx"]
                        )
                        return play_doc, states
        pytest.skip("No bait play found in behavioral data")

    def test_real_state_conversion(self):
        """Adapt a real zstate and verify grid structure."""
        play_doc, states = self._load_first_bait_play()
        first_state = states[0]

        # Detect block size
        adapter = ZstateAdapter(block_size=1)
        block_size = adapter.detect_block_size(first_state)
        assert block_size in (20, 35), f"Unexpected block_size: {block_size}"

        adapter = ZstateAdapter(block_size=block_size)
        adapter.register_objects(first_state)
        grid = adapter.adapt_observation(first_state)

        # Basic structural checks
        assert len(grid) > 0, "Grid width is 0"
        assert len(grid[0]) > 0, "Grid height is 0"

        # Should contain avatar (color is subject-specific per VGFMRI_DB_README.md)
        avatar_found = False
        for x, col in enumerate(grid):
            for y, cell in enumerate(col):
                for obj in cell:
                    if obj["name"] == "avatar":
                        avatar_found = True
                        # Color varies by subject (seed-based assignment)
                        assert isinstance(obj["color"], str) and obj["color"], (
                            f"Avatar color should be a non-empty string, got: {obj['color']!r}"
                        )

        assert avatar_found, "Avatar not found in adapted observation"

    def test_real_multi_frame_consistency(self):
        """Process multiple frames from a real play and verify no crashes."""
        play_doc, states = self._load_first_bait_play()
        first_state = states[0]

        block_size = ZstateAdapter(block_size=1).detect_block_size(first_state)
        adapter = ZstateAdapter(block_size=block_size)
        adapter.register_objects(first_state)

        prev_zstate = None
        for i, zstate in enumerate(states[:20]):
            result = adapter.adapt_frame(prev_zstate, zstate)
            assert "observation" in result
            assert "events" in result
            assert result["action_idx"] in range(6)
            prev_zstate = zstate

    def test_all_frames_processable(self):
        """Every frame in a real play should be processable without error."""
        play_doc, states = self._load_first_bait_play()

        block_size = ZstateAdapter(block_size=1).detect_block_size(states[0])
        adapter = ZstateAdapter(block_size=block_size)
        adapter.register_objects(states[0])

        prev_zstate = None
        for i, zstate in enumerate(states):
            adapter.adapt_frame(prev_zstate, zstate)
            prev_zstate = zstate

        # Should have processed all frames
        assert i == len(states) - 1
