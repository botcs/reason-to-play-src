"""Tests for replay_codec: delta encoding/expansion of .replay.json.gz states."""

import copy
import gzip
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.replay_codec import (
    delta_encode_states,
    expand_delta_states,
    load_replay,
    save_replay,
)


def test_delta_bytes_independent_of_python_hash_seed():
    """Release checksums must not depend on process hash randomization."""
    script = """
import json
from data.replay_codec import delta_encode_states
keys = ['wall', 'avatar', 'goal', 'projectile', 'floor', 'resource']
record = {'states': [
    {'sprites': {key: [{'col': 0}] for key in keys}},
    {'sprites': {key: [{'col': 1}] for key in keys[:-1]}},
]}
delta_encode_states(record)
print(json.dumps(record, separators=(',', ':')))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        for seed in (1, 23, 981)
    ]
    assert outputs[0] == outputs[1] == outputs[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WALL_A = {"id": 0, "key": "wall", "col": 0, "row": 0, "alive": True}
WALL_B = {"id": 1, "key": "wall", "col": 1, "row": 0, "alive": True}
WALL_C = {"id": 2, "key": "wall", "col": 2, "row": 0, "alive": True}

AVATAR_POS1 = {"id": 0, "key": "avatar", "col": 3, "row": 3, "alive": True}
AVATAR_POS2 = {"id": 0, "key": "avatar", "col": 4, "row": 3, "alive": True}
AVATAR_POS3 = {"id": 0, "key": "avatar", "col": 5, "row": 3, "alive": True}

KEY_SPRITE = {"id": 0, "key": "key", "col": 6, "row": 1, "alive": True}

GOAL_SPRITE = {"id": 0, "key": "goal", "col": 7, "row": 7, "alive": True}


def _make_state(sprites: dict, *, score=0, level=0, attempt=0) -> dict:
    return {
        "score": score,
        "time": 0,
        "ended": False,
        "won": False,
        "lose": False,
        "timeout": False,
        "level": level,
        "attempt": attempt,
        "action_log": "test",
        "sprites": sprites,
    }


def _make_replay(states: list[dict]) -> dict:
    return {
        "game": "test_game",
        "source": "human",
        "model": "test",
        "subject": "sub-01",
        "system_prompt": "test prompt",
        "prompt_name": "test",
        "suggestion_level": "minimal",
        "start_level": 0,
        "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:01:00",
        "outcome": "completed",
        "total_steps": len(states),
        "total_frames": len(states),
        "meta": {},
        "color_mapping": {},
        "game_description": "test",
        "steps": [],
        "states": states,
    }


# ---------------------------------------------------------------------------
# delta_encode_states / expand_delta_states round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Encoding then expanding must reproduce the input states exactly."""

    def test_static_game(self):
        """All frames identical (avatar doesn't move) -- sprites omitted after frame 0."""
        sprites = {"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(5)]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch after round-trip"

    def test_avatar_moves_each_frame(self):
        """Avatar changes every frame, walls static."""
        walls = [WALL_A, WALL_B]
        positions = [AVATAR_POS1, AVATAR_POS2, AVATAR_POS3, AVATAR_POS1, AVATAR_POS2]
        states = [_make_state({"wall": walls, "avatar": [p]}) for p in positions]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch"

    def test_sprite_collected(self):
        """Key sprite disappears mid-replay (collected)."""
        states = [
            _make_state(
                {"wall": [WALL_A], "avatar": [AVATAR_POS1], "key": [KEY_SPRITE]}
            ),
            _make_state(
                {"wall": [WALL_A], "avatar": [AVATAR_POS1], "key": [KEY_SPRITE]}
            ),
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS2], "key": []}),
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS3], "key": []}),
        ]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch"

    def test_new_sprite_type_appears(self):
        """A sprite type not present in frame 0 appears later (e.g. projectile spawned)."""
        projectile = {"id": 0, "key": "bullet", "col": 5, "row": 5, "alive": True}
        states = [
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS1]}),
            _make_state(
                {"wall": [WALL_A], "avatar": [AVATAR_POS1], "bullet": [projectile]}
            ),
            _make_state(
                {"wall": [WALL_A], "avatar": [AVATAR_POS1], "bullet": [projectile]}
            ),
        ]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch"

    def test_lemmings_walls_destroyed(self):
        """Walls shrink over time (lemmings-style destroyable walls)."""
        states = [
            _make_state({"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS2]}),
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS2]}),
            _make_state({"wall": [], "avatar": [AVATAR_POS3]}),
        ]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch"

    def test_level_transition_sprites_change_completely(self):
        """Level changes: entirely different sprite set."""
        level0_sprites = {
            "wall": [WALL_A],
            "avatar": [AVATAR_POS1],
            "key": [KEY_SPRITE],
        }
        level1_sprites = {
            "wall": [WALL_B, WALL_C],
            "avatar": [AVATAR_POS3],
            "goal": [GOAL_SPRITE],
        }
        states = [
            _make_state(level0_sprites, level=0),
            _make_state(level0_sprites, level=0),
            _make_state(level1_sprites, level=1),
            _make_state(level1_sprites, level=1),
        ]
        original = copy.deepcopy(states)

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        for i, (got, want) in enumerate(zip(data["states"], original)):
            assert got == want, f"state[{i}] mismatch"


# ---------------------------------------------------------------------------
# Encoding correctness: verify the delta structure itself
# ---------------------------------------------------------------------------


class TestEncodingShape:
    """Verify that encoding actually removes redundant data."""

    def test_identical_frames_have_no_sprites_key(self):
        """Consecutive identical frames should have sprites key removed."""
        sprites = {"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(4)]

        data = _make_replay(states)
        delta_encode_states(data)

        assert "sprites" in data["states"][0], "first frame must keep sprites"
        for i in range(1, 4):
            assert "sprites" not in data["states"][i], (
                f"state[{i}] should not have sprites (identical to previous)"
            )

    def test_only_changed_types_kept(self):
        """When only avatar moves, only avatar should be in the delta."""
        states = [
            _make_state({"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS2]}),
        ]

        data = _make_replay(states)
        delta_encode_states(data)

        delta_sprites = data["states"][1]["sprites"]
        assert "avatar" in delta_sprites, "changed type must be in delta"
        assert "wall" not in delta_sprites, "unchanged type must be omitted from delta"

    def test_delta_encoded_flag_set(self):
        states = [_make_state({"wall": [WALL_A]}) for _ in range(2)]
        data = _make_replay(states)
        delta_encode_states(data)
        assert data.get("delta_encoded") is True

    def test_delta_encoded_flag_cleared_after_expand(self):
        states = [_make_state({"wall": [WALL_A]}) for _ in range(2)]
        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)
        assert "delta_encoded" not in data

    def test_size_reduction(self):
        """Delta encoding should produce smaller JSON for redundant states."""
        sprites = {"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(100)]
        # Make avatar move on 10 frames
        for i in range(10, 100, 10):
            states[i] = _make_state(
                {"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS2]}
            )

        original_size = len(json.dumps(states))

        data = _make_replay(states)
        delta_encode_states(data)
        encoded_size = len(json.dumps(data["states"]))

        assert encoded_size < original_size * 0.5, (
            f"encoded ({encoded_size}) should be much smaller than original ({original_size})"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_states(self):
        data = _make_replay([])
        original = copy.deepcopy(data)
        delta_encode_states(data)
        expand_delta_states(data)
        assert data["states"] == original["states"]

    def test_single_state(self):
        data = _make_replay([_make_state({"wall": [WALL_A]})])
        original = copy.deepcopy(data)
        delta_encode_states(data)
        expand_delta_states(data)
        assert data["states"] == original["states"]

    def test_expand_noop_on_full_states(self):
        """expand_delta_states is a no-op if delta_encoded flag is absent."""
        states = [_make_state({"wall": [WALL_A]}) for _ in range(3)]
        data = _make_replay(states)
        original = copy.deepcopy(data)
        assert "delta_encoded" not in data
        expand_delta_states(data)
        assert data == original

    def test_double_encode_raises(self):
        """Encoding an already-encoded file should raise, not corrupt data."""
        states = [
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS1]}),
            _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS2]}),
        ]
        data = _make_replay(states)
        delta_encode_states(data)
        with pytest.raises(ValueError, match="already delta-encoded"):
            delta_encode_states(data)

    def test_expand_does_not_alias_sprites(self):
        """Expanded frames must not share the same sprites dict object,
        otherwise mutating one frame would corrupt another."""
        sprites = {"wall": [WALL_A], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(3)]

        data = _make_replay(states)
        delta_encode_states(data)
        expand_delta_states(data)

        data["states"][1]["sprites"]["wall"] = [WALL_B]
        assert data["states"][0]["sprites"]["wall"] == [WALL_A], (
            "mutating state[1] must not affect state[0]"
        )
        assert data["states"][2]["sprites"]["wall"] == [WALL_A], (
            "mutating state[1] must not affect state[2]"
        )


# ---------------------------------------------------------------------------
# File I/O: load_replay / save_replay
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_save_and_load_round_trip(self, tmp_path):
        """save_replay then load_replay returns the input data."""
        sprites = {"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS1]}
        states = [
            _make_state(sprites),
            _make_state(sprites),
            _make_state({"wall": [WALL_A, WALL_B], "avatar": [AVATAR_POS2]}),
        ]
        data = _make_replay(states)
        original = copy.deepcopy(data)

        path = tmp_path / "test.replay.json.gz"
        save_replay(data, path)
        loaded = load_replay(path)

        assert loaded["states"] == original["states"]
        assert loaded["game"] == original["game"]
        assert "delta_encoded" not in loaded

    def test_save_produces_valid_gzip(self, tmp_path):
        """Output file is valid gzip containing JSON with delta_encoded flag."""
        data = _make_replay(
            [
                _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS1]}),
                _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS2]}),
            ]
        )
        path = tmp_path / "test.replay.json.gz"
        save_replay(data, path)

        with gzip.open(path, "rt") as f:
            parsed = json.load(f)
        assert parsed["delta_encoded"] is True
        assert "states" in parsed

    def test_load_full_states_without_delta(self, tmp_path):
        """load_replay handles old files that lack delta_encoded flag."""
        data = _make_replay(
            [
                _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS1]}),
                _make_state({"wall": [WALL_A], "avatar": [AVATAR_POS2]}),
            ]
        )
        original_states = copy.deepcopy(data["states"])

        # Write without delta encoding
        path = tmp_path / "old.replay.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

        loaded = load_replay(path)
        assert loaded["states"] == original_states
        assert "delta_encoded" not in loaded

    def test_save_smaller_than_naive(self, tmp_path):
        """Delta-encoded gzip should be smaller than naive gzip for redundant data."""
        sprites = {"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(200)]
        for i in range(10, 200, 20):
            states[i] = _make_state(
                {"wall": [WALL_A, WALL_B, WALL_C], "avatar": [AVATAR_POS2]}
            )
        data = _make_replay(states)

        naive_path = tmp_path / "naive.json.gz"
        with gzip.open(naive_path, "wb") as f:
            f.write(json.dumps(data).encode("utf-8"))

        delta_path = tmp_path / "delta.replay.json.gz"
        save_replay(copy.deepcopy(data), delta_path)

        naive_size = naive_path.stat().st_size
        delta_size = delta_path.stat().st_size
        assert delta_size < naive_size, (
            f"delta ({delta_size}) should be smaller than naive ({naive_size})"
        )

    def test_save_does_not_mutate_input(self, tmp_path):
        """save_replay must not modify the caller's data dict."""
        sprites = {"wall": [WALL_A], "avatar": [AVATAR_POS1]}
        states = [_make_state(sprites) for _ in range(3)]
        data = _make_replay(states)
        original = copy.deepcopy(data)

        save_replay(data, tmp_path / "test.replay.json.gz")

        assert data == original, "save_replay must not mutate input data"
