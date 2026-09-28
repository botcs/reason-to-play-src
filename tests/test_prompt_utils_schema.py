"""Tests for prompt_utils._load_replay_gz schema enforcement.

Ensures that missing required top-level, meta, or per-step fields raise
KeyError rather than silently defaulting to 0 / "" / None, and that
replay-only fields are set to None for generative-source files so any
downstream misuse fails loudly.
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.replay_codec import load_replay as _load_replay

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.lrm.prompting.conversation import load_prompts


def _minimal_generative_step(step_num: int, action: str = "wait") -> dict:
    return {
        "step": step_num,
        "level": 0,
        "attempt": 0,
        "action": action,
        "action_log": f"[{step_num}] {action.upper()} -> no change",
        "formatted_obs": f"obs-{step_num}",
        "response": {"rationale": "", "action": action},
        "reward": 0,
        "won": False,
        "lose": False,
        "timeout": False,
        "realworld_ts": 1_700_000_000.0 + step_num,
    }


def _minimal_replay_step(step_num: int, action: str = "wait") -> dict:
    base = _minimal_generative_step(step_num, action)
    base.update(
        {
            "frame_idx": step_num,
            "trial_idx": 0,
            "play_idx": 0,
            "play_id": "sub-01_run1_p0",
            "run": 1,
        }
    )
    return base


def _make_file(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "test.replay.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return path


def _generative_file(tmp_path: Path, steps: list[dict] | None = None) -> Path:
    return _make_file(
        tmp_path,
        {
            "game": "bait_vgfmri4",
            "source": "generative",
            "system_prompt": "sys",
            "prompt_name": "action-only_elaborate",
            "meta": {
                "rationale_mode": "action-only",
                "suggestion_level": "elaborate",
            },
            "steps": steps
            if steps is not None
            else [_minimal_generative_step(0), _minimal_generative_step(1)],
        },
    )


def _replay_file(tmp_path: Path, steps: list[dict] | None = None) -> Path:
    return _make_file(
        tmp_path,
        {
            "game": "vgfmri4_bait",
            "source": "imputed",
            "system_prompt": "sys",
            "prompt_name": "copied-reasoning_elaborate",
            "meta": {
                "rationale_mode": "copied-reasoning",
                "suggestion_level": "elaborate",
                "subject": "sub-01",
                "num_trials": 1,
                "pipeline": "unified",
                "completed": True,
            },
            "steps": steps
            if steps is not None
            else [_minimal_replay_step(0), _minimal_replay_step(1)],
        },
    )


class TestGenerativeSource:
    def test_loads_successfully(self, tmp_path):
        prompts, meta = load_prompts(_generative_file(tmp_path))
        assert len(prompts) == 2
        assert meta["source"] == "generative"

    def test_replay_only_step_fields_are_none(self, tmp_path):
        prompts, _ = load_prompts(_generative_file(tmp_path))
        for p in prompts:
            for f in ("frame_idx", "trial_idx", "play_idx", "play_id", "run"):
                assert p[f] is None, f"generative prompt has non-None {f!r}: {p[f]!r}"

    def test_replay_only_meta_fields_are_none(self, tmp_path):
        _, meta = load_prompts(_generative_file(tmp_path))
        for f in ("subject", "num_trials", "pipeline", "completed"):
            assert meta[f] is None, f"generative meta has non-None {f!r}: {meta[f]!r}"


class TestReplaySource:
    def test_loads_successfully(self, tmp_path):
        prompts, meta = load_prompts(_replay_file(tmp_path))
        assert len(prompts) == 2
        assert meta["source"] == "imputed"
        assert meta["subject"] == "sub-01"
        assert meta["num_trials"] == 1

    def test_replay_only_step_fields_populated(self, tmp_path):
        prompts, _ = load_prompts(_replay_file(tmp_path))
        for i, p in enumerate(prompts):
            assert p["frame_idx"] == i
            assert p["play_id"] == "sub-01_run1_p0"


class TestMissingRequiredFields:
    @pytest.mark.parametrize(
        "drop",
        ["game", "source", "system_prompt", "prompt_name", "meta", "steps"],
    )
    def test_missing_top_level_raises(self, tmp_path, drop):
        path = _generative_file(tmp_path)
        data = _load_replay(path)
        data.pop(drop)
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        with pytest.raises(KeyError, match=drop):
            load_prompts(path)

    @pytest.mark.parametrize("drop", ["rationale_mode", "suggestion_level"])
    def test_missing_meta_raises(self, tmp_path, drop):
        path = _generative_file(tmp_path)
        data = _load_replay(path)
        data["meta"].pop(drop)
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        with pytest.raises(KeyError, match=drop):
            load_prompts(path)

    @pytest.mark.parametrize("drop", ["step", "level", "action", "won", "reward"])
    def test_missing_step_field_raises(self, tmp_path, drop):
        bad_step = _minimal_generative_step(0)
        bad_step.pop(drop)
        path = _generative_file(tmp_path, steps=[bad_step])
        with pytest.raises(KeyError, match=drop):
            load_prompts(path)

    @pytest.mark.parametrize(
        "drop", ["frame_idx", "trial_idx", "play_idx", "play_id", "run"]
    )
    def test_replay_missing_replay_only_step_field_raises(self, tmp_path, drop):
        bad_step = _minimal_replay_step(0)
        bad_step.pop(drop)
        path = _replay_file(tmp_path, steps=[bad_step])
        with pytest.raises(KeyError, match=drop):
            load_prompts(path)

    @pytest.mark.parametrize("drop", ["subject", "num_trials", "pipeline", "completed"])
    def test_replay_missing_replay_only_meta_field_raises(self, tmp_path, drop):
        path = _replay_file(tmp_path)
        data = _load_replay(path)
        data["meta"].pop(drop)
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        with pytest.raises(KeyError, match=drop):
            load_prompts(path)


class TestInvalidSource:
    def test_unknown_source_raises(self, tmp_path):
        path = _generative_file(tmp_path)
        data = _load_replay(path)
        data["source"] = "not-a-real-source"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="unknown source"):
            load_prompts(path)


class TestSyntheticMarkerSteps:
    def test_synthetic_step_has_none_won_and_score(self, tmp_path):
        """Synthetic marker steps (action startswith _) don't carry win/reward
        meaningfully.  We set them to None so any leak through the filter
        fails loud."""
        synthetic = {
            "step": 2,
            "level": 0,
            "attempt": 0,
            "action": "_level_advance",
            "action_log": "advance",
            "response": {},
        }
        path = _generative_file(
            tmp_path,
            steps=[_minimal_generative_step(0), synthetic, _minimal_generative_step(3)],
        )
        prompts, _ = load_prompts(path)
        assert len(prompts) == 3
        assert prompts[1]["action_taken"] == "_level_advance"
        assert prompts[1]["win"] is None
        assert prompts[1]["score"] is None
