"""Local replay resume keeps the gameplay path independent of experiment tracking."""

import gzip
import json

import pytest
from omegaconf import OmegaConf

from agents.lrm import play
from agents.lrm.config import Config


def test_local_resume_restores_before_running(monkeypatch, tmp_path):
    recorded = {"game": "bait_vgfmri4", "steps": [{"action": "right"}], "states": []}
    replay_path = tmp_path / "previous.generative.replay.json.gz"
    replay_path.write_bytes(gzip.compress(json.dumps(recorded).encode()))
    calls = []
    state = {"step_num": 12, "total_frames": 60}

    class FakeAgent:
        _log_filepath = "new.generative.replay.json.gz"

        def __init__(self, cfg, timestamp):
            assert cfg.game.resume_from == str(replay_path)

        def restore_from_replay(self, replay):
            assert replay == recorded
            calls.append("restore")
            return 3, state

        def run(self, start_level, resume_state):
            calls.append("run")
            assert start_level == 3
            assert resume_state is state
            return {"outcome": "incomplete", "final_level": 3, "total_steps": 12}

    monkeypatch.setattr(play, "GameplayAgent", FakeAgent)
    config = OmegaConf.structured(Config)
    config.game.resume_from = str(replay_path)
    config.llm.backend = "mock"
    play.main.__wrapped__(config)
    assert calls == ["restore", "run"]


def test_missing_resume_file_fails_before_gameplay(monkeypatch, tmp_path):
    class FakeAgent:
        _log_filepath = "new.generative.replay.json.gz"

        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            pytest.fail("A failed resume must not silently start a new game")

    monkeypatch.setattr(play, "GameplayAgent", FakeAgent)
    config = OmegaConf.structured(Config)
    config.game.resume_from = str(tmp_path / "missing.replay.json.gz")
    config.llm.backend = "mock"
    with pytest.raises(FileNotFoundError):
        play.main.__wrapped__(config)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"source": "human"}, "generative"),
        ({"game": "zelda_vgfmri4"}, "Replay game"),
        ({"meta": {"game": "zelda_vgfmri4"}}, "Replay game"),
        ({"meta": {"rationale_mode": "copied-reasoning"}}, "rationale_mode"),
        ({"suggestion_level": "oracle"}, "suggestion_level"),
    ],
)
def test_resume_rejects_identity_mismatch_before_restoring(monkeypatch, changes, error):
    from agents.lrm.gameplay.agent import GameplayAgent

    cfg = Config()
    cfg.llm.backend = "mock"
    agent = GameplayAgent(cfg)
    monkeypatch.setattr(
        agent.harness,
        "restore_conversation",
        lambda steps: pytest.fail(
            "Identity must be checked before restoring a conversation"
        ),
    )
    replay = {
        "source": "generative",
        "game": cfg.game.game,
        "meta": {
            "rationale_mode": cfg.harness.rationale_mode,
            "suggestion_level": cfg.harness.suggestion_level,
        },
        "steps": [{"action": "right"}],
        **changes,
    }
    with pytest.raises(ValueError, match=error):
        agent.restore_from_replay(replay)
