"""Preserve baseline outcome, chronology, and cross-cohort contracts."""

import json
import pickle
from types import SimpleNamespace

import pytest

from analysis.behavioral.baselines import (
    ddqn_rows,
    efficientzero_rows,
    empa_rows,
)
from agents.ddqn.export_history import export
from agents.empa.import_results import convert


def test_ez_orders_episodes_and_excludes_warmup(tmp_path):
    game = tmp_path / "vgfmri4_bait"
    game.mkdir()
    (game / "self_play_episodes.csv").write_text(
        "episode_index,resolved_level,episode_len,win\n2,0,4,1\n1,9,30,0\n0,0,12,0\n"
    )
    (row,) = efficientzero_rows(tmp_path)
    assert row["episode_steps"] == [12, 4]
    assert row["episode_outcomes"] == ["loss", "win"]
    assert row["episode_frames"] == [None, None]
    assert row["level"] == 0
    assert row["cohort"] == "vgfmri4"


def test_ddqn_keeps_run_identity_and_rejects_ambiguous_outcomes(tmp_path):
    path = tmp_path / "ddqn.json"
    record = {
        "run_id": "run-a",
        "game_raw": "vgfmri3_bait",
        "episodes": [
            {"level": 0, "steps": 4, "win": 1},
            {"level": 10, "steps": 5, "win": 0},
        ],
    }
    path.write_text(json.dumps([record]))
    (row,) = ddqn_rows(path)
    assert row["instance_id"] == "run-a"
    assert row["episode_steps"] == [4]
    record["episodes"][0]["win"] = -1
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="binary baseline"):
        ddqn_rows(path)


def test_empa_one_based_levels_and_three_valued_outcomes(tmp_path):
    directory = tmp_path / "vgfmri3_bait/1/dumps/interaction_data"
    directory.mkdir(parents=True)
    (directory / "lvl_1.json").write_text(
        json.dumps(
            {
                "episode_summaries": [
                    {"steps": 8, "outcome": "fmri_timeout"},
                    {"steps": 3, "outcome": "win"},
                ]
            }
        )
    )
    (row,) = empa_rows(tmp_path)
    assert row["level"] == 0
    assert row["instance_id"] == "trial1"
    assert row["episode_outcomes"] == ["fmri_timeout", "win"]


def test_ddqn_export_uses_unsampled_scan_and_deduplicates_runs():
    class Run:
        id = "run-a"
        state = "finished"
        config = {"game_name": "vgfmri3_bait", "random_seed": 7}

        def scan_history(self, **kwargs):
            assert kwargs["keys"] == ["episode_win", "episode_length", "current_level"]
            # Over 500 entries: historical r.history() silently samples this.
            yield from (
                {"episode_win": i % 2, "episode_length": 10, "current_level": 0}
                for i in range(1100)
            )

        def history(self, **kwargs):
            pytest.fail("Sampled history must not be used")

    api = SimpleNamespace(sweep=lambda _: SimpleNamespace(runs=[Run()]))
    (record,) = export(api, "test/project", ["first", "second"])
    assert len(record["episodes"]) == 1100
    assert record["seed"] == 7


def test_empa_conversion_preserves_episodes_in_flat_dataset_paths(tmp_path):
    source = tmp_path / "source"
    directory = source / "vgfmri4_avoidgeorge/15/dumps/interaction_data"
    directory.mkdir(parents=True)
    episodes = [{"steps": 3, "outcome": "loss"}, {"steps": 12, "outcome": "win"}]
    (directory / "lvl_2.pkl").write_bytes(pickle.dumps({"episode_summaries": episodes}))
    output = tmp_path / "output"
    assert convert(source, output) == 1
    path = output / "avoidGeorge_vgfmri4/trial-15/level-01.json"
    document = json.loads(path.read_text())
    assert document["episode_summaries"] == episodes
    (row,) = empa_rows(output)
    assert (row["game"], row["cohort"], row["instance_id"], row["level"]) == (
        "avoidgeorge",
        "vgfmri4",
        "trial15",
        1,
    )
    assert row["episode_steps"] == [3, 12]
    document["level"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metadata disagrees"):
        empa_rows(output)


def test_ez_release_and_source_alias_cannot_double_count(tmp_path):
    source = tmp_path / "vgfmri4_bait"
    released = tmp_path / "bait_vgfmri4"
    source.mkdir()
    released.mkdir()
    data = "episode_index,resolved_level,episode_len,win\n0,0,12,1\n"
    (released / "episodes.csv").write_text(data)
    assert efficientzero_rows(tmp_path)[0]["episode_steps"] == [12]
    (source / "self_play_episodes.csv").write_text(data)
    with pytest.raises(ValueError, match="Duplicate EfficientZero"):
        efficientzero_rows(tmp_path)
