"""Keep EfficientZero arrays bound to original recorded frames and BOLD rows."""

from copy import deepcopy
import builtins
import json
import sys
import types

import numpy as np
import pytest

from data.neural import file_sha256, validate_binding, load_inputs
from analysis.neural import align_efficientzero as ez

REP, VALUE, DYNAMICS = ez.ALL_LAYERS[0], ez.ALL_LAYERS[4], ez.ALL_LAYERS[11]


@pytest.fixture
def context(tmp_path, participant_writer):
    ids = [f"{index:024x}" for index in (1, 2)]
    base = {
        "subject": np.array("sub-01"),
        "tr": np.array(2.0),
        "ar1_corrected": np.array(True),
        "n_volumes": np.array(5),
        "n_games": 1,
        "game_names": np.array(["vgfmri3_bait"]),
        "play_ids": np.array(ids),
        "play_boundaries": np.array([0, 3, 5]),
        "play_n_volumes": np.array([3, 2]),
        "play_game_idx": np.array([0, 0]),
        "play_levels": np.array([0, 1]),
        "tr_run_idx": np.ones(5, dtype=int),
        "tr_game_idx": np.zeros(5, dtype=int),
        "tr_level_idx": np.array([0, 0, 0, 1, 1]),
        "tr_play_idx": np.array([0, 0, 0, 1, 1]),
        "voxel_ts": np.arange(10, dtype=np.float32).reshape(2, 5),
        "mask": np.ones((2, 1, 1), dtype=bool),
        "ez_layers": np.array([], dtype="U64"),
        "has_ez_data": False,
        "fc1_aligned": np.arange(15).reshape(5, 3),
    }
    plays = [
        {
            "_id": pid,
            "subj_id": "1",
            "run_id": 1,
            "level_id": index,
            "game_name": "vgfmri3_bait",
            "states": [
                {"gt": tick, "ts": 100.0 + onset} for tick, onset in enumerate(times)
            ],
        }
        for index, (pid, times) in enumerate(zip(ids, (range(7), range(6, 11))))
    ]
    runs = {(1, 1): {"subj_id": "1", "run_id": 1, "scan_start_ts": 100.0}}
    path = participant_writer(tmp_path / "sub-01", base)
    return ez.prepare_alignment(path, plays=plays, runs=runs), plays, runs


def trace(play):
    frames = len(play.timing["state_timestamps"])
    entries = [
        {
            "timestep": index,
            "phase": "initial",
            "activation": np.array([index + 1, 2 * (index + 1)], dtype=np.float32),
        }
        for index in range(frames)
    ]
    return {
        "metadata": {
            "play_key": play.play_id,
            "subj_id": "1",
            "run_id": 1,
            "play_id": 0,
            "game_name": "bait_vgfmri3",
            "trace_human_actions_only": False,
        },
        "traces": {REP: entries, VALUE: deepcopy(entries)},
    }


def test_frame_clock_sampling_preserves_ar1_rounding_and_scan_truncation(context):
    ctx, _, _ = context
    first, second = ctx.plays.values()
    aligned, gaps = ez.align_trace(trace(first), first, layers=(REP, VALUE))
    assert not gaps
    # np.round uses nearest-even at half-TR ties. AR1 removes scanner volume0.
    np.testing.assert_array_equal(aligned[REP], [[3, 6], [5, 10], [7, 14]])
    other, gaps = ez.align_trace(trace(second), second, layers=(REP, VALUE))
    assert not gaps
    # The last original frame extends beyond retained BOLD and stays excluded.
    np.testing.assert_array_equal(other[REP], [[1, 2], [3, 6]])
    assert other[REP].dtype == np.float32
    assert set(ctx.samples).isdisjoint({"voxel_ts", "fc1_aligned"})
    assert first.sample_stop == second.sample_start


def test_phase_filter_and_sparse_mcts_mean(context):
    ctx, _, _ = context
    play = next(iter(ctx.plays.values()))
    doc = trace(play)
    doc["traces"][VALUE].insert(
        3,
        {"timestep": 2, "phase": "recurrent", "activation": np.array([999, 999])},
    )
    doc["traces"][DYNAMICS] = [
        {"timestep": 2, "activation": np.array([2.0])},
        {"timestep": 2, "activation": np.array([6.0])},
        {"timestep": 4, "activation": np.array([9.0])},
    ]
    values, missing = ez.align_trace(doc, play, layers=(REP, VALUE, DYNAMICS))
    assert not missing
    np.testing.assert_array_equal(values[REP], values[VALUE])
    np.testing.assert_array_equal(values[DYNAMICS], [[4], [3], [0]])
    assert len(ez.STUDY_LAYERS) == 11
    assert DYNAMICS not in ez.STUDY_LAYERS


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("play_key", "0" * 24, "play identity"),
        ("subj_id", "12", "subject"),
        ("run_id", 2, "run"),
        ("game_name", "chase_vgfmri3", "game"),
        ("level_id", 4, "level"),
        ("trace_human_actions_only", True, "Action-only"),
    ],
)
def test_original_identity_is_checked_inside_trace(context, field, value, error):
    ctx, _, _ = context
    play = next(iter(ctx.plays.values()))
    doc = trace(play)
    doc["metadata"][field] = value
    with pytest.raises(ValueError, match=error):
        ez.align_trace(doc, play, layers=(REP,))


@pytest.mark.parametrize("defect", ["duplicate", "unordered", "negative", "bool"])
def test_equal_frame_counts_do_not_hide_timestep_errors(context, defect):
    ctx, _, _ = context
    play = next(iter(ctx.plays.values()))
    doc = trace(play)
    entries = doc["traces"][REP]
    if defect == "duplicate":
        entries[1]["timestep"] = 0
    elif defect == "unordered":
        entries[0], entries[1] = entries[1], entries[0]
    elif defect == "negative":
        entries[0]["timestep"] = -1
    else:
        entries[0]["timestep"] = False
    with pytest.raises(ValueError, match="frame index|frame indices"):
        ez.align_trace(doc, play, layers=(REP,))


def test_incomplete_hooks_are_explicit_gaps(context):
    ctx, _, _ = context
    play = next(iter(ctx.plays.values()))
    doc = trace(play)
    doc["traces"][REP].pop()
    del doc["traces"][VALUE]
    values, gaps = ez.align_trace(doc, play, layers=(REP, VALUE))
    assert not values
    assert gaps[REP] == "incomplete-frame-coverage:6/7"
    assert gaps[VALUE] == "hook-absent-or-no-initial-calls"


@pytest.mark.parametrize("defect", ["run", "level", "game", "frame", "clock"])
def test_base_and_human_samples_must_agree(context, defect):
    ctx, plays, runs = context
    if defect == "run":
        plays[0]["run_id"] = 2
    elif defect == "level":
        plays[0]["level_id"] = 1
    elif defect == "game":
        plays[0]["game_name"] = "chase_vgfmri3"
    elif defect == "frame":
        plays[0]["states"][0]["gt"] = 1
    else:
        plays[0]["states"][1]["ts"] = 90
    with pytest.raises(ValueError):
        ez.prepare_alignment(ctx.subject_dir, plays=plays, runs=runs)


def test_writer_preserves_base_bytes_and_binds_missing_intervals(context, tmp_path):
    ctx, _, _ = context
    original_sha = file_sha256(ctx.subject_dir / "bold.npz")
    first, second = ctx.plays.values()
    good, _ = ez.align_trace(trace(first), first, layers=(REP,))
    array = np.vstack([good[REP], np.zeros((2, 2), dtype=np.float32)])
    path = tmp_path / "efficientzero.npz"
    evidence = ez.write_aligned_features(
        path, ctx, {REP: array}, {REP: [second.play_id]}, source_records=[]
    )
    assert file_sha256(ctx.subject_dir / "bold.npz") == original_sha
    assert evidence["feature_sha256"] == file_sha256(path)
    assert evidence["feature_coverage"]["missing_feature_sample_count"] == 2
    with np.load(path, allow_pickle=False) as output:
        assert validate_binding(output, ctx.samples, ctx.subject_dir)
        assert "ez_layers" not in output and "has_ez_data" not in output
    loaded = load_inputs(
        ctx.subject_dir, path, ez.feature_key(REP).removesuffix("_aligned")
    )
    np.testing.assert_array_equal(loaded["voxel_ts"], np.arange(10).reshape(2, 5))
    coverage = json.loads(loaded["alignment_verification_json"])[-1]["feature_coverage"]
    assert coverage["missing_feature_play_ids"] == [second.play_id]
    array[-1] = 1
    with pytest.raises(ValueError, match="Missing-feature interval contains values"):
        ez.write_aligned_features(path, ctx, {REP: array}, {REP: [second.play_id]})


def test_discovery_uses_original_ids_not_subject_folder(tmp_path):
    pid = "0123456789abcdef01234567"
    path = tmp_path / "vgfmri3_bait/subj12/run1" / f"play0_key{pid}/traces.pt"
    path.parent.mkdir(parents=True)
    path.touch()
    assert ez.discover_traces(tmp_path) == {pid: path}


def test_database_identity_deserializes_without_database_import(tmp_path, monkeypatch):
    import torch

    package, module = types.ModuleType("bson"), types.ModuleType("bson.objectid")
    stored_type = type(
        "ObjectId",
        (),
        {"__module__": "bson.objectid", "__getstate__": lambda self: bytes(range(12))},
    )
    module.ObjectId = stored_type
    package.objectid = module
    monkeypatch.setitem(sys.modules, "bson", package)
    monkeypatch.setitem(sys.modules, "bson.objectid", module)
    path = tmp_path / "traces.pt"
    torch.save(
        {
            "identity": stored_type(),
            "activation": torch.arange(4),
            "metadata": {"play_key": stored_type()},
            "traces": {
                REP: [
                    {"timestep": 0, "phase": "initial", "activation": torch.arange(4)}
                ]
            },
        },
        path,
    )
    monkeypatch.delitem(sys.modules, "bson")
    monkeypatch.delitem(sys.modules, "bson.objectid")
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"bson", "pymongo"}:
            raise AssertionError("Database import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    restored = ez.load_trace_document(path)
    assert str(restored["identity"]) == bytes(range(12)).hex()
    np.testing.assert_array_equal(restored["activation"], np.arange(4))
    from analysis.neural.prepare_inputs import load_ez_traces

    original_reader = load_ez_traces(path, layers=[REP])
    assert str(original_reader["metadata"]["play_key"]) == bytes(range(12)).hex()
    np.testing.assert_array_equal(original_reader["activations"][REP], [[0, 1, 2, 3]])


def test_unrecognized_pickle_globals_are_not_executed(tmp_path):
    import torch

    class Unknown:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    path = tmp_path / "unrecognized.pt"
    torch.save(Unknown(), path)
    with pytest.raises(ValueError, match="Unsupported archived trace global"):
        ez.load_trace_document(path)


def test_per_layer_coverage_follows_requested_hook(context, tmp_path):
    ctx, _, _ = context
    first, second = ctx.plays.values()
    values = {}
    for layer in (REP, VALUE):
        values[layer] = np.concatenate(
            [
                ez.align_trace(trace(play), play, layers=(layer,))[0][layer]
                for play in (first, second)
            ]
        )
    values[VALUE][second.sample_start : second.sample_stop] = 0
    path = tmp_path / "efficientzero.npz"
    ez.write_aligned_features(path, ctx, values, {REP: [], VALUE: [second.play_id]})
    for layer, expected in ((REP, True), (VALUE, False)):
        loaded = load_inputs(
            ctx.subject_dir,
            path,
            ez.feature_key(layer).removesuffix("_aligned"),
        )
        coverage = json.loads(loaded["alignment_verification_json"])[-1][
            "feature_coverage"
        ]
        assert coverage["complete"] is expected
    association = path.with_name(path.name + ".alignment.json")
    document = json.loads(association.read_text())
    del document["feature_coverage_by_layer"][
        ez.feature_key(REP).removesuffix("_aligned")
    ]
    association.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Requested layer lacks"):
        load_inputs(
            ctx.subject_dir,
            path,
            ez.feature_key(REP).removesuffix("_aligned"),
        )


def test_trace_attempt_index_is_not_source_document_ordinal(context):
    ctx, _, _ = context
    play = next(iter(ctx.plays.values()))
    play.document_index = 99
    play.play_index = 0
    doc = trace(play)
    ez.align_trace(doc, play, layers=(REP,))
    doc["metadata"]["play_id"] = 99
    with pytest.raises(ValueError, match="per-level play index"):
        ez.align_trace(doc, play, layers=(REP,))


def test_embedded_scanner_clocks_avoid_second_dataset_scan(context):
    ctx, plays, runs = context
    for play in plays:
        play["scanner"] = deepcopy(runs[(1, 1)])
    prepared = ez.prepare_alignment(ctx.subject_dir, plays=plays)
    assert list(prepared.plays) == list(ctx.plays)
    plays[1]["scanner"]["scan_start_ts"] = 0
    with pytest.raises(ValueError, match="disagree on scanner"):
        ez.prepare_alignment(ctx.subject_dir, plays=plays)


def test_compact_clocks_match_canonical_reader_without_sprite_expansion(
    context, tmp_path, monkeypatch
):
    import gzip
    from human.behavior import iter_plays
    from human import behavior as replay_behavior

    ctx, plays, runs = context
    frames, records = [], []
    for ordinal, play in enumerate(plays):
        stored = {key: value for key, value in play.items() if key != "states"}
        stored.update(
            state_start=len(frames),
            state_count=len(play["states"]),
            source_document_index=ordinal,
            block_size=1,
            grid_size=[1, 1],
            scanner=runs[(1, 1)],
            outcome="win",
            win=True,
        )
        records.append(stored)
        for state in play["states"]:
            frames.append(
                {
                    "time": state["gt"],
                    "realworld_ts": state["ts"],
                    "source_play_id": play["_id"],
                    "sprites": {},
                }
            )
    path = tmp_path / "human.replay.json.gz"
    record = {
        "schema": "reason-to-play/human-replay",
        "schema_version": 1,
        "source": "human",
        "subject": "sub-01",
        "game": "bait_vgfmri3",
        "meta": {"suggestion_level": "elaborate"},
        "game_description": "BasicGame\n  SpriteSet\n    avatar > MovingAvatar\n",
        "total_frames": len(frames),
        "plays": records,
        "states": frames,
    }
    with gzip.open(path, "wt") as stream:
        json.dump(record, stream)
    original = list(iter_plays(path, subject="sub-01"))

    def no_visual_expansion(record):
        raise AssertionError("Sprites must not be expanded for clock alignment")

    monkeypatch.setattr(replay_behavior, "expand_delta_states", no_visual_expansion)
    compact = list(ez._human_clocks(path, "sub-01"))
    for left, right in zip(compact, original, strict=True):
        assert left["_id"] == right["_id"]
        assert left["scanner"] == right["scanner"]
        assert left["states"] == [
            {key: state[key] for key in ("gt", "ts")} for state in right["states"]
        ]
    prepared = ez.prepare_alignment(ctx.subject_dir, path)
    assert list(prepared.plays) == list(ctx.plays)
    frames[0]["source_play_id"] = plays[1]["_id"]
    with gzip.open(path, "wt") as stream:
        json.dump(record, stream)
    with pytest.raises(ValueError, match="frame/play identity differs"):
        list(ez._human_clocks(path, "sub-01"))


def test_compact_clocks_match_real_human_file():
    import os
    from human.behavior import read_record, record_plays, replay_paths

    root = os.environ.get("REASON_TO_PLAY_HUMAN_DATA")
    if not root:
        pytest.skip("Set REASON_TO_PLAY_HUMAN_DATA to check actual human frame clocks")
    path = replay_paths(root)[0]
    record = read_record(path)
    compact = {play["_id"]: play for play in ez._human_clocks(path, record["subject"])}
    count = 0
    for play in record_plays(record):
        clocks = compact.pop(play["_id"])
        assert clocks["states"] == [
            {key: state[key] for key in ("gt", "ts")} for state in play["states"]
        ]
        assert clocks["scanner"] == play["scanner"]
        assert clocks["play_id"] == play["play_id"]
        count += 1
    assert count > 0 and not compact


def test_last_frame_sampling_records_method_and_hook_mapping(context, tmp_path):
    ctx, _, _ = context
    arrays = np.concatenate(
        [
            ez.align_trace(trace(play), play, layers=(REP,), method="last")[0][REP]
            for play in ctx.plays.values()
        ]
    )
    np.testing.assert_array_equal(arrays, [[3, 6], [6, 12], [7, 14], [1, 2], [4, 8]])
    path = tmp_path / "last.npz"
    evidence = ez.write_aligned_features(
        path, ctx, {REP: arrays}, {REP: []}, method="last"
    )
    assert evidence["sampling"]["method"] == "last"
    assert evidence["sampling"]["ar1_volume_offset"] == -1
    assert evidence["hook_to_array"] == {REP: ez.feature_key(REP)}
    with np.load(path, allow_pickle=False) as data:
        assert data["efficientzero_alignment_method"].item() == "last"


def test_canonical_efficientzero_discovery(tmp_path):
    from analysis.neural.encoding import encoding_input_paths

    root = tmp_path / "neural/sub-01"
    paths = [
        root / name
        for name in (
            "bold.npz",
            "samples.npz",
            "nuisance.npz",
            "model-features/efficientzero.npz",
        )
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert (
        encoding_input_paths(
            "sub-01", tmp_path, ez.feature_key(REP).removesuffix("_aligned")
        )
        == paths
    )


@pytest.mark.parametrize("defect", ["missing-association", "missing-layer-coverage"])
def test_generated_efficientzero_requires_coverage_association(
    context, tmp_path, defect
):
    ctx, _, _ = context
    arrays = np.concatenate(
        [
            ez.align_trace(trace(play), play, layers=(REP,))[0][REP]
            for play in ctx.plays.values()
        ]
    )
    path = tmp_path / "efficientzero.npz"
    ez.write_aligned_features(path, ctx, {REP: arrays}, {REP: []})
    association = path.with_name(path.name + ".alignment.json")
    if defect == "missing-association":
        association.unlink()
    else:
        document = json.loads(association.read_text())
        del document["feature_coverage_by_layer"]
        association.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"[Dd]ownload|feature_coverage_by_layer"):
        load_inputs(ctx.subject_dir, path, ez.feature_key(REP).removesuffix("_aligned"))


@pytest.mark.parametrize(
    "alias", ["same-path", "symlink", "hardlink", "association-symlink"]
)
def test_output_cannot_replace_base_archive(context, tmp_path, alias):
    ctx, _, _ = context
    base_path = ctx.subject_dir / "bold.npz"
    original = base_path.read_bytes()
    if alias == "same-path":
        output = base_path
    else:
        output = tmp_path / "efficientzero.npz"
        if alias == "symlink":
            output.symlink_to(base_path)
        elif alias == "hardlink":
            output.hardlink_to(base_path)
        else:
            output.with_name(output.name + ".alignment.json").symlink_to(base_path)
    arrays = np.zeros((len(ctx.samples["tr_play_idx"]), 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must not overwrite participant inputs"):
        ez.write_aligned_features(output, ctx, {REP: arrays}, {REP: []})
    assert base_path.read_bytes() == original


def test_efficientzero_coverage_requires_mapping(context, tmp_path):
    ctx, _, _ = context
    output = tmp_path / "efficientzero.npz"
    arrays = np.zeros((len(ctx.samples["tr_play_idx"]), 2), dtype=np.float32)
    ez.write_aligned_features(output, ctx, {REP: arrays}, {REP: []})
    association = output.with_name(output.name + ".alignment.json")
    document = json.loads(association.read_text())
    for invalid in (None, [], {}, "coverage"):
        document["feature_coverage_by_layer"] = invalid
        association.write_text(json.dumps(document))
        with pytest.raises(
            ValueError,
            match="Per-layer feature coverage must be a mapping|feature_coverage_by_layer",
        ):
            load_inputs(
                ctx.subject_dir,
                output,
                ez.feature_key(REP).removesuffix("_aligned"),
            )
