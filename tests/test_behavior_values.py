"""Recorded metadata values remain unambiguous in ordinary JSON."""

from datetime import datetime, timezone
import gzip
import json

import pytest


from data.values import BinaryValue, decode_value, encode_value
from agents.empa.theory_features import iter_regressors
from human.behavior import load_runs
from test_replay_behavior import recording, write_record


def test_value_codec_preserves_keys_binary_subtypes_and_reserved_tags():
    original = {
        273: BinaryValue(b"sprite-id", subtype=4),
        "nested": {"$rtp": "not-a-type-tag"},
        "created": datetime(2023, 1, 1, tzinfo=timezone.utc),
    }
    actual = decode_value(json.loads(json.dumps(encode_value(original))))
    assert actual == original
    assert actual[273].subtype == 4


def test_empa_dataset_root_and_explicit_file_read_identical_typed_records(tmp_path):
    records = [
        {
            "play_key": "original-play",
            "regressors": {"theory_str": ["rule"], "ts": [2.5]},
        }
    ]
    path = tmp_path / "analysis/neural/inputs/theory-regressors.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        gzip.compress(
            json.dumps(
                encode_value(
                    {
                        "schema": "reason-to-play/empa-regressors",
                        "schema_version": 1,
                        "regressors": records,
                    }
                )
            ).encode()
        )
    )
    assert list(iter_regressors(tmp_path)) == list(iter_regressors(path)) == records


@pytest.mark.parametrize("defect", [None, "flag", "method", "date"])
def test_inline_derived_clock_requires_its_evidence(tmp_path, defect):
    record = recording()
    clock = record["plays"][0]["scanner"]
    clock.update(
        clock_only=True,
        scan_start_dt=encode_value(datetime(2021, 1, 1)),
        provenance={"method": "Exact consensus of nine original play clocks"},
    )
    if defect == "flag":
        clock["clock_only"] = False
    elif defect == "method":
        clock["provenance"] = {}
    elif defect == "date":
        clock["scan_start_dt"] = "untyped date"
    path = write_record(tmp_path, record)
    if defect is not None:
        with pytest.raises(ValueError, match="Invalid derived scanner clock"):
            load_runs(path)
    else:
        assert load_runs(path)[13, 1] == decode_value(clock)
