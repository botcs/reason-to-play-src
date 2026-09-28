"""Explicit cohort/stream/variant selection precedes encoding aggregation."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from analysis.neural.plots import select_encoding_rows


def experiment():
    return {
        "subjects": ["sub-12", "sub-13"],
        "stream": "main",
        "band": "main",
        "fit_condition": "with-nuisance",
        "model_conditions": [
            {"model": "qwen35_9b", "variant": "all", "expected_layers": [1, 2]},
            {"model": "dsv32", "variant": "compressed", "expected_layers": [1]},
        ],
    }


def table():
    return pd.DataFrame(
        {
            "model": condition["model"],
            "variant": variant,
            "subject": subject,
            "stream": stream,
            "layer_idx": layer,
            "ROI": "IFGtriang",
            "side": "left",
            "partition": partition,
            "band": "main",
            "fit_condition": "with-nuisance",
            "performance": 0.1 + 0.01 * layer,
        }
        for condition in experiment()["model_conditions"]
        for subject in ["sub-01", "sub-12", "sub-13"]
        for variant in ["all", "compressed"]
        for stream in ["main", "attn"]
        for layer in condition["expected_layers"]
        for partition in [0, 1, 2]
    )


def test_config_selects_model_specific_variants_cohort_and_stream():
    selected, models = select_encoding_rows(table(), selection=experiment())
    assert models == ["qwen35_9b", "dsv32"]
    assert len(selected) == 18
    assert set(selected["subject"]) == {"sub-12", "sub-13"}
    assert set(selected["stream"]) == {"main"}
    assert set(selected[selected["model"] == "qwen35_9b"]["variant"]) == {"all"}
    assert set(selected[selected["model"] == "dsv32"]["variant"]) == {"compressed"}


def test_mixed_cohorts_and_streams_need_explicit_selection():
    with pytest.raises(ValueError, match="mixed participant cohorts"):
        select_encoding_rows(table())
    with pytest.raises(ValueError, match="mixed feature streams"):
        select_encoding_rows(table(), subjects=["sub-12", "sub-13"])
    selected, _ = select_encoding_rows(
        table(), stream="main", subjects=["sub-12", "sub-13"], variant="all"
    )
    assert set(selected["stream"]) == {"main"}


def test_known_missing_cell_must_be_explicitly_acknowledged():
    original = table()
    absent = (
        (original["model"] == "qwen35_9b")
        & (original["subject"] == "sub-13")
        & (original["layer_idx"] == 2)
    )
    with pytest.raises(ValueError, match="missing expected"):
        select_encoding_rows(original[~absent], selection=experiment())
    config = experiment()
    config["allow_missing_cells"] = [
        {"model": "qwen35_9b", "subject": "sub-13", "layer_idx": 2}
    ]
    with pytest.warns(UserWarning, match="acknowledged missing"):
        selected, _ = select_encoding_rows(original[~absent], selection=config)
    assert len(selected) == 15


def test_missing_fit_condition_is_never_inferred():
    missing_condition = table().drop(columns="fit_condition")
    with pytest.raises(ValueError, match="requires fit_condition metadata"):
        select_encoding_rows(missing_condition, selection=experiment())
    config = experiment()
    config["historical_fit_condition"] = "with-nuisance"
    with pytest.warns(UserWarning, match="explicitly declared historical"):
        selected, _ = select_encoding_rows(missing_condition, selection=config)
    assert set(selected["fit_condition"]) == {"with-nuisance"}


def test_conflicting_overrides_and_duplicate_cells_are_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        select_encoding_rows(table(), selection=experiment(), stream="attn")
    selected, _ = select_encoding_rows(table(), selection=experiment())
    duplicate = pd.concat([selected, selected.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate encoding cells"):
        select_encoding_rows(duplicate, selection=experiment())


def test_unlabeled_baseline_streams_and_variants_require_exact_declarations():
    original = table()
    baseline = original[
        (original["model"] == "dsv32")
        & (original["variant"] == "compressed")
        & (original["stream"] == "main")
    ].copy()
    baseline["variant"] = None
    baseline["stream"] = None
    data = pd.concat([original[original["model"] != "dsv32"], baseline])
    config = experiment()
    with pytest.raises(ValueError, match="missing expected"):
        select_encoding_rows(data, selection=config)
    config["model_conditions"][1].update(variant=None, stream=None)
    selected, _ = select_encoding_rows(data, selection=config)
    assert len(selected) == 18
    assert selected.loc[selected["model"] == "dsv32", "stream"].isna().all()
    assert selected.loc[selected["model"] == "dsv32", "variant"].isna().all()
    # Explicit null is a selection, not a wildcard matching labeled variants.
    with pytest.raises(ValueError, match="missing expected"):
        select_encoding_rows(original, selection=config)


def test_archived_table_selection_preserves_unknown_fit_metadata():
    config = experiment()
    config["fit_condition"] = None
    with pytest.warns(UserWarning, match="preserves it as unknown"):
        selected, _ = select_encoding_rows(
            table().drop(columns="fit_condition"), selection=config
        )
    assert "fit_condition" not in selected


def test_archived_selection_rejects_changed_input_before_plotting(tmp_path):
    source = tmp_path / "results.csv"
    table().to_csv(source, index=False)
    config = experiment()
    config["input_sha256"] = "0" * 64
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.neural.plots",
            "aggregate",
            "--csv",
            str(source),
            "--selection-config",
            str(selection),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CSV checksum does not match" in result.stderr


def test_selection_entrypoint_runs_outside_checkout(tmp_path):
    csv = tmp_path / "results.csv"
    table().to_csv(csv, index=False)
    config = tmp_path / "selection.json"
    config.write_text(json.dumps(experiment()))
    script = Path(__file__).resolve().parents[1] / "analysis/neural/plots.py"
    output = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "aggregate",
            "--csv",
            str(csv),
            "--selection-config",
            str(config),
            "--outdir",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "encoding_aggregate_selected.pdf").read_bytes().startswith(b"%PDF")
