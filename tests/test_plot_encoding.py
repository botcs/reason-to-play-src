"""Fresh encoding tables need not contain every historical model or ROI."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from analysis.neural import plots as plotting


def fresh_rows():
    return [
        {
            "model": model,
            "variant": "all",
            "fit_condition": condition,
            "subject": "sub-13",
            "layer_idx": layer,
            "ROI": "IFGtriang",
            "band": "main",
            "performance": (0.1 if condition == "with-nuisance" else 0.2) * layer,
        }
        for model in ("qwen35_9b_sugmin", "ddqn_fc1_rerun")
        for condition in ("main-only", "with-nuisance")
        for layer in (1, 2)
    ]


def test_headless_single_subject_unknown_models_and_missing_rois(tmp_path):
    path = tmp_path / "fresh.csv"
    pd.DataFrame(fresh_rows()).to_csv(path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "analysis/neural/plots.py",
            "--csv",
            str(path),
            "--outdir",
            str(tmp_path / "figures"),
            "--models",
            "qwen35_9b_sugmin",
            "ddqn_fc1_rerun",
            "--variant",
            "all",
            "--fit-condition",
            "with-nuisance",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Traceback" not in result.stderr
    assert "identical low and high ylims" not in result.stderr
    for name in ("aggregate", "rois", "groups"):
        output = tmp_path / "figures" / f"encoding_{name}_selected.pdf"
        assert output.read_bytes().startswith(b"%PDF")
        assert output.stat().st_size > 1000
    assert plotting.model_label("qwen35_9b_sugmin") == "qwen35_9b_sugmin"
    other_process_color = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from analysis.neural.plots import model_color; "
            "print(model_color('qwen35_9b_sugmin'))",
        ],
        text=True,
    ).strip()
    assert plotting.model_color("qwen35_9b_sugmin") == other_process_color


def test_mixed_conditions_cannot_be_silently_combined(tmp_path, monkeypatch, capsys):
    path = tmp_path / "mixed.csv"
    pd.DataFrame(fresh_rows()).to_csv(path, index=False)
    monkeypatch.setattr(sys, "argv", ["plot_encoding", "--csv", str(path)])
    with pytest.raises(SystemExit, match="2"):
        plotting.main()
    assert "mixed fit conditions" in capsys.readouterr().err


def test_historical_defaults_and_partial_selection(tmp_path, monkeypatch):
    path = tmp_path / "table.csv"
    calls = []
    for name in ("aggregate", "rois", "groups", "groups_combined", "permutation"):
        monkeypatch.setattr(
            plotting,
            f"plot_{name}",
            lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)),
        )
    models = list(dict.fromkeys(plotting.BASELINES_ORDER + plotting.LLMS_ORDER))
    pd.DataFrame([dict(fresh_rows()[0], model=model) for model in models]).to_csv(
        path, index=False
    )
    monkeypatch.setattr(sys, "argv", ["plot_encoding", "--csv", str(path)])
    plotting.main()
    assert [name for name, _, _ in calls] == [
        "aggregate",
        "aggregate",
        "rois",
        "rois",
        "groups",
        "groups",
        "groups_combined",
        "permutation",
    ]
    assert calls[0][1][1] == plotting.BASELINES_ORDER
    assert calls[1][1][1] == plotting.LLMS_ORDER
    assert calls[5][2] == {"best_only": True}
    assert plotting.model_color("qwen35_9b") == plotting.ALL_COLORS["qwen35_9b"]
    assert (
        plotting.model_label("qwen35_9b", long=True)
        == plotting.LABELS_LONG["qwen35_9b"]
    )

    calls.clear()
    pd.DataFrame([fresh_rows()[0]]).to_csv(path, index=False)
    plotting.main()
    assert [name for name, _, _ in calls] == ["aggregate", "rois", "groups"]
    assert all(args[1] == ["qwen35_9b_sugmin"] for _, args, _ in calls)
    assert Path(calls[2][1][2]).name == "encoding_groups_selected.pdf"
