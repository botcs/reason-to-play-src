"""Batch continuation must preserve successes without hiding failed sessions."""

import importlib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")


@pytest.mark.parametrize("backend", ["generic", "v32", "v4"])
@pytest.mark.parametrize(
    "outcomes", [("success", "fail", "skip"), ("fail",), ("skip",)]
)
def test_completed_batch_distinguishes_failed_and_skipped_sessions(
    tmp_path, monkeypatch, capsys, backend, outcomes
):
    if backend != "generic":
        pytest.importorskip("transformers")
        pytest.importorskip("safetensors")
    suffix = "" if backend == "generic" else f"_deepseek_{backend}"
    module = importlib.import_module(f"agents.lrm.features.extract{suffix}")
    runtime = "_load_runtime" if backend == "generic" else f"_load_{backend}_runtime"
    monkeypatch.setattr(module, runtime, lambda *args: (None, None, None))
    monkeypatch.setattr(module, "IS_MAIN", True)
    monkeypatch.setattr(module, "IS_DISTRIBUTED", False)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = []
    for index, outcome in enumerate(outcomes):
        path = inputs / f"{index}_{outcome}.replay.json.gz"
        path.touch()
        paths.append(path)
    visited = []
    retained = tmp_path / "successful-output.pt"

    def extract(path, *args, **kwargs):
        visited.append(path)
        if "fail" in path.name:
            raise ValueError("bad session")
        if "skip" in path.name:
            return None
        retained.write_text("successful feature artifact")
        return {
            "output_file": str(retained),
            "total_tokens": 8,
            "n_targets": 1,
            "n_windows": 1,
            "peak_gpu_mem_mb": 0,
            "total_forward_s": 0.1,
            "session_wall_s": 0.2,
            "out_mb": 0.001,
        }

    monkeypatch.setattr(module, "extract_for_session", extract)
    config = {
        "prompts": str(inputs / "*.replay.json.gz"),
        "model": "test/model",
        "output_dir": str(tmp_path / "outputs"),
        "continue_on_session_error": True,
    }
    if backend != "generic":
        config.update(ckpt_path="test/shards", ds_config="test/config.json")
    if "fail" in outcomes:
        with pytest.raises(
            RuntimeError, match="Feature extraction failed for 1 session"
        ):
            module.main.__wrapped__(OmegaConf.create(config))
    else:
        module.main.__wrapped__(OmegaConf.create(config))
    assert visited == paths
    assert retained.exists() == ("success" in outcomes)
    output = capsys.readouterr().out
    assert f"completed={outcomes.count('success')}" in output
    assert f"skipped={outcomes.count('skip')}" in output
    assert f"failed={outcomes.count('fail')}" in output
    if "success" in outcomes:
        assert len(list((tmp_path / "outputs").rglob("benchmark_*.csv"))) == 1


def test_default_stops_on_first_error(tmp_path, monkeypatch):
    from agents.lrm.features import extract as module
    from agents.lrm.config import ExtractionConfig

    assert ExtractionConfig().continue_on_session_error is False
    for name in ("a_fail", "b_other"):
        (tmp_path / f"{name}.replay.json.gz").touch()
    visited = []

    def extract(path, *args, **kwargs):
        visited.append(Path(path).name)
        raise ValueError("bad session")

    monkeypatch.setattr(module, "_load_runtime", lambda cfg: (None, None, None))
    monkeypatch.setattr(module, "extract_for_session", extract)
    config = OmegaConf.create(
        {"prompts": str(tmp_path / "*.replay.json.gz"), "model": "test/model"}
    )
    with pytest.raises(ValueError, match="bad session"):
        module.main.__wrapped__(config)
    assert visited == ["a_fail.replay.json.gz"]
