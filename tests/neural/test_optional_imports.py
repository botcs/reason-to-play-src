"""Installed data readers must not initialize optional inference runtimes."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


from agents.ddqn import extract_features as ddqn
from analysis.neural import prepare_inputs as align_baselines


def test_canonical_and_npz_readers_do_not_import_tensor_or_cloud_clients(tmp_path):
    script = r"""
import builtins
import importlib
import logging
from pathlib import Path
import sys
import numpy as np

original_import = builtins.__import__
forbidden = {"torch", "bson", "pymongo", "wandb", "boto3"}
def checked_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError("Unexpected optional import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
original_basic_config = logging.basicConfig
logging.basicConfig = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("Import configured process-wide logging")
)
original_path = list(sys.path)
original_handlers = list(logging.getLogger().handlers)
modules = [
    importlib.import_module(name)
    for name in ["analysis.neural.prepare_inputs", "analysis.neural.align_llm", "human.neural.process_bold",
                 "human.neural.bold", "agents.ddqn.extract_features", "analysis.neural.roi"]
]
assert sys.path == original_path
assert logging.getLogger().handlers == original_handlers
logging.basicConfig = original_basic_config
folder = Path("features/sub-13/vgfmri4_bait")
folder.mkdir(parents=True)
np.savez(folder / "level_00.npz", play_3_model_layer_1=np.array([[1., 2.]]),
         play_3_behavioral_timestamps=np.array([10.]))
for module in modules[:2]:
    assert module.discover_llm_layers(Path("features"), "sub-13") == [1]
    loaded = module.load_llm_features_for_level(
        Path("features"), "sub-13", "vgfmri4_bait", 0, layers=["layer_1"])
    np.testing.assert_array_equal(loaded[0]["activations"]["layer_1"], [[1., 2.]])
    np.testing.assert_array_equal(loaded[0]["timestamps"], [10.])
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_editable_baseline_discovery_checks_pinned_sources():
    directory = ddqn.resolve_baseline_directory()
    assert (directory / "VGDLEnv.py").is_file()
    source = ddqn.baseline_source_provenance(directory)
    assert source["distribution"] == "curated-vendor"
    assert source["revision"] == ddqn.RC_RL_EXTRACTION_REVISION


def test_installed_extractor_resolves_included_environment(tmp_path):
    path = tmp_path / "site-packages/agents/ddqn/extract_features.py"
    path.parent.mkdir(parents=True)
    shutil.copyfile(Path(ddqn.__file__), path)
    environment = path.parent / "environment/extraction"
    shutil.copytree(ddqn.DEFAULT_RC_RL_DIR, environment)
    spec = importlib.util.spec_from_file_location("installed_ddqn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.resolve_baseline_directory() == environment.resolve()
    source = module.baseline_source_provenance(module.resolve_baseline_directory())
    assert source["revision"] == ddqn.RC_RL_EXTRACTION_REVISION
    assert (
        source["manifest_sha256"]
        == ddqn.baseline_source_provenance(ddqn.DEFAULT_RC_RL_DIR)["manifest_sha256"]
    )


def test_ez_matching_uses_canonical_frame_counts_and_original_play_order(monkeypatch):
    monkeypatch.setattr(
        align_baselines, "load_ez_traces", lambda *a, **k: {"n_timesteps": 2}
    )
    available = {
        ("vgfmri4_bait", 1, 1): (Path("later/traces.pt"),),
        ("vgfmri4_bait", 1, 0): (Path("earlier/traces.pt"),),
    }
    plays = [
        {
            "_id": identity,
            "subj_id": subject,
            "run_id": 1,
            "game_name": "vgfmri4_bait",
            "start_time": start,
            "states": [{"ts": start}, {"ts": start + 0.05}],
        }
        for identity, subject, start in [
            ("later", "13", 20),
            ("other-subject", "12", 5),
            ("earlier", "13", 10),
        ]
    ]
    assert align_baselines.build_ez_to_behavioral_mapping(
        available, plays, "sub-13"
    ) == {
        ("vgfmri4_bait", 1, 0): "earlier",
        ("vgfmri4_bait", 1, 1): "later",
    }
