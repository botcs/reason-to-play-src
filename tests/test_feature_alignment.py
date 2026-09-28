"""Contract between released session tensors and the fMRI feature reader."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Feature alignment needs optional PyTorch")


def load_alignment_module():
    path = Path(__file__).resolve().parents[1] / "analysis/neural/align_llm.py"
    spec = importlib.util.spec_from_file_location("public_feature_alignment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_session_layout_preserves_stream_layer_play_and_irregular_timing(tmp_path):
    module = load_alignment_module()
    source_dir = tmp_path / "model-Qwen_Qwen3.5-9B" / "compressed"
    subject_dir = source_dir / "sub-13"
    subject_dir.mkdir(parents=True)
    features = torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2)
    # A compressed trace has nonuniform intervals, including across play boundaries.
    payload = {
        "features": features,
        "features_attn": features + 20,
        "features_mlp": features + 40,
        "session": {"num_layers": 2, "model": "Qwen/Qwen3.5-9B"},
        "metadata": [
            {
                "play_id": "run-01-play-0",
                "trial_idx": 0,
                "level_id": 0,
                "realworld_ts": 100.0,
            },
            {
                "play_id": "run-01-play-0",
                "trial_idx": 0,
                "level_id": 0,
                "realworld_ts": 100.35,
            },
            {
                "play_id": "run-01-play-1",
                "trial_idx": 1,
                "level_id": 0,
                "realworld_ts": 104.0,
            },
        ],
    }
    torch.save(payload, subject_dir / "avoidGeorge_vgfmri4.pt")
    assert module.discover_multiturn_layers(source_dir, "sub-13") == [1, 2]
    for stream, offset in [("main", 0), ("attn", 20), ("mlp", 40)]:
        plays = module.load_multiturn_features_for_level(
            source_dir,
            "sub-13",
            "vgfmri4_avoidgeorge",
            0,
            layers=["layer_2"],
            stream=stream,
        )
        assert list(plays) == [0, 1]
        assert plays[0]["metadata"]["play_id"] == "run-01-play-0"
        np.testing.assert_allclose(plays[0]["timestamps"], [100.0, 100.35])
        np.testing.assert_allclose(
            plays[0]["activations"]["layer_2"],
            features[:2, 1, :].float().numpy() + offset,
        )
        np.testing.assert_allclose(plays[1]["timestamps"], [104.0])
