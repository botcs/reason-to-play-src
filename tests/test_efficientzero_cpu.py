"""Inference must not initialize CUDA or the distributed training package."""

from pathlib import Path

import pytest


def test_recurrent_dynamics_runs_on_cpu_without_ray(monkeypatch):
    torch = pytest.importorskip("torch")
    vendor = Path(__file__).resolve().parents[1] / "baselines/vendor/efficientzero"
    monkeypatch.syspath_prepend(str(vendor))
    from ez.agents.models.base_model import DynamicsNetwork

    model = DynamicsNetwork(num_blocks=1, num_channels=8, action_space_size=6).eval()
    state = torch.zeros((2, 8, 6, 6), device="cpu")
    action = torch.tensor([[1], [4]], device="cpu")
    with torch.no_grad():
        predicted = model(state, action)
    assert predicted.shape == state.shape
    assert predicted.device.type == "cpu"
    assert torch.isfinite(predicted).all()
