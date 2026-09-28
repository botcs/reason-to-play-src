"""Inference must not initialize CUDA or the distributed training package."""

import pytest


def test_recurrent_dynamics_runs_on_cpu_without_ray():
    torch = pytest.importorskip("torch")
    from agents.efficientzero.inference.ez.agents.models.base_model import (
        DynamicsNetwork,
    )

    model = DynamicsNetwork(num_blocks=1, num_channels=8, action_space_size=6).eval()
    state = torch.zeros((2, 8, 6, 6), device="cpu")
    action = torch.tensor([[1], [4]], device="cpu")
    with torch.no_grad():
        predicted = model(state, action)
    assert predicted.shape == state.shape
    assert predicted.device.type == "cpu"
    assert torch.isfinite(predicted).all()
