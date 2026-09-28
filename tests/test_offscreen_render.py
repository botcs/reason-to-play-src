"""Experimental images remain available without an interactive SDL player."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
pytest.importorskip("gym")

from src.vgdl.interfaces.gym.env import VGDLEnv  # noqa: E402


@pytest.fixture
def prohibit_display(monkeypatch):
    """Image observations must work without opening or controlling a display."""

    def unexpected_display(*args, **kwargs):
        raise AssertionError("Image observations must not use the SDL display")

    for name in ("set_mode", "update", "init", "quit"):
        monkeypatch.setattr(pygame.display, name, unexpected_display)
    monkeypatch.setattr(pygame, "init", unexpected_display)
    monkeypatch.setattr(pygame, "quit", unexpected_display)
    monkeypatch.setenv("SDL_VIDEODRIVER", "unavailable-display-for-render-test")


def make_env():
    game = Path(__file__).resolve().parents[1] / "games/bait_vgfmri4_v0"
    return VGDLEnv(
        game_file=str(game / "bait_vgfmri4.txt"),
        level_file=str(game / "bait_vgfmri4_lvl0.txt"),
        obs_type="objects",
        block_size=24,
        seed=7,
    )


def test_rgb_images_preserve_headless_observations_and_rewards(prohibit_display):
    with_images, without_images = make_env(), make_env()
    initial_image, initial_info = with_images.reset(with_img=True)
    no_image, expected_info = without_images.reset()
    assert no_image is None
    assert initial_info == expected_info
    assert initial_image.dtype == np.uint8
    assert initial_image.shape == (288, 504, 3)
    assert len(np.unique(initial_image.reshape(-1, 3), axis=0)) > 1

    for action in [3] * 7 + [1] * 2:
        image, reward, done, truncated, info = with_images.step(action, with_img=True)
        no_image, expected_reward, expected_done, expected_truncated, expected_info = (
            without_images.step(action)
        )
        assert no_image is None
        assert (reward, done, truncated, info) == (
            expected_reward,
            expected_done,
            expected_truncated,
            expected_info,
        )
        assert image.shape == initial_image.shape
    assert not np.array_equal(initial_image, image)
    with_images.close()
    without_images.close()
    # Releasing another environment must not invalidate this image renderer.
    np.testing.assert_array_equal(with_images.render(), image)
    with_images.close()


def test_interactive_mode_rejected_before_renderer_creation(prohibit_display):
    env = make_env()
    assert env.metadata["render.modes"] == ["rgb_array"]
    with pytest.raises(ValueError, match="web app"):
        env.render(mode="human")
    assert env.renderer is None
    env.close()
    env.close()


def test_default_render_and_close_are_offscreen(prohibit_display):
    env = make_env()
    env.close()  # Valid before any image request.
    image = env.render()
    assert image.shape == (288, 504, 3)
    assert env.render(close=True) is None
    assert env.renderer is None
    np.testing.assert_array_equal(env.render(), image)
    assert os.environ["SDL_VIDEODRIVER"] == ("unavailable-display-for-render-test")
    env.close()


def test_base_termination_uses_only_explicit_engine_keys(monkeypatch):
    from src.vgdl.core import Termination

    def unexpected_events(*args, **kwargs):
        raise AssertionError("Engine termination must not poll desktop events")

    monkeypatch.setattr(pygame.event, "peek", unexpected_events)
    termination = Termination(win=False)
    assert termination.is_done(SimpleNamespace(active_keys=[])) == (False, None)
    assert termination.is_done(SimpleNamespace(active_keys=[pygame.K_ESCAPE])) == (
        True,
        False,
    )
