"""DDQN resources and engine revisions remain usable in the same installation."""

import importlib.util
import subprocess
import sys

import pytest


def test_training_configuration_does_not_initialize_optional_runtimes(tmp_path):
    script = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'pygame', 'wandb'}:
        raise AssertionError('Unexpected runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from agents.ddqn.train import build_parser
config = build_parser().parse_args(['--no_wandb', '--game_name', 'vgfmri4_bait'])
assert config.threads == 32
assert (config.game_dir / 'vgfmri4_bait.txt').is_file()
assert (config.game_dir / 'vgfmri4_bait_lvl0.txt').is_file()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "order", [("training", "extraction"), ("extraction", "training")]
)
def test_engine_revisions_do_not_share_modules_or_sprite_classes(tmp_path, order):
    for dependency in ("pygame", "skimage", "scipy", "IPython", "termcolor"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"DDQN environment dependency unavailable: {dependency}")
    script = """
from pathlib import Path
import importlib
import sys
from agents.ddqn.extract_features import baseline_source_provenance
environments = []
for revision in sys.argv[1:]:
    prefix = 'agents.ddqn.environment.' + revision
    module = importlib.import_module(prefix + '.VGDLEnv')
    base = Path(module.__file__).parent
    baseline_source_provenance(base)
    env = module.VGDLEnv('vgfmri4_bait', str(base / 'all_games'))
    for action in [0, 1, 2, 0]:
        env.step(action)
    game = env.current_env._game
    assert type(game).__module__.startswith(prefix + '.')
    for sprites in game.sprite_groups.values():
        assert all(type(sprite).__module__.startswith(prefix + '.') for sprite in sprites)
    environments.append(env)
assert type(environments[0]) is not type(environments[1])
assert not set(('vgdl', 'VGDLEnv', 'rl_models', 'utils', 'ontology', 'core',
                'tools', 'colors', 'util')).intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script, *order],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
