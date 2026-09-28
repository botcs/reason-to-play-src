"""Dataset analysis and extraction work without the optional trainer checkout."""

import configparser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import json

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_REVISION = "29157d4892afd9467b1bd0994de1355086145490"


@pytest.fixture(scope="module")
def checkout_without_training(tmp_path_factory):
    checkout = tmp_path_factory.mktemp("efficientzero-without-training")
    shutil.copytree(
        ROOT / "agents",
        checkout / "agents",
        ignore=shutil.ignore_patterns("training", "__pycache__"),
    )
    shutil.copytree(
        ROOT / "src" / "reason_to_play",
        checkout / "src" / "reason_to_play",
        ignore=shutil.ignore_patterns("__pycache__", "resources"),
    )
    assert not (checkout / "agents/efficientzero/training").exists()
    return checkout


def run_isolated(checkout, script, *arguments):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(checkout), str(checkout / "src"))),
        OMP_NUM_THREADS="32",
        OPENBLAS_NUM_THREADS="32",
        MKL_NUM_THREADS="32",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
    )
    guard = r"""
import importlib.abc
from pathlib import Path
import sys

blocked = {'ray', 'wandb', 'dmc2gym', 'dm_env', 'bson', 'pymongo', 'ez'}
class NoTrainer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in blocked or fullname.startswith('agents.efficientzero.training'):
            raise AssertionError('Optional training dependency requested: ' + fullname)
sys.meta_path.insert(0, NoTrainer())

def no_git(event, arguments):
    if event == 'subprocess.Popen' and Path(arguments[0]).name == 'git':
        raise AssertionError('Analysis/extraction attempted a Git operation')
sys.addaudithook(no_git)
"""
    completed = subprocess.run(
        [sys.executable, "-c", guard + textwrap.dedent(script), *map(str, arguments)],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_readers_and_model_forward_do_not_need_trainer(checkout_without_training):
    pytest.importorskip("torch")
    run_isolated(
        checkout_without_training,
        """
        import torch
        from reason_to_play.data import behavior
        from reason_to_play.analysis.behavioral.baselines import efficientzero_rows
        from agents.efficientzero import extract_features, extract_traces
        from agents.efficientzero.inference.ez.agents.models.base_model import DynamicsNetwork

        for module in (behavior, extract_features, extract_traces):
            assert Path(module.__file__).is_relative_to(Path.cwd()), module.__file__
        assert extract_features.EfficientZero is extract_traces.EfficientZero

        export = Path('episodes/bait_vgfmri4/self_play_episodes.csv')
        export.parent.mkdir(parents=True)
        export.write_text('episode_index,resolved_level,episode_len,win\\n0,0,7,1\\n1,0,9,0\\n')
        row, = efficientzero_rows(export.parent.parent)
        assert row['episode_steps'] == [7, 9]
        assert row['episode_outcomes'] == ['win', 'loss']

        model = DynamicsNetwork(num_blocks=1, num_channels=8, action_space_size=6).eval()
        with torch.no_grad():
            result = model(torch.zeros(2, 8, 6, 6), torch.tensor([[1], [4]]))
        assert result.shape == (2, 8, 6, 6)
        assert result.device.type == 'cpu'
        assert torch.isfinite(result).all()
        assert not Path('agents/efficientzero/training').exists()
        """,
    )


def test_trace_metadata_distinguishes_checkpoint_from_inference_source(
    checkout_without_training,
):
    pytest.importorskip("torch")
    run_isolated(
        checkout_without_training,
        """
        import argparse
        import hashlib
        import json
        from agents.efficientzero.extract_traces import record_trace_provenance

        checkpoint = Path('checkpoint.pt')
        checkpoint.write_bytes(b'checkpoint selected by the caller')
        config = Path('configuration.yaml')
        config.write_text('env: {n_stack: 4}\\n')
        trace = Path('trace.npz')
        trace.write_bytes(b'saved trace bytes')
        args = argparse.Namespace(
            model=str(checkpoint), config_override=str(config),
            trace_layers_json=None, play_key='requested-play',
        )
        metadata_path = record_trace_provenance(args, trace, play_key='selected-play')
        metadata = json.loads(metadata_path.read_text())
        assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert metadata['configuration_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert metadata['trace_sha256'] == hashlib.sha256(trace.read_bytes()).hexdigest()
        assert metadata['model_source']['revision'] == sys.argv[1]
        assert 'training_revision' not in metadata
        assert 'checkpoint_revision' not in metadata
        assert metadata['requested_play_key'] == 'requested-play'
        assert metadata['play_key'] == 'selected-play'
        assert not Path('agents/efficientzero/training').exists()
        """,
        TRAINING_REVISION,
    )


def test_supplied_training_config_works_without_submodule(checkout_without_training):
    pytest.importorskip("omegaconf")
    run_isolated(
        checkout_without_training,
        """
        from omegaconf import OmegaConf
        from agents.efficientzero.prepare_training_config import main

        source = Path('supplied-experiment.yaml')
        source.write_text('env: {game_folder: /original/path, game: bait}\\ntrain: {seed: 17}\\n')
        output = Path('local-experiment.yaml')
        sys.argv = ['prepare_training_config', '--experiment', str(source), '--output', str(output)]
        main()
        result = OmegaConf.load(output)
        assert result.train.seed == 17
        assert result.env.game == 'bait'
        assert Path(result.env.game_folder).is_dir()
        assert OmegaConf.load(source).env.game_folder == '/original/path'
        assert not Path('agents/efficientzero/training').exists()
        """,
    )


def test_optional_trainer_pin_matches_source_lock():
    config = configparser.ConfigParser()
    config.read(ROOT / ".gitmodules")
    trainer = config['submodule "efficientzero-training"']
    assert trainer["path"] == "agents/efficientzero/training"
    assert trainer["url"] == "https://github.com/A-Andrews/EfficientZeroV2.git"
    sources = json.loads((ROOT / "agents/efficientzero/sources.json").read_text())
    assert sources["sources"]["training"]["revision"] == TRAINING_REVISION
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("A source archive has no Git index to inspect")
    entry = subprocess.check_output(
        ["git", "ls-files", "--stage", "--", trainer["path"]], cwd=ROOT, text=True
    )
    mode, revision, stage, path = entry.split()
    assert (mode, revision, stage, path) == (
        "160000",
        TRAINING_REVISION,
        "0",
        trainer["path"],
    )
