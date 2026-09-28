"""EfficientZero adapters consume human JSON records."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "agents.efficientzero.extract_features",
    "agents.efficientzero.extract_traces",
)


def environment():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "32"),
        OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS", "32"),
    )
    return env


@pytest.fixture
def behavior_fixture(tmp_path):
    from copy import deepcopy
    from test_replay_behavior import recording, write_record

    record = recording("helper", ordinal=0)
    record["game_description"] = """BasicGame
    SpriteSet
        avatar > MovingAvatar color=DARKBLUE
        wall > Immovable color=BLACK
    LevelMapping
        w > wall
    InteractionSet
        avatar wall > stepBack
    TerminationSet
        SpriteCounter stype=avatar limit=0 win=False
"""
    identity = "60735733f24111b59fd666f7"
    record["plays"][0].update(
        _id=identity,
        state_count=3,
        grid_size=[4, 3],
        actions=[["right", 1000.05], ["spacebar", 1000.1]],
    )
    template = record["states"][0]
    record["states"] = []
    for i in range(3):
        state = deepcopy(template)
        state.update(time=i, realworld_ts=1000 + i * 0.05, source_play_id=identity)
        state["sprites"] = {
            "avatar": [
                {
                    "id": 0,
                    "key": "avatar",
                    "col": (35 + i * 6) / 35,
                    "row": 1.0,
                    "x": 35 + i * 5,
                    "y": 35,
                    "color": [20, 40, 180],
                    "color_name": "DARKBLUE",
                    "_uuid": b"sprite-id".hex(),
                    "source_key": "original-avatar",
                    "alive": True,
                }
            ]
        }
        record["states"].append(state)
    record["total_frames"] = 3
    target = tmp_path / "release" / "behavior" / "human"
    path = write_record(target, record)
    return target, path


@pytest.mark.parametrize("module", MODULES)
def test_canonical_load_render_and_stack_without_bson(
    module, behavior_fixture, tmp_path
):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("pygame")
    target, _ = behavior_fixture
    script = r"""
import importlib, importlib.abc, json, sys
class NoBson(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'bson', 'pymongo'}:
            raise AssertionError('Canonical EZ loading attempted BSON: ' + fullname)
sys.meta_path.insert(0, NoBson())
module = importlib.import_module(sys.argv[1])
loader = module.VGDLZStateLoader(sys.argv[2])
play = loader.get_play(subj_id='sub-13', run_id=1, play_id=0)
assert play['_id'] == '60735733f24111b59fd666f7'
assert [s['gt'] for s in loader.decode_zstates(play)] == [0, 1, 2]
assert play['states'][1]['objects']['avatar']['original-avatar']['rect']['pos'] == [41, 35]
assert play['actions'] == [['right', 1000.05], ['spacebar', 1000.1]]
incomplete = dict(play)
incomplete.pop('actions')
if sys.argv[1] == 'agents.efficientzero.extract_traces':
    assert loader.get_play(play_key=play['_id'])['_id'] == play['_id']
    assert module._human_action_indices(play) == [2, 5]
    action_indices = module._human_action_indices
else:
    evaluator = object.__new__(module.EfficientZeroActivationEvaluator)
    assert evaluator._human_action_indices(play) == [2, 5]
    action_indices = evaluator._human_action_indices
    import torch
    from types import SimpleNamespace
    evaluator.config = module.OmegaConf.create({'env': {'obs_shape': [1, 2, 2], 'n_stack': 1}, 'model': {'value_prefix': False}})
    evaluator.device = torch.device('cpu')
    evaluator._observations_from_play = lambda doc: torch.ones((1, 1, 2, 2))
    evaluator._decode_values = lambda values: torch.tensor(0.)
    evaluator.model = SimpleNamespace(
        initial_inference=lambda obs, **kw: (obs, torch.zeros((1, 1, 1)), torch.zeros((1, 6))),
        projection_model=lambda value: value,
        projection_head_model=lambda value: value,
    )
    for outcome in (True, False, None):
        result, _ = evaluator.evaluate_play(dict(play, win=outcome, actions=[]))
        assert result['human_win'] is outcome
    try:
        evaluator.evaluate_play(dict(play, win=-1, actions=[]))
    except ValueError:
        pass
    else:
        raise AssertionError('Terminal-state sentinel accepted as play outcome')
try:
    action_indices(incomplete)
except ValueError as error:
    assert 'original human actions' in str(error)
else:
    raise AssertionError('Missing actions were silently accepted')
frames, actual = loader.load_play_observations(subj_id=13, run_id=1, play_id=0, n_stack=4, as_tensor=False)
assert frames.shape == (3, 4, 84, 84)
assert str(frames.dtype) == 'uint8'
assert frames.sum() > 0
assert 'bson' not in sys.modules
print(json.dumps({'shape': list(frames.shape), 'pixels': int(frames.sum())}))
"""
    # Exercise release-root resolution and imports from outside the checkout.
    completed = subprocess.run(
        [sys.executable, "-c", script, module, str(target.parents[1])],
        cwd=tmp_path,
        env=environment(),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["shape"] == [3, 4, 84, 84]


@pytest.mark.parametrize("module", MODULES)
def test_standalone_and_dataset_root_read_the_same_record(
    module, behavior_fixture, tmp_path
):
    target, path = behavior_fixture
    script = r"""
import importlib, sys
module = importlib.import_module(sys.argv[1])
whole = module.VGDLZStateLoader(sys.argv[2]).get_play(subj_id=13)
single = module.VGDLZStateLoader(sys.argv[3]).get_play(subj_id=13)
assert whole == single
try:
    module.VGDLZStateLoader(sys.argv[3] + '.missing')
except FileNotFoundError as error:
    assert 'JSON replay' in str(error)
else:
    raise AssertionError('Missing JSON input accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, module, str(target), str(path)],
        cwd=tmp_path,
        env=environment(),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("module", MODULES)
def test_direct_extraction_command_from_outside_checkout(module, tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        env=environment(),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--dataset-root" in completed.stdout


@pytest.mark.parametrize("module", MODULES)
def test_recorded_frames_match_reference_pixels(module, behavior_fixture, tmp_path):
    target, _ = behavior_fixture
    script = r"""
import hashlib, importlib, numpy as np, sys
module = importlib.import_module(sys.argv[1])
loader = module.VGDLZStateLoader(sys.argv[2])
recorded = loader.get_play(subj_id=13)
# Reference pixels come from RC_RL b1e33768b9f1d799d7780a7e6556d328be5174ab,
# including its palette, 30-pixel canvas, clipping and OpenCV resize/grayscale.
# The fixture includes walls, fractional avatar positions and an emptied group.
for state in recorded['states']:
    state['objects']['wall'] = {
        f'{row}-{col}': {
            'x': col * 35, 'y': row * 35, 'color': [55, 71, 79],
            'rect': {'pos': [col * 35, row * 35], 'size': [35, 35]},
        }
        for row, line in enumerate(['wwww', 'wA.w', 'wwww'])
        for col, character in enumerate(line) if character == 'w'
    }
recorded['states'][-1]['objects']['avatar'] = {}
expected = {
    (1, None): '7a211e87d106a782389f81f9604620299f2fd288fd8e25b15fe07ca209bd7a89',
    (1, 84): '8ae6a9778696c01f8d434f9a89672e308fc83d76f428eae375ac415dc3259dfe',
    (3, None): '1fa46363db4df21fe3f25c81175add021920bd204a1225069495f0226fd7783f',
    (3, 84): '560f38b312848f94d3cc8c86a92436a8e3af9c555000d229f688f4007490d3e7',
}
for (channels, resize), digest in expected.items():
    actual = loader.render_frames(recorded, resize=resize, num_channels=channels)
    assert actual.dtype == np.uint8
    assert hashlib.sha256(actual.tobytes()).hexdigest() == digest
    assert actual.shape == (3, channels, 90, 120) if resize is None else actual.shape == (3, channels, 84, 84)
for dimensions in ([4, 0], [True, 3], [4.5, 3], [4]):
    try:
        loader.render_frames(dict(recorded, grid_size=dimensions))
    except ValueError as error:
        assert 'grid_size' in str(error)
    else:
        raise AssertionError('Invalid dimensions accepted: ' + str(dimensions))
missing = dict(recorded)
missing.pop('grid_size')
try:
    loader.render_frames(missing)
except ValueError as error:
    assert 'grid_size' in str(error)
else:
    raise AssertionError('Missing recorded dimensions accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, module, str(target)],
        cwd=tmp_path,
        env=environment(),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
