"""EfficientZero adapters consume human JSON records."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("get_efficientzero_activations", "get_attention_matrix")


def environment():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(
            str(ROOT / path)
            for path in (
                "baselines/vendor/rc_rl/ez",
                "baselines/vendor/efficientzero",
                "baselines/extraction/efficientzero",
                "src",
                ".",
            )
        ),
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
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
if sys.argv[1] == 'get_attention_matrix':
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


@pytest.mark.parametrize("argument", ["separate", "equals"])
def test_wrapper_resolves_canonical_root_before_changing_cwd(
    argument, tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "ez_wrapper", ROOT / "baselines/run_efficientzero.py"
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.chdir(tmp_path)
    args = (
        ["--dataset-root", "release"]
        if argument == "separate"
        else ["--dataset-root=release"]
    )
    argv, env, _ = wrapper.command("traces", tmp_path, args)
    expected = str(tmp_path / "release")
    assert argv[-1] == (
        expected if argument == "separate" else "--dataset-root=" + expected
    )
    assert str(ROOT / "src") in env["PYTHONPATH"].split(os.pathsep)


@pytest.mark.parametrize("module", MODULES)
def test_recorded_dimensions_replace_ascii_layout_without_changing_pixels(
    module, behavior_fixture, tmp_path
):
    target, _ = behavior_fixture
    script = r"""
import importlib, numpy as np, sys
module = importlib.import_module(sys.argv[1])
loader = module.VGDLZStateLoader(sys.argv[2])
recorded = loader.get_play(subj_id=13)
original = dict(recorded, level_str="wwww\nwA.w\nwwww\n")
original.pop('grid_size')
# A complete recorded state includes walls as well as the moving avatar.
# The earlier minimal source-import fixture deliberately omitted its walls;
# without this, the old renderer keeps walls invented from the ASCII map.
for state in original['states']:
    state['objects']['wall'] = {
        f'{row}-{col}': {
            'x': col * 35, 'y': row * 35, 'color': list(module.colors.BLACK),
            'rect': {'pos': [col * 35, row * 35], 'size': [35, 35]},
        }
        for row, line in enumerate(original['level_str'].splitlines())
        for col, character in enumerate(line) if character == 'w'
    }
# Both paths see an explicit emptied group when the avatar disappears.
original['states'][-1]['objects']['avatar'] = {}
for channels in (1, 3):
    for resize in (None, 84):
        reference = loader.render_frames(original, resize=resize, num_channels=channels)
        actual = loader.render_frames(recorded, resize=resize, num_channels=channels)
        np.testing.assert_array_equal(reference, actual)
        from reason_to_play.features.efficientzero import render_recorded_frames
        direct = render_recorded_frames(recorded, recorded['states'], resize=resize, num_channels=channels)
        np.testing.assert_array_equal(reference, direct)
        if resize is None:
            assert actual.shape[-2:] == (90, 120), actual.shape
for dimensions in ([4, 0], [True, 3], [4.5, 3], [4]):
    try:
        loader.render_frames(dict(recorded, grid_size=dimensions))
    except ValueError as error:
        assert 'grid_size' in str(error)
    else:
        raise AssertionError('Invalid dimensions accepted: ' + str(dimensions))
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
