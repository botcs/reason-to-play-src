"""Missing button measurements must not become observed no-keypress data."""

import numpy as np
import pytest

from analysis.neural.prepare_inputs import extract_behavioral_features


def test_missing_button_fields_rejects_play_even_with_other_input_events():
    play = {
        "_id": "original-play",
        "keydowns": [273],
        "actions": ["up"],
        "states": [{"score": 0}, {"score": 1, "keyPressType": 273}],
    }
    with pytest.raises(ValueError, match="original-play.*no recorded keystate"):
        extract_behavioral_features(play)


@pytest.mark.parametrize("keys", [None, [], {}, [False] * 277, {273: False}])
def test_explicit_no_keypress_measurements_remain_valid(keys):
    result = extract_behavioral_features(
        {"_id": "quiet-play", "states": [{"score": 0, "keystate": keys}]}
    )
    np.testing.assert_array_equal(result["keystates"], np.zeros((1, 5)))
    np.testing.assert_array_equal(result["any_keypress"], [0])


def test_isolated_missing_field_rejected_but_explicit_none_keeps_reduction():
    play = {
        "_id": "partially-recorded-play",
        "states": [
            {"score": 0, "keystate": None},
            {"score": 2, "keystate": {273: True, 32: True}},
            {"score": 1},
        ],
    }
    with pytest.raises(ValueError, match="frame 2 has no recorded keystate"):
        extract_behavioral_features(play)
    play["states"][2]["keystate"] = None
    result = extract_behavioral_features(play)
    np.testing.assert_array_equal(
        result["keystates"], [[0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(result["any_keypress"], [0, 1, 0])
    np.testing.assert_array_equal(result["scores"], [0, 2, 1])
    np.testing.assert_array_equal(result["score_deltas"], [0, 2, -1])
    np.testing.assert_array_equal(result["time_in_play"], [0, 0.5, 1])
