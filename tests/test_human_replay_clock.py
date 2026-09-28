"""Human engine ticks must survive conversion into the replay display format."""

from agents.lrm.prompting.observations import convert_zstate_to_viewer


def test_human_frame_clock_uses_original_engine_ticks():
    frames = [
        {"objects": {}, "gt": tick, "time": 0, "ts": 1_000.0 + tick * 0.05}
        for tick in range(6)
    ]
    converted = [convert_zstate_to_viewer(frame, 35) for frame in frames]
    assert [frame["time"] for frame in converted] == list(range(6))
    assert frames[-1]["ts"] == 1_000.25
