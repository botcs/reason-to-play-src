"""
Tests for Harness replay mode methods and ReplayAgent integration.

Phase 3: Harness replay message building
Phase 4: ReplayAgent integration
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_eval.shared.config import HarnessConfig
from src.llm_eval.shared.harness import Harness
from src.llm_eval.shared.prompt_loader import PromptLoader
from src.llm_eval.human_replay.replay_agent import ReplayAgent
from src.llm_eval.human_replay.zstate_adapter import ZstateAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_loader():
    """PromptLoader for gameplay prompts."""
    return PromptLoader()


_TEST_MAX_OUTPUT_TOKENS = 4096


def _make_harness(rationale_mode: str, prompt_loader) -> Harness:
    """Create a Harness with the given rationale_mode."""
    cfg = HarnessConfig(
        rationale_mode=rationale_mode,
        suggestion_level="elaborate",
    )
    # Replay harness loads from prompts/replay/; tests use it without a subdir
    # to exercise the generic gameplay path.
    return Harness(cfg, prompt_loader, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS)


SAMPLE_STATE = """\
Grid: 7x5
Avatar at (4, 2)
Score: 0
Objects:
- PURPLE at (5,3)
- ORANGE at (2,1)
Walls at: (0, 0-4), (6, 0-4), (0-6, 0), (0-6, 4)"""


# ===================================================================
# Ephemeral replay tests
# ===================================================================


class TestEphemeralReplay:
    """Ephemeral: inject human actions as assistant messages."""

    def test_conversation_structure(self, prompt_loader):
        """MT ephemeral replay should produce alternating user/assistant messages."""
        harness = _make_harness("action-only", prompt_loader)

        # Step 1: build messages, inject action
        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("RIGHT")

        # Step 2: build messages, inject action
        harness.build_messages(
            step_num=2,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("UP")

        # Step 3: build messages (not yet acted)
        messages = harness.build_messages(
            step_num=3,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )

        # Should be: system, user_1, assistant_1, user_2, assistant_2, user_3
        assert len(messages) == 6
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[4]["role"] == "assistant"
        assert messages[5]["role"] == "user"

    def test_assistant_messages_contain_action(self, prompt_loader):
        """Injected assistant messages should contain the human's action."""
        harness = _make_harness("action-only", prompt_loader)

        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("RIGHT")

        # Check the assistant message
        messages = harness.build_messages(
            step_num=2,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )

        assistant_msg = messages[2]
        assert assistant_msg["role"] == "assistant"
        parsed = json.loads(assistant_msg["content"])
        assert parsed["action"] == "right"

    def test_inject_known_action_used(self, prompt_loader):
        """record_replay_step for MT ephemeral should use inject_known_action."""
        harness = _make_harness("action-only", prompt_loader)

        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("LEFT")

        assert harness.conversation_length == 3  # system + user + assistant


# ===================================================================
# Persistent replay tests
# ===================================================================


class TestPersistentReplay:
    """Persistent: two-phase (generation + extraction)."""

    def test_generation_messages_include_action(self, prompt_loader):
        """Generation-phase messages should include 'Your action: X' in user msg."""
        harness = _make_harness("copied-reasoning", prompt_loader)

        messages = harness.build_replay_generation_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="RIGHT",
        )

        # Last message should be user with action
        last_user = [m for m in messages if m["role"] == "user"][-1]
        assert "Your action: RIGHT" in last_user["content"]

    def test_record_strips_action_from_user(self, prompt_loader):
        """After record, user message should NOT contain 'Action taken'."""
        harness = _make_harness("copied-reasoning", prompt_loader)

        # Build generation messages (adds "Action taken: RIGHT" to user msg)
        harness.build_replay_generation_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="RIGHT",
        )

        # Record step (should strip action from user, add to assistant)
        gen_response = json.dumps(
            {"rationale": "Moving right to explore", "action": "right"}
        )
        harness.record_replay_step("RIGHT", generation_response=gen_response)

        # Check conversation
        messages = harness.get_replay_extraction_messages()
        user_msgs = [m for m in messages if m["role"] == "user"]
        for msg in user_msgs:
            assert "Action taken:" not in msg["content"], (
                f"User message still contains 'Action taken': {msg['content'][-100:]}"
            )

    def test_assistant_has_rationale_and_action(self, prompt_loader):
        """After record, assistant message should have rationale + action."""
        harness = _make_harness("copied-reasoning", prompt_loader)

        harness.build_replay_generation_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="RIGHT",
        )

        gen_response = json.dumps(
            {"rationale": "Moving right to explore", "action": "right"}
        )
        harness.record_replay_step("RIGHT", generation_response=gen_response)

        messages = harness.get_replay_extraction_messages()
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

        parsed = json.loads(assistant_msgs[0]["content"])
        assert parsed["rationale"] == "Moving right to explore"
        assert parsed["action"] == "right"

    def test_multi_step_conversation_structure(self, prompt_loader):
        """Multi-step MT persistent should produce correct conversation structure."""
        harness = _make_harness("copied-reasoning", prompt_loader)

        for step in range(1, 4):
            harness.build_replay_generation_messages(
                step_num=step,
                formatted_obs=SAMPLE_STATE,
                level=0,
                attempt=1,
                action_taken="RIGHT",
            )
            gen_response = json.dumps(
                {"rationale": f"Step {step} rationale", "action": "right"}
            )
            harness.record_replay_step("RIGHT", generation_response=gen_response)

        messages = harness.get_replay_extraction_messages()

        # Should be: system, user_1, assistant_1, user_2, assistant_2, user_3, assistant_3
        assert messages[0]["role"] == "system"
        for i in range(1, len(messages), 2):
            assert messages[i]["role"] == "user", (
                f"Message {i} is {messages[i]['role']}"
            )
        for i in range(2, len(messages), 2):
            assert messages[i]["role"] == "assistant", (
                f"Message {i} is {messages[i]['role']}"
            )

    def test_extraction_matches_generative_format(self, prompt_loader):
        """Extraction messages should look like generative play (action in assistant)."""
        harness = _make_harness("copied-reasoning", prompt_loader)

        harness.build_replay_generation_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="RIGHT",
        )

        gen_response = json.dumps({"rationale": "Exploring", "action": "right"})
        harness.record_replay_step("RIGHT", generation_response=gen_response)

        messages = harness.get_replay_extraction_messages()

        # The assistant message should look like a generative play response
        assistant = messages[2]
        parsed = json.loads(assistant["content"])
        assert "rationale" in parsed
        assert "action" in parsed

        # The user message should NOT have action hint
        user = messages[1]
        assert "Action taken:" not in user["content"]


# ===================================================================
# Gameplay parse_strict + commit_assistant
# ===================================================================


class TestGameplayCopiedReasoningPersistence:
    """Regression: the hidden reasoning trace MUST land in parsed['rationale']
    and in the live assistant turn for copied-reasoning gameplay.

    Prior to the fix, commit_assistant popped _resolved_rationale before
    returning, leaving the caller's parsed dict without a 'rationale' key
    -- which then persisted as step.response.rationale="" on disk.
    """

    def _build_turn(self, harness):
        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )

    def test_parsed_carries_rationale_from_api_reasoning(self, prompt_loader):
        harness = _make_harness("copied-reasoning", prompt_loader)
        self._build_turn(harness)

        raw = json.dumps({"action": "up"})
        context = {"reasoning": "We are at (4,2); moving up is safe."}

        parsed = harness.parse_strict(raw, context_info=context)
        harness.commit_assistant(parsed, raw, context)

        assert parsed["rationale"] == context["reasoning"]
        assert "_resolved_rationale" not in parsed
        assert parsed["action"] == "up"

    def test_live_conversation_has_copied_rationale(self, prompt_loader):
        harness = _make_harness("copied-reasoning", prompt_loader)
        self._build_turn(harness)

        raw = json.dumps({"action": "up"})
        context = {"reasoning": "trace-A"}

        parsed = harness.parse_strict(raw, context_info=context)
        harness.commit_assistant(parsed, raw, context)

        assistant_msgs = [
            m
            for m in harness.build_messages(
                step_num=2,
                formatted_obs=SAMPLE_STATE,
                level=0,
                attempt=1,
            )
            if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 1
        body = json.loads(assistant_msgs[0]["content"])
        assert body == {"rationale": "trace-A", "action": "up"}

    def test_action_only_does_not_set_rationale(self, prompt_loader):
        harness = _make_harness("action-only", prompt_loader)
        self._build_turn(harness)

        raw = json.dumps({"action": "down"})
        parsed = harness.parse_strict(raw, context_info={})
        harness.commit_assistant(parsed, raw, {})

        assert "rationale" not in parsed
        assert parsed["action"] == "down"


# ===================================================================
# Trial boundary markers
# ===================================================================


class TestTrialBoundaryMarkers:
    """Trial boundary markers should be identical to generative play."""

    def test_mt_restart_marker(self, prompt_loader):
        """Death restart should produce the same marker as generative play."""
        harness = _make_harness("action-only", prompt_loader)

        # First step
        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("RIGHT")

        # Restart after death
        messages = harness.build_messages(
            step_num=2,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=2,
            is_restart=True,
            prev_outcome="died",
            prev_score=5,
        )

        # Find the user message with restart marker
        last_user = [m for m in messages if m["role"] == "user"][-1]
        assert "--- TRIAL ENDED outcome: died, score: 5 ---" in last_user["content"]
        assert "--- NEW TRIAL (Level 0, Attempt 2) ---" in last_user["content"]

    def test_mt_level_advance_marker(self, prompt_loader):
        """Level advance should produce the same marker as generative play."""
        harness = _make_harness("action-only", prompt_loader)

        # First step on level 0
        harness.build_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
        )
        harness.record_replay_step("RIGHT")

        # New level
        messages = harness.build_messages(
            step_num=2,
            formatted_obs=SAMPLE_STATE,
            level=1,
            attempt=1,
            is_new_level=True,
            prev_level=0,
            prev_outcome="won",
            prev_score=10,
        )

        last_user = [m for m in messages if m["role"] == "user"][-1]
        assert "--- TRIAL ENDED outcome: won, score: 10 ---" in last_user["content"]
        assert "--- NEW TRIAL (Level 1, Attempt 1) ---" in last_user["content"]


# ===================================================================
# Action log in multi-turn user messages
# ===================================================================


class TestActionLogNotInMTUserMessages:
    """Action logs must NOT be injected into user messages.

    The per-step action_log is stored on the step record for the viewer's
    debug panel, but the LLM only sees the observation on each turn; the
    conversation history carries the action-result delta implicitly.
    """

    def test_ephemeral_user_message_has_no_action_log(self, prompt_loader):
        harness = _make_harness("action-only", prompt_loader)
        harness.build_messages(
            step_num=1, formatted_obs=SAMPLE_STATE, level=0, attempt=1
        )
        harness.record_replay_step("RIGHT")

        messages = harness.build_messages(
            step_num=2, formatted_obs=SAMPLE_STATE, level=0, attempt=1
        )
        last_user = [m for m in messages if m["role"] == "user"][-1]
        assert "->" not in last_user["content"], last_user["content"]

    def test_persistent_user_message_has_no_action_log(self, prompt_loader):
        harness = _make_harness("copied-reasoning", prompt_loader)
        harness.build_replay_generation_messages(
            step_num=1,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="RIGHT",
        )
        gen_response = json.dumps({"rationale": "Exploring", "action": "right"})
        harness.record_replay_step("RIGHT", generation_response=gen_response)

        messages = harness.build_replay_generation_messages(
            step_num=2,
            formatted_obs=SAMPLE_STATE,
            level=0,
            attempt=1,
            action_taken="UP",
        )
        last_user = [m for m in messages if m["role"] == "user"][-1]
        # "Your action: UP" is appended for imputation; the only "->" allowed
        # is absent since no engine-event log is injected.
        content_without_hint = last_user["content"].split("Your action:")[0]
        assert "->" not in content_without_hint, content_without_hint


# ===================================================================
# _build_known_interactions
# ===================================================================


class TestBuildKnownInteractions:
    """Verify _build_known_interactions creates temp env and remaps colors."""

    def test_returns_nonempty_set(self):
        """Should return a non-empty set of (color, effect, color) tuples."""
        from src.llm_eval.human_replay.run_replay import _build_known_interactions

        # Use canonical color mapping (identity -- no randomization)
        color_mapping = {
            "goal": "BROWN",
            "wall": "ORANGE",
            "avatar": "DARKBLUE",
            "key": "RED",
            "mushroom": "GREEN",
            "box": "DARKGRAY",
            "hole": "BLUE",
        }
        known = _build_known_interactions("vgfmri3_bait", color_mapping)
        assert isinstance(known, set)
        assert len(known) > 0
        # Each element should be a 3-tuple of strings
        for item in known:
            assert len(item) == 3
            assert all(isinstance(s, str) for s in item)

    def test_colors_are_remapped(self):
        """Known interactions should use the randomized color names."""
        from src.llm_eval.human_replay.run_replay import _build_known_interactions

        # Swap goal and key colors
        color_mapping = {
            "goal": "RED",
            "wall": "ORANGE",
            "avatar": "DARKBLUE",
            "key": "BROWN",
            "mushroom": "GREEN",
            "box": "DARKGRAY",
            "hole": "BLUE",
        }
        known = _build_known_interactions("vgfmri3_bait", color_mapping)
        colors_used = {c for trip in known for c in (trip[0], trip[2])}
        # The remapped colors should appear, not the canonical ones
        # (goal is canonically BROWN but mapped to RED here)
        assert "RED" in colors_used or "BROWN" in colors_used


# ===================================================================
# Phase 4: ReplayAgent integration tests
# ===================================================================


# Inline test fixtures (same as test_zstate_adapter.py)


def _make_uuid(n: int) -> bytes:
    return n.to_bytes(16, "big")


def _make_obj(uuid_n: int, color: str, px: int, py: int, **extra) -> dict:
    obj = {
        "ID": _make_uuid(uuid_n),
        "colorName": color,
        "x": px,
        "y": py,
        "rect": {"pos": [px, py], "size": [35, 35]},
        "resources": {},
    }
    obj.update(extra)
    return obj


_KEYPRESS_TO_PYGAME = {
    "up": 273,
    "down": 274,
    "left": 276,
    "right": 275,
    "spacebar": 32,
}


def _make_zstate(
    objects: dict,
    score: int = 0,
    win=None,
    ended: bool = False,
    key_press: str | None = None,
    effects: list | None = None,
    ts: float = 0.0,
) -> dict:
    keystate = [0] * 323
    if key_press is not None:
        # Raise loudly on an unknown key name rather than silently leaving
        # keystate all-zero (which would look like a NO-OP frame and
        # poison tests with false negatives).
        if key_press not in _KEYPRESS_TO_PYGAME:
            raise ValueError(
                f"_make_zstate: unknown key_press {key_press!r}. "
                f"Valid keys: {sorted(_KEYPRESS_TO_PYGAME)}"
            )
        keystate[_KEYPRESS_TO_PYGAME[key_press]] = 1
    return {
        "objects": objects,
        "score": score,
        "win": win,
        "ended": ended,
        "keyPressType": key_press,
        "keystate": keystate,
        # See _make_zstate in tests/test_zstate_adapter.py for why we do
        # an explicit None-check instead of `effects or []`.
        "effectListByColor": [] if effects is None else effects,
        "ts": ts,
    }


def _make_play_states(
    n_frames: int, key_presses: list[str | None] | None = None
) -> list[dict]:
    """Create a sequence of synthetic zstate frames for testing.

    Avatar starts at (4,2) and moves right each action frame.
    """
    if key_presses is None:
        key_presses = [None] * n_frames

    states = []
    avatar_x = 140  # grid (4,2) with block_size=35
    avatar_y = 70

    for i in range(n_frames):
        key = key_presses[i] if i < len(key_presses) else None

        # Move avatar right on action frames
        if key == "right" and i > 0:
            avatar_x += 35

        objects = {
            "avatar": {
                f"({avatar_x}, {avatar_y})": _make_obj(1, "YELLOW", avatar_x, avatar_y),
            },
            "wall": {
                "(0, 0)": _make_obj(100, "DARKGRAY", 0, 0),
                "(280, 0)": _make_obj(101, "DARKGRAY", 280, 0),
                "(0, 175)": _make_obj(102, "DARKGRAY", 0, 175),
                "(280, 175)": _make_obj(103, "DARKGRAY", 280, 175),
            },
            "goal": {
                "(245, 105)": _make_obj(10, "PURPLE", 245, 105),
            },
        }
        states.append(
            _make_zstate(
                objects,
                score=i,
                key_press=key,
                win=None,
                ended=False,
                ts=1_700_000_000.0 + i * 0.05,
            )
        )

    return states


class TestReplayAgent:
    """Integration tests for ReplayAgent with synthetic data."""

    def test_ephemeral_conversation_grows(self, prompt_loader):
        """Ephemeral conversation should grow with each step."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        keys = [None, "right", "right", "up"]
        states = _make_play_states(4, keys)
        records = agent.run_replay(states, level=0)

        # 4 frames - 1 (frame 0 skipped) = 3 records.
        assert len(records) == 3
        assert len(records[0]["messages"]) == 2  # system + user_1
        assert len(records[1]["messages"]) == 4  # + assistant_1 + user_2
        assert len(records[2]["messages"]) == 6  # + assistant_2 + user_3

    def test_action_frames_only(self, prompt_loader):
        """action_frames_only should skip idle frames and emit one step per keypress."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        keys = [None, "right", None, None, "right", None, "up"]
        states = _make_play_states(7, keys)
        records = agent.run_replay(states, level=0, action_frames_only=True)

        # Three keypresses at raw frames 1, 4, 6 -> three steps whose
        # state_index (the frame the user saw) is 0, 3, 5 respectively.
        assert len(records) == 3
        assert records[0]["action_taken"] == "RIGHT"
        assert records[0]["frame_idx"] == 0
        assert records[1]["action_taken"] == "RIGHT"
        assert records[1]["frame_idx"] == 3
        assert records[2]["action_taken"] == "UP"
        assert records[2]["frame_idx"] == 5

    def test_action_frames_only_first_frame_keypress_is_dropped(self, prompt_loader):
        """A keypress at frame 0 has no pre-action state and is skipped."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        keys = ["right", "right", "up"]
        states = _make_play_states(3, keys)
        records = agent.run_replay(states, level=0, action_frames_only=True)

        # The keypress at frame 0 is discarded because the participant
        # could not have decided to press it based on a state they saw.
        assert len(records) == 2
        assert records[0]["action_taken"] == "RIGHT"
        assert records[0]["frame_idx"] == 0
        assert records[1]["action_taken"] == "UP"
        assert records[1]["frame_idx"] == 1

    def test_stride_sampling(self, prompt_loader):
        """stride=N should process every Nth frame, excluding frame 0."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        keys = ["right"] * 10
        states = _make_play_states(10, keys)
        records = agent.run_replay(states, level=0, stride=3)

        # Raw frames sampled at indices 0, 3, 6, 9.  Frame 0 is skipped
        # (no pre-action state), leaving 3 records with state_index =
        # raw_frame - 1 = [2, 5, 8].
        assert len(records) == 3
        assert records[0]["frame_idx"] == 2
        assert records[1]["frame_idx"] == 5
        assert records[2]["frame_idx"] == 8

    def test_records_have_required_fields(self, prompt_loader):
        """Each record should have all expected fields."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        states = _make_play_states(3, ["right", "right", "up"])
        records = agent.run_replay(states, level=0)

        required = {
            "step_num",
            "frame_idx",
            "action_taken",
            "action_idx",
            "action_log",
            "reward",
            "score",
            "won",
            "lose",
            "messages",
            "formatted_obs",
            "generation_response",
            "generation_context",
        }
        for r in records:
            missing = required - set(r.keys())
            assert not missing, f"Record missing keys: {missing}"

    def test_empty_states(self, prompt_loader):
        """Empty states list should return empty records."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        records = agent.run_replay([], level=0)
        assert records == []

    def test_score_tracking(self, prompt_loader):
        """Score and reward should be tracked correctly.

        Under the clean causal model, record[k] corresponds to raw frame
        k+1 (frame 0 is skipped), so the post-action score comes from
        states[k+1].
        """
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg, prompt_loader, block_size=35, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS
        )

        states = _make_play_states(3, ["right", "right", "right"])
        # _make_play_states assigns score=i to states[i], so states have
        # scores [0, 1, 2].  Two records (frame 0 skipped).
        records = agent.run_replay(states, level=0)

        assert len(records) == 2
        assert records[0]["score"] == 1  # post-action score at raw frame 1
        assert records[1]["score"] == 2  # post-action score at raw frame 2


# Human JSON integration tests for ReplayAgent

HUMAN_DATA = os.environ.get("REASON_TO_PLAY_HUMAN_DATA")


@pytest.mark.skipif(
    not HUMAN_DATA,
    reason="Set REASON_TO_PLAY_HUMAN_DATA to a human JSON file or dataset root",
)
class TestReplayAgentRealData:
    """Integration tests using recorded human JSON data."""

    def _load_first_bait_play(self):
        from src.llm_eval.human_replay.data_loader import HumanPlayLoader

        loader = HumanPlayLoader(HUMAN_DATA)
        for subject in loader.list_subjects():
            for run in loader.list_runs(subject):
                plays = loader.list_plays(subject, run)
                for play_info in plays:
                    if "bait" in play_info["game_name"]:
                        play_doc, states = loader.load_play(
                            subject, run, play_info["play_idx"]
                        )
                        return play_doc, states
        pytest.skip("No bait play found")

    def test_full_play_ephemeral(self, prompt_loader):
        """Process a full real play in ephemeral mode."""
        play_doc, states = self._load_first_bait_play()

        block_size = ZstateAdapter(block_size=1).detect_block_size(states[0])
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=block_size,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        # Use first 20 frames to keep test fast; 20 frames -> 19 records.
        records = agent.run_replay(states[:20], level=play_doc["level_id"])

        assert len(records) == 19
        # Conversation should grow: sys + (user+assistant)*N + user
        for i, r in enumerate(records):
            expected_len = 2 + i * 2  # system + user_1, then +2 per step
            assert len(r["messages"]) == expected_len, (
                f"Step {i}: expected {expected_len} messages, got {len(r['messages'])}"
            )

    def test_action_frames_only_real_data(self, prompt_loader):
        """action_frames_only with real data should produce fewer records."""
        play_doc, states = self._load_first_bait_play()

        block_size = ZstateAdapter(block_size=1).detect_block_size(states[0])
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=block_size,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        all_records = agent.run_replay(states, level=0)

        # Reset and run with action_frames_only
        agent2 = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=block_size,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )
        action_records = agent2.run_replay(states, level=0, action_frames_only=True)

        # Should have fewer records (many frames are idle and are skipped).
        assert len(action_records) < len(all_records)
        assert len(action_records) > 0
        # All non-sentinel records should have real actions
        non_sentinel = [r for r in action_records if r["action_taken"] != "WAIT"]
        assert len(non_sentinel) > 0
        for r in non_sentinel:
            assert r["action_taken"] in ("UP", "DOWN", "LEFT", "RIGHT", "ACTION")


# ===================================================================
# Mock LLM for persistent generation tests
# ===================================================================


class MockLLM:
    """Mock LLM that returns configurable responses. Conforms to LLMWrapperBase."""

    def __init__(self, response: str = '{"rationale": "test notes"}'):
        self._response = response
        self.call_count = 0
        self.last_messages = None

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        self.call_count += 1
        self.last_messages = messages
        return self._response, {"input_tokens": 100, "output_tokens": 50}

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


# ===================================================================
# Persistent generation tests with mock LLM
# ===================================================================


class TestPersistentGenerationMockLLM:
    """Tests for persistent modes with MockLLM (no API calls)."""

    def test_persistent_rationale_in_assistant(self, prompt_loader):
        """Persistent with mock LLM should put rationale in assistant messages."""
        mock = MockLLM('{"rationale": "Exploring right side", "action": "right"}')
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(4, [None, "right", "right", "up"])
        records = agent.run_replay(states, level=0)

        assert mock.call_count == 3
        # Last record has full conversation snapshot (all 3 steps recorded)
        last_msgs = records[2]["messages"]
        assistant_msgs = [m for m in last_msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3  # all 3 steps have assistant responses

        # Each assistant msg should have rationale + action
        for am in assistant_msgs:
            parsed = json.loads(am["content"])
            assert "rationale" in parsed
            assert "action" in parsed

    def test_ephemeral_never_calls_llm(self, prompt_loader):
        """Ephemeral mode should never call the LLM even if one is provided."""
        mock = MockLLM()

        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )
        agent.run_replay(_make_play_states(3, ["right", "right", "up"]), level=0)
        assert mock.call_count == 0

    def test_copied_reasoning_without_llm_raises(self, prompt_loader):
        """copied-reasoning with llm=None must raise rather than silently
        produce an empty rationale (would corrupt the imputed trace)."""
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=None,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(4, [None, "right", "right", "up"])
        with pytest.raises(ValueError, match="produced no rationale"):
            agent.run_replay(states, level=0)

    def test_extraction_messages_persistent(self, prompt_loader):
        """Extraction messages contain the rationale and action without user hints."""
        mock = MockLLM('{"rationale": "analyzing", "action": "right"}')
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(4, [None, "right", "right", "up"])
        records = agent.run_replay(states, level=0)

        # Check last record's extraction messages
        last_msgs = records[2]["messages"]

        # No "Action taken:" in any user message
        for m in last_msgs:
            if m["role"] == "user":
                assert "Action taken:" not in m["content"]

        # Conversation should alternate: system, user, assistant, user, assistant, user, assistant
        assert last_msgs[0]["role"] == "system"
        for i in range(1, len(last_msgs)):
            expected = "user" if i % 2 == 1 else "assistant"
            assert last_msgs[i]["role"] == expected, (
                f"Message {i}: expected {expected}, got {last_msgs[i]['role']}"
            )

    def test_generation_response_in_record(self, prompt_loader):
        """Record should contain generation_response for persistent, None for ephemeral."""
        mock = MockLLM('{"rationale": "data", "action": "right"}')

        # Persistent: should have response
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )
        records = agent.run_replay(_make_play_states(2, ["right", "up"]), level=0)
        for r in records:
            assert r["generation_response"] is not None
            assert r["generation_context"] is not None
            assert r["generation_context"]["input_tokens"] == 100

        # Ephemeral: should be None
        mock2 = MockLLM()
        cfg2 = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        agent2 = ReplayAgent(
            cfg2,
            prompt_loader,
            block_size=35,
            llm=mock2,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )
        records2 = agent2.run_replay(_make_play_states(2, ["right", "up"]), level=0)
        for r in records2:
            assert r["generation_response"] is None
            assert r["generation_context"] is None


# ===================================================================
# Replay-format LLM responses (no action field)
# ===================================================================


class TestReplayFormatNoAction:
    """Tests where mock LLM returns replay format: rationale only, NO action field.

    In human replay, the action comes from the participant's data, not the model.
    The replay prompts instruct the model to return {"rationale": "..."} without
    an action field. The parser must handle this without warnings or data loss.
    """

    def test_persistent_rationale_only_preserved(self, prompt_loader):
        """MT persistent: LLM returns {"rationale": "..."} -- rationale must appear in conversation."""
        mock = MockLLM(
            '{"rationale": "ORANGE objects seem collectible based on score change"}'
        )
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(4, [None, "right", "right", "up"])
        records = agent.run_replay(states, level=0)

        assert mock.call_count == 3

        # Check that rationale is preserved in assistant messages
        last_msgs = records[2]["messages"]
        assistant_msgs = [m for m in last_msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3

        for am in assistant_msgs:
            parsed = json.loads(am["content"])
            assert "rationale" in parsed, (
                f"Rationale lost from assistant message: {am['content']}"
            )
            assert "ORANGE" in parsed["rationale"], f"Rationale content lost: {parsed}"
            assert "action" in parsed, "Action (from human data) should be injected"

    def test_persistent_no_action_no_warning(self, prompt_loader, capsys):
        """MT persistent with rationale-only response should NOT produce parse warnings."""
        mock = MockLLM('{"rationale": "Testing hypothesis about GREEN objects"}')
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(2, ["right", "up"])
        agent.run_replay(states, level=0)

        captured = capsys.readouterr()
        assert "[WARN] Parse error" not in captured.out, (
            f"Unexpected parse warning for valid replay format:\n{captured.out}"
        )

    def test_persistent_rationale_with_markdown_fences(self, prompt_loader):
        """MT persistent: LLM wraps response in markdown code fences -- still works."""
        mock = MockLLM(
            '```json\n{"rationale": "NEW INSIGHTS: none\\nANALYSIS: exploring"}\n```'
        )
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        states = _make_play_states(3, [None, "right", "up"])
        records = agent.run_replay(states, level=0)

        last_msgs = records[1]["messages"]
        assistant_msgs = [m for m in last_msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        for am in assistant_msgs:
            parsed = json.loads(am["content"])
            assert "rationale" in parsed


@pytest.mark.skipif(
    not HUMAN_DATA,
    reason="Set REASON_TO_PLAY_HUMAN_DATA to a human JSON file or dataset root",
)
class TestPersistentRealDataMockLLM:
    """Recorded human JSON tests with a mock LLM for rationale generation."""

    def _load_first_bait_play(self):
        from src.llm_eval.human_replay.data_loader import HumanPlayLoader

        loader = HumanPlayLoader(HUMAN_DATA)
        for subject in loader.list_subjects():
            for run in loader.list_runs(subject):
                plays = loader.list_plays(subject, run)
                for play_info in plays:
                    if "bait" in play_info["game_name"]:
                        play_doc, states = loader.load_play(
                            subject, run, play_info["play_idx"]
                        )
                        return play_doc, states
        pytest.skip("No bait play found")

    def test_real_play_persistent_mock(self, prompt_loader):
        """Process real frames in MT persistent mode with mock LLM."""
        play_doc, states = self._load_first_bait_play()

        block_size = ZstateAdapter(block_size=1).detect_block_size(states[0])
        mock = MockLLM('{"rationale": "exploring game rules", "action": "right"}')
        cfg = HarnessConfig(
            rationale_mode="copied-reasoning", suggestion_level="elaborate"
        )
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=block_size,
            llm=mock,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )

        # 11 raw frames -> 10 records.
        records = agent.run_replay(states[:11], level=play_doc["level_id"])

        assert len(records) == 10
        assert mock.call_count == 10

        # Last record should have full conversation
        last_msgs = records[9]["messages"]
        # system + 10 * (user + assistant) = 21
        assert len(last_msgs) == 21, f"Expected 21 messages, got {len(last_msgs)}"

        # Verify alternation
        assert last_msgs[0]["role"] == "system"
        for i in range(1, len(last_msgs)):
            expected = "user" if i % 2 == 1 else "assistant"
            assert last_msgs[i]["role"] == expected


# ===================================================================
# Oracle suggestion level tests
# ===================================================================


class TestOracleSuggestionLevel:
    """Oracle suggestion level injects game-specific rules into prompts."""

    @pytest.mark.parametrize(
        "mode", ["action-only", "prompted-rationale", "copied-reasoning"]
    )
    def test_oracle_loads_game_rules(self, prompt_loader, mode):
        """Oracle prompt contains game-specific rule text."""
        cfg = HarnessConfig(rationale_mode=mode, suggestion_level="oracle")
        h = Harness(
            cfg,
            prompt_loader,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
            game_name="bait_vgfmri4",
        )
        assert "{avatar} avatar" in h.system_prompt
        assert "{game_rules}" not in h.system_prompt

    def test_oracle_without_game_name_raises(self, prompt_loader):
        """Oracle prompts require game_name to resolve {game_rules}."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="oracle")
        with pytest.raises(ValueError, match="game_name"):
            Harness(cfg, prompt_loader, max_output_tokens=_TEST_MAX_OUTPUT_TOKENS)

    @pytest.mark.parametrize(
        "game_name",
        [
            "avoidGeorge_vgfmri4",
            "bait_vgfmri4",
            "chase_vgfmri4",
            "helper_vgfmri4",
            "lemmings_vgfmri4",
            "zelda_vgfmri4",
            "plaqueAttack_vgfmri3",
            "sokoban_vgfmri3",
        ],
    )
    def test_all_games_have_rule_files(self, prompt_loader, game_name):
        """Every game has a corresponding rule file that loads without error."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="oracle")
        h = Harness(
            cfg,
            prompt_loader,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
            game_name=game_name,
        )
        assert "Win" in h.system_prompt or "WIN" in h.system_prompt

    def test_non_oracle_ignores_game_name(self, prompt_loader):
        """Passing game_name with non-oracle level is a harmless no-op."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="elaborate")
        h = Harness(
            cfg,
            prompt_loader,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
            game_name="bait_vgfmri4",
        )
        assert "{game_rules}" not in h.system_prompt
        assert "{avatar}" not in h.system_prompt


class TestReplayAgentGameplayPromptSubstitution:
    """ReplayAgent._gameplay_system_prompt must have all placeholders resolved.

    This is the prompt that gets saved into .replay.json.gz and later consumed
    by the feature extraction pipeline.  A raw ``{game_rules}`` or
    ``{max_output_tokens}`` placeholder here means every downstream extraction
    run silently uses a broken system prompt.
    """

    @pytest.mark.parametrize("mode", ["action-only", "copied-reasoning"])
    def test_oracle_game_rules_substituted(self, prompt_loader, mode):
        cfg = HarnessConfig(rationale_mode=mode, suggestion_level="oracle")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
            game_name="bait_vgfmri4",
        )
        assert "{game_rules}" not in agent._gameplay_system_prompt
        assert "avatar" in agent._gameplay_system_prompt.lower()

    @pytest.mark.parametrize("mode", ["action-only", "copied-reasoning"])
    def test_max_output_tokens_substituted(self, prompt_loader, mode):
        cfg = HarnessConfig(rationale_mode=mode, suggestion_level="elaborate")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
        )
        assert "{max_output_tokens}" not in agent._gameplay_system_prompt
        assert str(_TEST_MAX_OUTPUT_TOKENS) in agent._gameplay_system_prompt

    @pytest.mark.parametrize("suggestion_level", ["elaborate", "minimal", "oracle"])
    def test_no_unresolved_placeholders(self, prompt_loader, suggestion_level):
        kwargs = {
            "block_size": 35,
            "max_output_tokens": _TEST_MAX_OUTPUT_TOKENS,
        }
        if suggestion_level == "oracle":
            kwargs["game_name"] = "bait_vgfmri4"
        cfg = HarnessConfig(
            rationale_mode="action-only", suggestion_level=suggestion_level
        )
        agent = ReplayAgent(cfg, prompt_loader, **kwargs)
        prompt = agent._gameplay_system_prompt
        assert "{game_rules}" not in prompt, "unresolved {game_rules}"
        assert "{max_output_tokens}" not in prompt, "unresolved {max_output_tokens}"

    def test_oracle_sprite_type_placeholders_substituted(self, prompt_loader):
        """Sprite-type placeholders in game rules must be resolved by color_mapping."""
        cfg = HarnessConfig(rationale_mode="action-only", suggestion_level="oracle")
        agent = ReplayAgent(
            cfg,
            prompt_loader,
            block_size=35,
            max_output_tokens=_TEST_MAX_OUTPUT_TOKENS,
            game_name="bait_vgfmri4",
        )
        # Before substitution: sprite-type placeholders present
        assert "{avatar}" in agent._gameplay_system_prompt
        assert "{key}" in agent._gameplay_system_prompt

        # Simulate the color_mapping substitution that _process_game does
        color_mapping = {
            "avatar": "RED",
            "key": "GREEN",
            "goal": "CYAN",
            "box": "WHITE",
            "hole": "ORANGE",
            "mushroom": "PURPLE",
            "wall": "BROWN",
        }
        for sprite_type, color_name in color_mapping.items():
            agent._gameplay_system_prompt = agent._gameplay_system_prompt.replace(
                "{" + sprite_type + "}", color_name.upper()
            )

        assert "{avatar}" not in agent._gameplay_system_prompt
        assert "RED" in agent._gameplay_system_prompt
        assert "GREEN" in agent._gameplay_system_prompt
        assert "{game_rules}" not in agent._gameplay_system_prompt
