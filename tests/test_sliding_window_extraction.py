"""Tests for the sliding-window feature extractor.

Covers three layers:

1. Tokenizer probes (Qwen3.5-9B tokenizer only):
   - ``apply_chat_template(tokenize=True, return_tensors='pt', return_dict=False)``
     returns a 2-D tensor we can .tolist() reliably.
   - String-path retokenization matches the canonical tokenization modulo
     a possible single leading BOS.
   - ``_compute_message_boundaries`` produces monotonic per-message offsets
     with ``msg_ends[-1] == len(full_ids)``.

2. Window planning + target assignment (no tokenizer / no model):
   - Uniform strides, snapping to boundaries, edge cases (single window,
     exact fit, sparse boundaries).
   - Target-before-W0-back-half -> W0.
   - Targets in back halves assigned correctly.
   - Each target is assigned to exactly one window.

3. Action compression (no tokenizer / no model):
   - Consecutive same actions collapse to the first.
   - Trial boundaries always survive the compression, even if the action
     repeats across the boundary.

The tokenizer-dependent tests are ``@pytest.mark.skipif`` guarded on
network reachability / HF cache presence so CI in offline environments
still passes; locally they exercise the real Qwen3.5-9B tokenizer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "torch", reason="Feature extraction tests require the optional PyTorch install"
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_eval.human_replay.extract_features import (  # noqa: E402
    Window,
    _assign_targets_to_windows,
    _build_conversation,
    _compress_consecutive_actions,
    _compute_message_boundaries,
    _detect_template_boundaries,
    _plan_windows,
    _real_steps,
)


# ---------------------------------------------------------------------------
# Tokenizer fixture -- loaded once per session
# ---------------------------------------------------------------------------


_QWEN_MODEL_ID = "Qwen/Qwen3.5-9B"


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Real Qwen3.5-9B tokenizer.  Skips if not loadable.

    Tokenizer weights are ~10 MB and cache under ``HF_HOME`` after first
    download; subsequent test runs are offline-friendly.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            _QWEN_MODEL_ID, trust_remote_code=True
        )
    except Exception as e:
        pytest.skip(f"Cannot load {_QWEN_MODEL_ID} tokenizer: {e}")
    return tok


# ---------------------------------------------------------------------------
# 1. Tokenizer probes
# ---------------------------------------------------------------------------


class TestTokenizerContract:
    """Probe the tokenizer API shape we depend on.

    If Transformers changes the default return type of
    ``apply_chat_template(tokenize=True)``, these tests surface the break
    immediately rather than at the first real extraction run.
    """

    _MSGS = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    def test_pt_return_is_2d_tensor(self, qwen_tokenizer):
        """return_tensors='pt' + return_dict=False -> LongTensor (1, seq)."""
        import torch

        pt = qwen_tokenizer.apply_chat_template(
            self._MSGS,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=False,
        )
        assert isinstance(pt, torch.Tensor), (
            f"Expected torch.Tensor, got {type(pt).__name__}. "
            "apply_chat_template contract changed."
        )
        assert pt.ndim == 2 and pt.shape[0] == 1, (
            f"Expected shape (1, seq_len), got {tuple(pt.shape)}"
        )
        assert pt.dtype == torch.long, f"Expected torch.long, got {pt.dtype}"

    def test_string_retokenisation_matches_canonical(self, qwen_tokenizer):
        """String->retokenise(add_special_tokens=False) equals pt ids (modulo BOS).

        The extractor depends on this: it tokenises the chat-template string
        once with offset_mapping to derive per-message boundaries, then uses
        the pt-path tokens as the sequence the model actually sees.  If
        these diverge by more than a leading BOS the extractor refuses to
        continue rather than silently mis-align targets.
        """
        pt = qwen_tokenizer.apply_chat_template(
            self._MSGS,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=False,
        )
        canonical_ids = pt[0].tolist()

        text = qwen_tokenizer.apply_chat_template(
            self._MSGS, tokenize=False, add_generation_prompt=False
        )
        retok = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]

        # Either exact match, or canonical has one extra leading BOS.
        if canonical_ids != retok:
            assert (
                len(canonical_ids) == len(retok) + 1 and canonical_ids[1:] == retok
            ), (
                f"Tokenizer paths disagree beyond a single leading BOS: "
                f"canonical {len(canonical_ids)} tokens, retok {len(retok)} tokens."
            )

    def test_offset_mapping_available(self, qwen_tokenizer):
        """Fast tokenizer exposes offset_mapping -- required for boundary inference."""
        assert qwen_tokenizer.is_fast, (
            "Extractor requires a fast tokenizer (offset_mapping support)"
        )
        text = qwen_tokenizer.apply_chat_template(
            self._MSGS, tokenize=False, add_generation_prompt=False
        )
        enc = qwen_tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        assert "offset_mapping" in enc
        offsets = list(enc["offset_mapping"])
        assert offsets, "offset_mapping was empty"
        # Offsets are monotonic non-decreasing in both start and end.
        for i in range(1, len(offsets)):
            assert offsets[i][0] >= offsets[i - 1][0]
            assert offsets[i][1] >= offsets[i - 1][1]
        # Last offset ends at len(text) (template emits no trailing slack).
        assert offsets[-1][1] == len(text), (
            f"last offset end {offsets[-1][1]} != text len {len(text)}"
        )


# ---------------------------------------------------------------------------
# 2. Message boundary computation
# ---------------------------------------------------------------------------


class TestMessageBoundaries:
    """``_compute_message_boundaries`` invariants."""

    _MSGS_SHORT = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    def test_shapes_and_monotonicity(self, qwen_tokenizer):
        full_ids, msg_starts, msg_ends = _compute_message_boundaries(
            qwen_tokenizer, self._MSGS_SHORT
        )
        assert full_ids.ndim == 1
        assert len(msg_starts) == len(self._MSGS_SHORT)
        assert len(msg_ends) == len(self._MSGS_SHORT)

        # Boundaries are monotonic and tile [0, total_tokens).
        assert msg_starts[0] == 0
        for i in range(len(self._MSGS_SHORT)):
            assert msg_starts[i] < msg_ends[i], (
                f"Message {i} has empty span [{msg_starts[i]}, {msg_ends[i]})"
            )
            if i > 0:
                assert msg_starts[i] == msg_ends[i - 1], (
                    f"Gap / overlap between message {i - 1} and {i}"
                )
        assert msg_ends[-1] == int(full_ids.shape[0])

    def test_canonical_match(self, qwen_tokenizer):
        """full_ids must equal apply_chat_template(return_tensors='pt')."""
        full_ids, _, _ = _compute_message_boundaries(qwen_tokenizer, self._MSGS_SHORT)
        pt = qwen_tokenizer.apply_chat_template(
            self._MSGS_SHORT,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=False,
        )
        assert full_ids.tolist() == pt[0].tolist()

    def test_deepseek_template_dispatch(self, qwen_tokenizer):
        """Swap in DeepSeek V3's template; boundary detection still lines up.

        Tests the DeepSeek code path without needing to download the
        DeepSeek V3.2 weights -- we swap the Qwen tokenizer's
        ``chat_template`` attribute for the DeepSeek one from
        ``transformers_wrapper``.  Token IDs obviously won't match what
        real DeepSeek inference would produce (we keep the Qwen
        vocabulary), but the char->token mapping logic is exercised
        end-to-end.
        """
        from src.llm_eval.shared.transformers_wrapper import (
            DEEPSEEK_V3_CHAT_TEMPLATE,
        )

        original = qwen_tokenizer.chat_template
        qwen_tokenizer.chat_template = DEEPSEEK_V3_CHAT_TEMPLATE
        try:
            full_ids, msg_starts, msg_ends = _compute_message_boundaries(
                qwen_tokenizer, self._MSGS_SHORT
            )
        finally:
            qwen_tokenizer.chat_template = original

        assert len(msg_starts) == len(self._MSGS_SHORT)
        assert len(msg_ends) == len(self._MSGS_SHORT)
        assert msg_ends[-1] == int(full_ids.shape[0])
        # Contiguous tiling.
        for i in range(1, len(self._MSGS_SHORT)):
            assert msg_starts[i] == msg_ends[i - 1]

    def test_unknown_template_raises(self, qwen_tokenizer):
        """An empty template output should raise the unrecognised-marker error."""
        # A deliberately broken template with no known role markers.
        broken = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        original = qwen_tokenizer.chat_template
        qwen_tokenizer.chat_template = broken
        try:
            with pytest.raises(RuntimeError, match="no recognised role markers"):
                _compute_message_boundaries(qwen_tokenizer, self._MSGS_SHORT)
        finally:
            qwen_tokenizer.chat_template = original


class TestTemplateBoundaryDispatch:
    """Unit tests for ``_detect_template_boundaries`` (no tokenizer)."""

    _MSGS = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    def test_chatml_dispatch(self):
        text = (
            "<|im_start|>system\nsys<|im_end|>\n"
            "<|im_start|>user\nu0<|im_end|>\n"
            "<|im_start|>assistant\na0<|im_end|>\n"
            "<|im_start|>user\nu1<|im_end|>\n"
            "<|im_start|>assistant\na1<|im_end|>\n"
        )
        starts = _detect_template_boundaries(text, self._MSGS)
        assert len(starts) == 5
        assert starts[0] == 0
        # Each start is exactly where we expect ``<|im_start|>`` to appear.
        for i, s in enumerate(starts):
            assert text[s : s + len("<|im_start|>")] == "<|im_start|>"

    def test_deepseek_dispatch(self):
        text = (
            "<bos>sys"
            "<｜User｜>u0"
            "<｜Assistant｜></think>a0<｜end▁of▁sentence｜>"
            "<｜User｜>u1"
            "<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"
        )
        starts = _detect_template_boundaries(text, self._MSGS)
        assert len(starts) == 5
        assert starts[0] == 0  # system
        # Boundaries strictly increasing.
        for i in range(1, 5):
            assert starts[i] > starts[i - 1]
        # User messages sit on <｜User｜> markers, assistants on <｜Assistant｜>.
        assert text[starts[1] : starts[1] + len("<｜User｜>")] == "<｜User｜>"
        assert text[starts[2] : starts[2] + len("<｜Assistant｜>")] == "<｜Assistant｜>"
        assert text[starts[3] : starts[3] + len("<｜User｜>")] == "<｜User｜>"
        assert text[starts[4] : starts[4] + len("<｜Assistant｜>")] == "<｜Assistant｜>"

    def test_unknown_format_raises(self):
        with pytest.raises(RuntimeError, match="no recognised role markers"):
            _detect_template_boundaries("just plain text", self._MSGS)

    def test_mixed_markers_raises(self):
        # A pathological template that has BOTH ChatML and DeepSeek markers
        # in its output is ambiguous and should fail loud.
        with pytest.raises(RuntimeError, match="BOTH ChatML and DeepSeek"):
            _detect_template_boundaries(
                "<|im_start|>system\nx<|im_end|>\n<｜User｜>x",
                self._MSGS[:2],
            )


# ---------------------------------------------------------------------------
# 3. Window planning
# ---------------------------------------------------------------------------


class TestPlanWindows:
    def test_single_window_when_total_fits(self):
        # total=100, window=200 -> one window covers [0, 100)
        ws = _plan_windows(
            total_tokens=100,
            window_tokens=200,
            stride_tokens=100,
            msg_starts_sorted=[0, 20, 40, 60, 80],
        )
        assert len(ws) == 1
        assert (ws[0].start, ws[0].end) == (0, 100)

    def test_uniform_stride_no_snapping(self):
        # Boundaries aligned to stride -> clean overlap
        boundaries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        ws = _plan_windows(
            total_tokens=50,
            window_tokens=20,
            stride_tokens=10,
            msg_starts_sorted=boundaries,
        )
        # W0=[0,20) W1=[10,30) W2=[20,40) W3=[30,50)
        assert [w.start for w in ws] == [0, 10, 20, 30]
        assert [w.end for w in ws] == [20, 30, 40, 50]
        assert ws[-1].end == 50  # last covers total

    def test_snapping_to_boundary(self):
        """Ideal stride position falls mid-turn -> snap down to last boundary <= ideal."""
        boundaries = [0, 8, 22, 37, 50]
        ws = _plan_windows(
            total_tokens=50,
            window_tokens=20,
            stride_tokens=10,
            msg_starts_sorted=boundaries,
        )
        # W0=[0,20). Ideal W1.start = 0+10=10. Largest boundary <= 10 is 8.
        # W1=[8,28). Ideal W2.start = 8+10=18. Largest <= 18 is 8 -> same as W0.
        # Progress guard: use next boundary strictly > 8 = 22. W2=[22,42).
        # Ideal W3.start = 22+10=32. Largest <= 32 is 22 -> same.
        # Guard: next > 22 is 37. W3=[37, 50). Covers end. Stop.
        starts = [w.start for w in ws]
        assert starts == [0, 8, 22, 37], starts
        for w in ws:
            assert w.start in boundaries, f"Window start {w.start} not on boundary"
            assert w.length <= 20

    def test_last_window_covers_total(self):
        ws = _plan_windows(
            total_tokens=37,
            window_tokens=10,
            stride_tokens=5,
            msg_starts_sorted=list(range(0, 38)),
        )
        assert ws[-1].end == 37

    def test_stride_must_be_positive(self):
        with pytest.raises(ValueError, match="stride_tokens"):
            _plan_windows(100, 50, 0, [0, 25, 50, 75])


# ---------------------------------------------------------------------------
# 4. Target assignment
# ---------------------------------------------------------------------------


class TestAssignTargets:
    def _uniform_windows(self, total, window, stride):
        """Build uniform (no-snapping) windows for test setup."""
        ws: list[Window] = []
        start = 0
        while True:
            end = min(start + window, total)
            ws.append(Window(start=start, end=end))
            if end >= total:
                break
            start += stride
        return ws

    def test_before_back_half_goes_to_w0(self):
        ws = self._uniform_windows(total=50, window=20, stride=10)
        # W0 back_half_start = 10. Target at 5 (< 10) -> W0.
        assign = _assign_targets_to_windows([5], ws)
        assert assign == [0]

    def test_back_half_assignment(self):
        # Windows: W0=[0,20) W1=[10,30) W2=[20,40) W3=[30,50)
        # Back-half starts: 10, 20, 30, 40
        ws = self._uniform_windows(total=50, window=20, stride=10)
        # Target at 15 is in [10, 20): back half of W0.
        # Target at 25 is in [20, 30): back half of W1.
        # Target at 35 is in [30, 40): back half of W2.
        # Target at 45 is in [40, 50): back half of W3.
        assign = _assign_targets_to_windows([15, 25, 35, 45], ws)
        assert assign == [0, 1, 2, 3]

    def test_uniqueness(self):
        ws = self._uniform_windows(total=50, window=20, stride=10)
        # Every position in [0, 50) must be assignable.
        assigns = _assign_targets_to_windows(list(range(50)), ws)
        assert len(assigns) == 50
        for T, k in enumerate(assigns):
            assert 0 <= k < len(ws), f"T={T} got invalid k={k}"
            # Verify T is inside the assigned window's span (not just back half;
            # front-half targets may fall back to the "last containing window"
            # branch in sparse-boundary cases).
            assert ws[k].start <= T < ws[k].end, (
                f"T={T} outside W{k}=[{ws[k].start},{ws[k].end})"
            )

    def test_empty_windows_raises(self):
        with pytest.raises(ValueError, match="no windows"):
            _assign_targets_to_windows([0], [])


# ---------------------------------------------------------------------------
# 5. Action compression
# ---------------------------------------------------------------------------


def _step(action: str, level: int = 0, attempt: int = 0, step_num: int = 0) -> dict:
    return {
        "step": step_num,
        "level": level,
        "attempt": attempt,
        "action": action,
    }


class TestActionCompression:
    def test_empty(self):
        assert _compress_consecutive_actions([]) == []

    def test_collapses_consecutive_duplicates(self):
        steps = [
            _step("up", step_num=0),
            _step("up", step_num=1),
            _step("up", step_num=2),
            _step("down", step_num=3),
            _step("down", step_num=4),
            _step("left", step_num=5),
        ]
        out = _compress_consecutive_actions(steps)
        assert [s["step"] for s in out] == [0, 3, 5]

    def test_preserves_trial_boundary_with_matching_action(self):
        """If the new trial starts with the same action as the last of the
        previous trial, the compression must still keep the new-trial step
        so ``_mt_user_content`` can detect the boundary."""
        steps = [
            _step("up", level=0, attempt=0, step_num=0),
            _step(
                "up", level=0, attempt=0, step_num=1
            ),  # drop: same (action, level, attempt)
            # New trial starts here, still action=up
            _step("up", level=0, attempt=1, step_num=2),  # keep: different attempt
            _step("up", level=0, attempt=1, step_num=3),  # drop
            # Another new trial, different level
            _step("up", level=1, attempt=0, step_num=4),  # keep: different level
        ]
        out = _compress_consecutive_actions(steps)
        assert [s["step"] for s in out] == [0, 2, 4]

    def test_all_unique_actions_kept(self):
        steps = [
            _step("up", step_num=0),
            _step("down", step_num=1),
            _step("left", step_num=2),
            _step("right", step_num=3),
        ]
        out = _compress_consecutive_actions(steps)
        assert len(out) == 4

    def test_real_steps_filters_synthetic(self):
        steps = [
            {"action": "up", "step": 0},
            {"action": "_level_advance", "step": 1},
            {"action": "down", "step": 2},
            {"action": "_something", "step": 3},
        ]
        out = _real_steps(steps)
        assert [s["step"] for s in out] == [0, 2]


# ---------------------------------------------------------------------------
# 6. End-to-end integration on the bundled smoke-test replay (opt-in)
# ---------------------------------------------------------------------------


_SMOKE_REPLAY = Path(
    "/weka/scratch/schmidt/ssci-jbt/csabi-smoke/replays/"
    "action-only_~_elaborate_sub-09_chase_vgfmri3.human.replay.json.gz"
)


@pytest.mark.skipif(
    not _SMOKE_REPLAY.exists(),
    reason="smoke-test replay not downloaded to this host",
)
class TestSmokeReplayBoundaries:
    """End-to-end tokenisation test against the real sub-09 x chase_vgfmri3 replay.

    Catches regressions in the tokenizer path that unit tests on
    hand-crafted 3-message conversations can miss (e.g. repeated chat
    templates over long conversations that trigger code paths in the
    Jinja renderer).
    """

    @pytest.fixture(scope="class")
    def session(self):
        from src.llm_eval.shared.replay_codec import load_replay

        return load_replay(_SMOKE_REPLAY)

    def test_session_structure(self, session):
        assert session["subject"] == "sub-09"
        assert session["game"] == "chase_vgfmri3"
        assert session["source"] == "human"
        assert session["meta"]["rationale_mode"] == "action-only"
        assert len(session["steps"]) > 0

    def test_boundaries_cover_all_tokens(self, qwen_tokenizer, session):
        real = _real_steps(session["steps"])
        # Avoid tokenising all 3000 turns twice in the test suite; truncate
        # to the first 60 messages (~30 decision points) which still
        # exercises the long-context code path.
        real_small = real[:30]
        msgs = _build_conversation(
            session["system_prompt"], real_small, session["meta"]["rationale_mode"]
        )
        full_ids, msg_starts, msg_ends = _compute_message_boundaries(
            qwen_tokenizer, msgs
        )
        assert msg_ends[-1] == int(full_ids.shape[0])
        # Contiguous tiling invariant.
        for i in range(1, len(msgs)):
            assert msg_starts[i] == msg_ends[i - 1]
