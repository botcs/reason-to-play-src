"""Tests for the ExpDB / S3 integration in extract_features.

Covers the pure-data slot/key derivation helper plus the rank-0 lifecycle
(claim -> upload -> complete; claim -> fail on extraction error;
non-claimable -> early None) using a mock ExpDB client.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_eval.human_replay.extract_features import (  # noqa: E402
    _derive_slot_and_s3,
)
from src.llm_eval.shared.config import ExtractionConfig  # noqa: E402


# -- _derive_slot_and_s3 ------------------------------------------------


def _session(**overrides):
    base = {
        "game": "bait_vgfmri4",
        "subject": "sub-01",
        "meta": {"suggestion_level": "elaborate", "rationale_mode": "action-only"},
    }
    base.update(overrides)
    return base


def test_derive_slot_and_s3_default_variant():
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")
    slot_id, pt_key, prompts_key = _derive_slot_and_s3(
        cfg, _session(), Path("/tmp/x.replay.json.gz")
    )
    assert slot_id == "Qwen/Qwen3.5-9B|suggestion-elaborate|sub-01|bait_vgfmri4"
    expected_base = (
        "extract_model_features_to_py/multi-turn"
        "/model-Qwen_Qwen3.5-9B/all/sub-01/bait_vgfmri4"
    )
    assert pt_key == f"{expected_base}.pt"
    assert prompts_key == f"{expected_base}_prompts.jsonl.gz"


def test_derive_slot_and_s3_compressed_variant():
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B", action_compression=True)
    _, pt_key, prompts_key = _derive_slot_and_s3(
        cfg, _session(), Path("/tmp/x.replay.json.gz")
    )
    assert "/compressed/" in pt_key
    assert "/all/" not in pt_key
    assert "/compressed/" in prompts_key


def test_derive_slot_and_s3_subject_from_meta_fallback():
    """When session lacks top-level ``subject`` it should fall back to meta."""
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")
    session = {
        "game": "bait_vgfmri4",
        "meta": {
            "subject": "sub-99",
            "suggestion_level": "minimal",
            "rationale_mode": "action-only",
        },
    }
    slot_id, pt_key, _ = _derive_slot_and_s3(
        cfg, session, Path("/tmp/x.replay.json.gz")
    )
    assert "sub-99" in slot_id
    assert "/sub-99/" in pt_key
    assert "suggestion-minimal" in slot_id


def test_derive_slot_and_s3_missing_subject_raises():
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")
    session = {"game": "bait_vgfmri4", "meta": {"suggestion_level": "elaborate"}}
    with pytest.raises(ValueError, match="missing subject"):
        _derive_slot_and_s3(cfg, session, Path("/tmp/x.replay.json.gz"))


def test_derive_slot_and_s3_missing_suggestion_level_raises():
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")
    session = {"game": "bait_vgfmri4", "subject": "sub-01", "meta": {}}
    with pytest.raises(ValueError, match="missing meta.suggestion_level"):
        _derive_slot_and_s3(cfg, session, Path("/tmp/x.replay.json.gz"))


def test_derive_slot_and_s3_custom_prefix():
    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")
    cfg.exp_db.feature_s3_prefix = "alt/cohort/"
    _, pt_key, _ = _derive_slot_and_s3(cfg, _session(), Path("/tmp/x.replay.json.gz"))
    assert pt_key.startswith("alt/cohort/model-Qwen_Qwen3.5-9B/")
    # rstrip("/") on the prefix should leave a single slash before model-
    assert "//" not in pt_key


def test_derive_slot_id_format_matches_populate():
    """Slot id must use the same composition as ``populate_feature_extraction``.

    populate.py:237 builds prompt_config = f"suggestion-{suggestion}".  If
    that string ever drifts, claim_slot would silently never find the
    pre-populated slot.  This test pins the format.
    """
    cfg = ExtractionConfig(model="A/B")
    slot_id, _, _ = _derive_slot_and_s3(cfg, _session(), Path("/tmp/x.replay.json.gz"))
    parts = slot_id.split("|")
    assert len(parts) == 4
    assert parts[0] == "A/B"
    assert parts[1].startswith("suggestion-")
    assert parts[2] == "sub-01"
    assert parts[3] == "bait_vgfmri4"


# -- Mock-ExpDB integration smoke ---------------------------------------


def test_mock_expdb_skips_when_not_claimable(tmp_path, monkeypatch):
    """If ``claim_slot`` returns False, ``extract_for_session`` returns None
    immediately, doing no extraction and no upload."""
    import gzip
    import json

    from src.llm_eval.human_replay import extract_features as ef

    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")

    # Build a tiny session payload so ``_load_session`` succeeds.
    payload = {
        "game": "bait_vgfmri4",
        "subject": "sub-01",
        "system_prompt": "sys",
        "meta": {"suggestion_level": "elaborate", "rationale_mode": "action-only"},
        "steps": [
            {
                "action": "left",
                "level": 0,
                "attempt": 0,
                "step": 0,
                "response": {"action": "left"},
            },
        ],
    }
    prompts_file = tmp_path / "session.replay.json.gz"
    with gzip.open(prompts_file, "wt") as f:
        json.dump(payload, f)

    fake_client = MagicMock()
    fake_client.claim_slot.return_value = False

    # IS_MAIN must be true for the rank-0-only ExpDB path; it already is in
    # a non-distributed test process, but assert it for clarity.
    assert ef.IS_MAIN

    fake_wandb = MagicMock(id="run-id-x", url="https://wandb/run/x")

    result = ef.extract_for_session(
        prompts_file,
        cfg,
        model=MagicMock(),
        tokenizer=MagicMock(),
        hook=MagicMock(),
        wandb_run=fake_wandb,
        session_idx=0,
        exp_db_client=fake_client,
        exp_db_worker="test-host",
    )
    assert result is None
    fake_client.claim_slot.assert_called_once()
    # No upload, no complete, no fail when slot was not claimable.
    fake_client._s3.upload_file.assert_not_called()
    fake_client.complete_slot.assert_not_called()
    fake_client.fail_slot.assert_not_called()


def test_mock_expdb_fail_slot_on_extract_error(tmp_path, monkeypatch):
    """When ``claim_slot`` succeeds but extraction crashes, ``fail_slot``
    is called with a phase-tagged error message and the exception
    re-raises out of ``extract_for_session``."""
    import gzip
    import json

    from src.llm_eval.human_replay import extract_features as ef

    cfg = ExtractionConfig(model="Qwen/Qwen3.5-9B")

    # Valid session payload: claim block + initial parsing succeed.
    payload = {
        "game": "bait_vgfmri4",
        "subject": "sub-01",
        "system_prompt": "sys",
        "meta": {"suggestion_level": "elaborate", "rationale_mode": "action-only"},
        "steps": [{"action": "left"}],
    }
    prompts_file = tmp_path / "session.replay.json.gz"
    with gzip.open(prompts_file, "wt") as f:
        json.dump(payload, f)

    # Force the post-claim work to crash inside the wrapped try block.
    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic extract failure")

    monkeypatch.setattr(ef, "_build_conversation", boom)

    fake_client = MagicMock()
    fake_client.claim_slot.return_value = True
    fake_wandb = MagicMock(id="rid", url="https://wandb/run/rid")

    with pytest.raises(RuntimeError, match="synthetic extract failure"):
        ef.extract_for_session(
            prompts_file,
            cfg,
            model=MagicMock(),
            tokenizer=MagicMock(),
            hook=MagicMock(),
            wandb_run=fake_wandb,
            session_idx=0,
            exp_db_client=fake_client,
            exp_db_worker="test-host",
        )

    fake_client.claim_slot.assert_called_once()
    fake_client.fail_slot.assert_called_once()
    _, kwargs = fake_client.fail_slot.call_args
    assert kwargs["error_msg"].startswith("[extract] ")
    assert "synthetic extract failure" in kwargs["error_msg"]
    fake_client.complete_slot.assert_not_called()
    fake_client._s3.upload_file.assert_not_called()
