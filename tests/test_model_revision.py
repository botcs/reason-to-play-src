"""Tokenizer and weights must follow the configuration's resolved HF snapshot."""

import sys

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.llm_eval.human_replay import extract_features as extraction
from src.llm_eval.shared.config import ExtractionConfig


def test_runtime_pins_all_components_to_one_resolved_snapshot(monkeypatch):
    resolved = "a" * 40
    config = SimpleNamespace(
        _commit_hash=resolved, model_type="test", num_hidden_layers=2, hidden_size=4
    )
    model = MagicMock()
    model.config = config
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
    auto_config = MagicMock()
    auto_config.from_pretrained.return_value = config
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=auto_config,
            AutoModelForCausalLM=auto_model,
            AutoTokenizer=auto_tokenizer,
        ),
    )
    monkeypatch.setattr(extraction, "IS_DISTRIBUTED", False)
    monkeypatch.setattr(extraction, "_WindowHookExtractor", MagicMock())
    monkeypatch.setattr(extraction.torch.cuda, "is_available", lambda: False)

    actual, _, _ = extraction._load_runtime(
        ExtractionConfig(
            model="test/model",
            model_revision="release-tag",
            no_compile=True,
        )
    )
    assert actual is model
    assert auto_config.from_pretrained.call_args.kwargs["revision"] == "release-tag"
    assert auto_tokenizer.from_pretrained.call_args.kwargs["revision"] == resolved
    assert auto_model.from_pretrained.call_args.kwargs["revision"] == resolved
    assert auto_model.from_pretrained.call_args.kwargs["config"] is config
    assert tokenizer.pad_token == tokenizer.eos_token


def test_pinned_revision_cannot_reuse_legacy_experiment_slots():
    cfg = ExtractionConfig(model="test/model", model_revision="a" * 40)
    cfg.exp_db.enabled = True
    with pytest.raises(ValueError, match="slot IDs do not include the model revision"):
        extraction._validate_revision_tracking(cfg)
    cfg.exp_db.enabled = False
    extraction._validate_revision_tracking(cfg)
