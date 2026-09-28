"""Original human identities survive prompt loading and actual CPU extraction."""

import gzip
import json

import pytest

from agents.lrm.prompting.conversation import load_prompts


SOURCE = {
    "source_play_id": "60735733f24111b59fd666f7",
    "source_recording": "sub-13/helper_vgfmri4/elaborate.human.replay.json.gz",
    "source_frame_index": 17,
    "source_document_index": 0,
}


def session(source="human"):
    return {
        "game": "helper_vgfmri4",
        "source": source,
        "subject": "sub-13",
        "system_prompt": "Choose an action.",
        "prompt_name": "action-only_elaborate",
        "meta": {
            "subject": "sub-13",
            "rationale_mode": "action-only",
            "suggestion_level": "elaborate",
            "num_trials": 1,
            "pipeline": "unified",
            "completed": True,
        },
        "steps": [
            {
                **SOURCE,
                "step": 0,
                "level": 0,
                "attempt": 0,
                "action": "left",
                "response": {"action": "left"},
                "formatted_obs": "Avatar at 1,1",
                "reward": 0,
                "won": False,
                "lose": False,
                "timeout": False,
                "realworld_ts": 1618171684.821119,
                "frame_idx": 17,
                "trial_idx": 0,
                "play_idx": 0,
                "play_id": "sub-13_1_0",
                "run": 1,
            }
        ],
    }


def save(tmp_path, data):
    path = tmp_path / "source.replay.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(data, stream)
    return path


@pytest.mark.parametrize("kind", ["human", "imputed", "narration"])
def test_prompt_identity_roundtrip_preserves_messages(tmp_path, kind):
    data = session(kind)
    linked, _ = load_prompts(save(tmp_path, data))
    assert {key: linked[0][key] for key in SOURCE} == SOURCE
    for key in SOURCE:
        data["steps"][0].pop(key)
    historical, _ = load_prompts(save(tmp_path, data))
    assert linked[0]["messages"] == historical[0]["messages"]
    assert all(historical[0][key] is None for key in SOURCE)
    assert linked[0]["realworld_ts"] == historical[0]["realworld_ts"]


def test_generative_and_synthetic_rows_do_not_claim_human_identity(tmp_path):
    data = session("generative")
    records, _ = load_prompts(save(tmp_path, data))
    assert all(records[0][key] is None for key in SOURCE)
    data = session()
    data["steps"][0]["action"] = "_level_advance"
    records, _ = load_prompts(save(tmp_path, data))
    assert all(records[0][key] is None for key in SOURCE)


def test_source_identity_in_saved_cpu_features_and_prompt_log(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from agents.lrm.features import extract as extraction
    from agents.lrm.config import ExtractionConfig

    vocab = {token: index for index, token in enumerate(["<unk>", "<pad>", "left"])}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
    )
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
        )
    ).eval()
    hook = extraction._WindowHookExtractor(model, 1, 16, "cpu")
    hook.register_hooks()
    cfg = ExtractionConfig(
        model="local/source-link-fixture", output_dir=str(tmp_path / "features")
    )
    try:
        extraction.extract_for_session(
            save(tmp_path, session()), cfg, model, tokenizer, hook
        )
    finally:
        hook.remove_hooks()
    tensor_file = next((tmp_path / "features").rglob("*.pt"))
    saved = torch.load(tensor_file, map_location="cpu", weights_only=False)
    assert saved["features"].shape == (1, 1, 16)
    assert {key: saved["metadata"][0][key] for key in SOURCE} == SOURCE
    assert saved["metadata"][0]["realworld_ts"] == 1618171684.821119
    prompt_file = next((tmp_path / "features").rglob("*_prompts.jsonl.gz"))
    with gzip.open(prompt_file, "rt") as stream:
        prompt = json.loads(stream.readline())
    assert prompt == session()["steps"][0]
