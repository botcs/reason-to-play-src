"""Verify the vLLM adapter without requiring vLLM or a GPU."""

import subprocess
import sys
from types import SimpleNamespace

from agents.lrm.backends.adapters.vllm import VLLMWrapper


def test_wrapper_import_does_not_import_gpu_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agents.lrm.backends.adapters.vllm import VLLMWrapper; "
            "assert not VLLMWrapper.__abstractmethods__; "
            "assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_batch_generation_preserves_message_order_and_sampling(monkeypatch):
    batches = []
    sampling = {}

    class Tokenizer:
        chat_template = "test template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert add_generation_prompt
            text = messages[-1]["content"]
            return list(text.encode()) if tokenize else text

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs["tensor_parallel_size"] == 1

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, prompts, params, use_tqdm):
            batches.append(len(prompts))
            assert params.temperature == 0.5
            assert params.seed == 0
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="observed "
                            + bytes(prompt["prompt_token_ids"]).decode(),
                            token_ids=[101, 102],
                        )
                    ]
                )
                for prompt in prompts
            ]

    def sampling_params(**kwargs):
        sampling.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=sampling_params),
    )
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0)),
    )
    wrapper = VLLMWrapper(model_path="test/model", max_model_len=1024, n_gpus=1)
    messages = [
        [{"role": "user", "content": f"sub-{subject:02d} step {step}"}]
        for subject in (1, 2)
        for step in range(3)
    ]
    outputs = []
    for start in range(0, len(messages), 4):
        outputs.extend(
            wrapper.generate_batch(messages[start : start + 4], return_prompt_text=True)
        )
    assert batches == [4, 2]
    assert sampling["top_k"] == 50
    assert sampling["top_p"] == 0.95
    for message, (response, context) in zip(messages, outputs, strict=True):
        text = message[0]["content"]
        assert response == f"observed {text}"
        assert context["output_tokens"] == 2
        assert context["prompt_text"] == text
        assert context["input_tokens"] == context["prompt_token_len"] == len(text)
