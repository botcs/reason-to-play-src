"""Verify observation generation batches without requiring vLLM or a GPU."""

import json
import subprocess
import sys
from types import SimpleNamespace

from src.llm_eval.human_replay.generate_observations import generate_observations


def test_wrapper_import_and_cli_help_do_not_import_gpu_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.llm_eval.shared.vllm_wrapper import VLLMWrapper; "
            "assert not VLLMWrapper.__abstractmethods__; "
            "assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.llm_eval.human_replay.generate_observations",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--prompts" in result.stdout


def test_generation_preserves_prompt_identity_across_file_batches(
    monkeypatch, tmp_path
):
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
    paths = []
    for subject in ("sub-01", "sub-02"):
        path = tmp_path / f"{subject}_vgfmri3_bait_first_person_all.jsonl"
        records = [
            {
                "messages": [{"role": "user", "content": f"{subject} step {step}"}],
                "game_name": "vgfmri3_bait",
                "level_id": 0,
                "trial_idx": 0,
                "step_num": step,
            }
            for step in range(3)
        ]
        path.write_text("\n".join(json.dumps(row) for row in records))
        path.with_suffix(".meta.json").write_text(
            json.dumps({"subject": subject, "game_name": "vgfmri3_bait"})
        )
        paths.append(path)

    outputs = generate_observations(
        paths,
        model="test/model",
        batch_size=4,
        max_model_len=1024,
        n_gpus=1,
        frame_stride=1,
        output_dir=str(tmp_path / "outputs"),
        wandb_project=None,
        verbose=False,
    )
    assert batches == [4, 2]
    assert sampling["top_k"] == 50
    assert sampling["top_p"] == 0.95
    for subject, output in zip(("sub-01", "sub-02"), outputs, strict=True):
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row["prompt_idx"] for row in rows] == [0, 1, 2]
        assert [row["response"] for row in rows] == [
            f"observed {subject} step {step}" for step in range(3)
        ]
        assert all(row["output_tokens"] == 2 for row in rows)
        assert all(row["prompt_token_len"] == len(row["prompt_text"]) for row in rows)
        metadata = json.loads(output.with_suffix(".meta.json").read_text())
        assert metadata["subject"] == subject
        assert metadata["total_output_tokens"] == 6
