# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Recovered vLLM batch wrapper for the legacy observation-generation utility.

Source: development llm-vgdl 0cfe0aff2ebc4efe0807cd2287a3dc1aff7166cb,
src/llm_eval/llm_wrapper.py, Git blob 2a752657c287e323bb5b0c93f5eb497e2e73977a.
Source-file SHA-256: f8ec5fa1f973613363f52445e23ed7c34b2a0dff30867d93e90b076953b5b030.

Adaptations: separate optional module, current context_length interface,
clear missing-dependency error, no single-call Weave telemetry, and removal
of an incorrect FP8 log message. Batch tokenization, loader arguments,
sampling settings and output fields preserve the recovered implementation.
Torch and vLLM are imported only when constructing or using the wrapper.
"""

from src.llm_eval.shared.llm_wrapper import LLAMA_31_CHAT_TEMPLATE, LLMWrapperBase


class VLLMWrapper(LLMWrapperBase):
    """Minimal vLLM wrapper for game scaffolding.

    Features:
    - Dynamic context length detection from model config
    - Memory reporting on load
    - Context usage logging per generation
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        n_gpus: int | None = None,
        max_num_seqs: int = 256,
        seed: int = 0,
    ):
        """
        Initialize LLM wrapper.

        Args:
            model_path: Path to the model
            max_model_len: Max context length in tokens
            n_gpus: Number of GPUs for tensor parallelism (auto-detect if None)
            max_num_seqs: Max concurrent sequences for batching (default: 256)
            seed: Random seed for deterministic sampling
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The legacy observation generator needs vLLM in a separate GPU "
                "environment; see docs/legacy-observation-generation.md."
            ) from exc
        import os
        import torch

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_path = model_path
        self.max_model_len = max_model_len

        # Auto-detect number of GPUs if not specified
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No CUDA GPUs available")
        self.n_gpus = n_gpus

        print("\n=== Loading LLM ===")
        print(f"Model: {model_path}")
        print(f"GPUs: {n_gpus}")
        print(f"Max context length: {self.max_model_len:,} tokens")
        print(f"Max concurrent seqs: {max_num_seqs}")

        # Preserve the historical loader settings (no forced quantization).
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=n_gpus,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.95,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )

        # Get tokenizer from vLLM (handles new tokenizer formats)
        self.tokenizer = self.model.get_tokenizer()

        # Set Llama 3.1 chat template if tokenizer doesn't have one
        if not self.tokenizer.chat_template:
            print("Warning: No chat template found, using Llama 3.1 template")
            self.tokenizer.chat_template = LLAMA_31_CHAT_TEMPLATE

        # Sampling parameters - allow model to use remaining context for generation
        # vLLM automatically clips to (max_model_len - input_tokens)
        self.sampling_params = SamplingParams(
            max_tokens=self.max_model_len,
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            seed=seed,
        )

        # Print memory report
        self._print_memory_report(torch)

    def _print_memory_report(self, torch) -> None:
        """Print GPU memory usage after model load."""
        print("\n=== Memory Report ===")
        device_count = torch.cuda.device_count()

        total_allocated = 0.0
        total_memory = 0.0

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            total = props.total_memory / (1024**3)
            total_allocated += allocated
            total_memory += total

        available = total_memory - total_allocated
        print(f"Model weights: {total_allocated:.2f} GB")
        print(f"Available for KV cache: {available:.2f} GB")
        print(f"Max context capacity: {self.max_model_len:,} tokens")
        print("=" * 30 + "\n")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate response from chat messages.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Tuple of (generated_text, context_info) where context_info has:
            - input_tokens: number of input tokens
            - max_tokens: max context length
            - pct_capacity: percentage of context used
        """
        from vllm.inputs import TokensPrompt

        # Apply chat template with tokenize=True (required for Mistral tokenizers)
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Count input tokens
        input_tokens = len(prompt_token_ids)
        pct_capacity = (input_tokens / self.max_model_len) * 100

        print(
            f"[LLM] Context: {input_tokens:,} / {self.max_model_len:,} tokens ({pct_capacity:.1f}%)"
        )

        context_info = {
            "input_tokens": input_tokens,
            "max_tokens": self.max_model_len,
            "pct_capacity": pct_capacity,
        }

        # Generate using token IDs
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text

        return generated_text, context_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        return_prompt_text: bool = False,
    ) -> list[tuple[str, dict]]:
        """
        Generate responses for a batch of message sequences.

        vLLM handles batching efficiently with PagedAttention and continuous batching.
        This method is the core of the hybrid pipeline's Phase 1.

        Args:
            messages_batch: List of message sequences, each is list of {'role': str, 'content': str}
            return_prompt_text: If True, include 'prompt_text' and 'prompt_token_len' in context_info

        Returns:
            List of (generated_text, context_info) tuples
        """
        from vllm.inputs import TokensPrompt

        # Prepare all prompts
        all_prompts = []
        all_token_ids = []
        all_prompt_texts = []

        for messages in messages_batch:
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            all_prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
            all_token_ids.append(prompt_token_ids)
            all_prompt_texts.append(prompt_text)

        # vLLM batch generation
        outputs = self.model.generate(all_prompts, self.sampling_params, use_tqdm=False)

        # Process results
        results = []
        for i, output in enumerate(outputs):
            input_tokens = len(all_token_ids[i])
            generated_text = output.outputs[0].text
            output_tokens = len(output.outputs[0].token_ids)

            pct_capacity = (input_tokens / self.max_model_len) * 100

            context_info = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "max_tokens": self.max_model_len,
                "pct_capacity": pct_capacity,
            }

            if return_prompt_text:
                context_info["prompt_text"] = all_prompt_texts[i]
                context_info["prompt_token_len"] = input_tokens

            results.append((generated_text, context_info))

        return results

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False
    ) -> str | list[int]:
        """
        Apply chat template to messages.

        Useful for getting prompt text to concatenate with response for feature extraction.

        Args:
            messages: List of {'role': str, 'content': str} dicts
            tokenize: If True, return token IDs instead of text

        Returns:
            Formatted prompt text or token IDs
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True,
        )

    @property
    def context_length(self) -> int:
        """Expose the current shared wrapper interface without changing sampling."""
        return self.max_model_len
