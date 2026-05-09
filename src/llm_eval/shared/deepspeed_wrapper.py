# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
DeepSpeed-Inference wrapper for LLM feature extraction with tensor parallelism.

Provides LLM hidden state extraction using DeepSpeed-Inference for models
larger than a single GPU. Uses tensor parallelism for efficient memory
distribution across GPUs.

Key features:
- DeepSpeed-Inference tensor parallelism for large models
- Hidden state extraction with replace_with_kernel_inject=False
- Compatible interface with TransformersFeatureWrapper

Usage:
    # Initialize with tensor parallelism
    wrapper = DeepSpeedFeatureWrapper(
        model_path='meta-llama/Llama-3.1-70B-Instruct',
        tensor_parallel_size=4,
    )

    # Extract features (matches TransformersFeatureWrapper interface)
    results = wrapper.extract_features_batch(full_texts, prompt_token_lens)
"""

import os
from contextlib import nullcontext


from src.llm_eval.shared.llm_wrapper import LLMWrapperBase, LLAMA_31_CHAT_TEMPLATE


# Try to import NVTX for profiling markers (optional)
try:
    import torch.cuda.nvtx as nvtx

    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

    # Provide a dummy context manager if NVTX is not available
    class _DummyNVTX:
        @staticmethod
        def range(name):
            return nullcontext()

    nvtx = _DummyNVTX()


class DeepSpeedFeatureWrapper(LLMWrapperBase):
    """DeepSpeed-Inference wrapper with hidden state extraction.

    Features:
    - Tensor parallelism for large models (>1 GPU)
    - Extracts all hidden layer representations
    - Uses last token pooling for sequence dimension
    - Compatible with TransformersFeatureWrapper interface
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int | None = None,
        torch_dtype: str = "float16",
        max_tokens: int | None = None,
        verbose: bool = False,
        batch_size: int = 1,
        max_prompt_tokens: int | None = None,
        random_init: bool = False,
        extract_sublayers: bool = False,
    ):
        """
        Initialize DeepSpeed-Inference wrapper.

        Args:
            model_path: HuggingFace model path or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism (auto-detect if None)
            torch_dtype: Data type ('float16', 'bfloat16')
            max_tokens: Maximum sequence length (None = use model default)
            verbose: Whether to print token counts for each inference
            max_prompt_tokens: If set, truncate prompts to this many tokens
            random_init: If True, initialize model with random weights (architecture only, no checkpoint).
                Used as an ablation control to measure how much signal comes from the architecture alone.
            extract_sublayers: Whether to also extract pre-residual attention/MLP outputs
        """
        import torch
        import deepspeed
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.verbose = verbose
        self.batch_size = batch_size
        self.max_prompt_tokens = max_prompt_tokens
        self.extract_sublayers = extract_sublayers

        # Auto-detect tensor parallel size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
            if tensor_parallel_size == 0:
                raise RuntimeError("No CUDA GPUs available for DeepSpeed-Inference")
        self.tensor_parallel_size = tensor_parallel_size

        # Parse torch dtype
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported torch_dtype for DeepSpeed: {torch_dtype}")

        # Initialize distributed if not already done
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )

        if self.rank == 0:
            print("\n=== Loading DeepSpeed-Inference Model ===")
            print(f"Model: {model_path}")
            print(f"Tensor parallel size: {tensor_parallel_size}")
            print(f"Dtype: {self.torch_dtype}")
            print(f"World size: {self.world_size}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set chat template if tokenizer doesn't have one (e.g. base models)
        if not self.tokenizer.chat_template:
            if "deepseek" in model_path.lower():
                from src.llm_eval.shared.transformers_wrapper import (
                    DEEPSEEK_V3_CHAT_TEMPLATE,
                )

                print(
                    "Warning: No chat template found for DeepSeek, using DeepSeek V3 template"
                )
                self.tokenizer.chat_template = DEEPSEEK_V3_CHAT_TEMPLATE
            else:
                print("Warning: No chat template found, using Llama 3.1 template")
                self.tokenizer.chat_template = LLAMA_31_CHAT_TEMPLATE

        # Load base model (on CPU first, DeepSpeed will distribute)
        if self.rank == 0:
            print("Loading base model...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        # Get model info before wrapping
        self.model_type = type(self.base_model).__name__
        self.num_layers = self.base_model.config.num_hidden_layers
        self.hidden_dim = self.base_model.config.hidden_size

        # Wrap with DeepSpeed-Inference
        # replace_with_kernel_inject=False preserves hidden state access
        if self.rank == 0:
            print("Initializing DeepSpeed-Inference...")

        self.model = deepspeed.init_inference(
            self.base_model,
            mp_size=tensor_parallel_size,
            dtype=self.torch_dtype,
            replace_with_kernel_inject=False,  # Critical: preserve hidden states
            enable_cuda_graph=False,  # Disable for hidden state access
        )

        # Initialize hook-based feature extractor for sublayer extraction
        from src.llm_eval.shared.transformers_wrapper import HookBasedFeatureExtractor

        base_model = self._get_base_model()
        self.hook_extractor = HookBasedFeatureExtractor(
            model=base_model,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            primary_device=f"cuda:{self.local_rank}",
            extract_sublayers=extract_sublayers,
        )
        self.hook_extractor.register_hooks()
        num_hooks = self.num_layers + (self.num_layers * 2 if extract_sublayers else 0)

        if self.rank == 0:
            print(f"Model type: {self.model_type}")
            print(f"Hidden layers: {self.num_layers}")
            print(f"Hidden dim: {self.hidden_dim}")
            if self.max_prompt_tokens:
                print(f"Max prompt tokens: {self.max_prompt_tokens}")
            if extract_sublayers:
                print("Extract sublayers: Yes (attn + mlp pre-residual)")
            print(f"Hook-based feature extractor initialized ({num_hooks} hooks)")
            print("=" * 30 + "\n")

    def _get_base_model(self):
        """Get the underlying model from DeepSpeed wrapper."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate response from chat messages.

        Note: This wrapper is optimized for feature extraction, not generation.
        For efficient generation, use VLLMWrapper instead.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Tuple of (generated_text, context_info)
        """
        import torch

        # Apply chat template
        with nvtx.range("tokenization"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tok_kwargs = dict(return_tensors="pt", padding=False, truncation=True)
            if self.max_prompt_tokens:
                tok_kwargs["max_length"] = self.max_prompt_tokens
            inputs = self.tokenizer(prompt, **tok_kwargs)
            inputs = {k: v.to(f"cuda:{self.local_rank}") for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]

        base_model = self._get_base_model()

        with torch.inference_mode():
            with nvtx.range("model.forward"):
                outputs = base_model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            with nvtx.range("hidden_state_extraction"):
                hidden_states_tuple = outputs.hidden_states
                stacked = torch.stack(
                    [layer[0, -1, :] for layer in hidden_states_tuple], dim=0
                )

            with nvtx.range("D2H_transfer"):
                hidden_states = stacked.to(torch.bfloat16).cpu()

        context_info = {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "model_type": self.model_type,
            "hidden_states": hidden_states,
        }

        if self.verbose and self.rank == 0:
            print(f"[DeepSpeed] Input: {input_tokens} tokens")

        return "", context_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    @property
    def context_length(self) -> int:
        return int(
            getattr(self.model.config, "max_position_embeddings", None)
            or self.tokenizer.model_max_length
        )

    def extract_features_at_positions(
        self,
        full_text: str,
        prompt_token_len: int,
    ) -> dict:
        """
        Extract hidden states at two positions in a single forward pass.

        Args:
            full_text: Concatenated prompt + response text
            prompt_token_len: Number of tokens in the prompt

        Returns:
            Dict with hidden_states_pre, hidden_states_post, input_tokens
        """
        import torch

        with nvtx.range("extract_features_tokenization"):
            tok_kwargs = dict(return_tensors="pt", padding=False, truncation=True)
            if self.max_prompt_tokens:
                tok_kwargs["max_length"] = self.max_prompt_tokens
            inputs = self.tokenizer(full_text, **tok_kwargs)
            inputs = {k: v.to(f"cuda:{self.local_rank}") for k, v in inputs.items()}

        total_tokens = inputs["input_ids"].shape[1]

        # Clamp prompt_token_len to valid range
        pre_position = min(prompt_token_len - 1, total_tokens - 1)
        pre_position = max(0, pre_position)

        base_model = self._get_base_model()

        with torch.inference_mode():
            with nvtx.range("extract_features_forward"):
                outputs = base_model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            with nvtx.range("extract_features_stack"):
                all_hidden = torch.stack(outputs.hidden_states, dim=0)
                hidden_pre = all_hidden[:, 0, pre_position, :]
                hidden_post = all_hidden[:, 0, -1, :]

            with nvtx.range("extract_features_D2H"):
                result = {
                    "hidden_states_pre": hidden_pre.to(torch.bfloat16).cpu(),
                    "hidden_states_post": hidden_post.to(torch.bfloat16).cpu(),
                    "input_tokens": total_tokens,
                }

        if self.verbose and self.rank == 0:
            print(f"[DeepSpeed] Extract features: {total_tokens} tokens")

        return result

    def extract_features_batch(
        self,
        full_texts: list[str],
        prompt_token_lens: list[int],
        extract_post: bool = True,
    ) -> list[dict]:
        """
        Batch version of extract_features_at_positions.

        Args:
            full_texts: List of concatenated prompt + response texts
            prompt_token_lens: List of prompt token lengths for each item
            extract_post: Whether to extract hidden states at the post (last token)
                position. Set to False when no text is generated to avoid
                duplicate extraction at the same position as pre.

        Returns:
            List of dicts, each with hidden_states_pre, hidden_states_post, input_tokens
        """
        import torch

        batch_size = len(full_texts)
        if batch_size == 0:
            return []

        with nvtx.range("batch_extract_tokenization"):
            self.tokenizer.padding_side = "right"
            tok_kwargs = dict(return_tensors="pt", padding=True, truncation=True)
            if self.max_prompt_tokens:
                tok_kwargs["max_length"] = self.max_prompt_tokens
            inputs = self.tokenizer(full_texts, **tok_kwargs)
            inputs = {k: v.to(f"cuda:{self.local_rank}") for k, v in inputs.items()}

        # Get actual sequence lengths (without padding)
        seq_lens = inputs["attention_mask"].sum(dim=1).tolist()

        base_model = self._get_base_model()

        with torch.inference_mode():
            with nvtx.range("batch_extract_forward"):
                outputs = base_model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            with nvtx.range("batch_extract_stack"):
                # Stack all layers: (num_layers+1, batch, seq_len, hidden_dim)
                all_hidden = torch.stack(outputs.hidden_states, dim=0)

                batch_pre_hidden = []
                batch_post_hidden = [] if extract_post else None

                for i in range(batch_size):
                    actual_len = seq_lens[i]
                    prompt_len = prompt_token_lens[i]

                    # Clamp pre_position to valid range
                    pre_position = min(prompt_len - 1, actual_len - 1)
                    pre_position = max(0, pre_position)

                    hidden_pre = all_hidden[:, i, pre_position, :]
                    batch_pre_hidden.append(hidden_pre)

                    if extract_post:
                        # Post position is last real token (not padding)
                        post_position = actual_len - 1
                        hidden_post = all_hidden[:, i, post_position, :]
                        batch_post_hidden.append(hidden_post)

                # Stack for single D2H transfer
                all_pre = torch.stack(batch_pre_hidden, dim=0)
                all_post = (
                    torch.stack(batch_post_hidden, dim=0) if extract_post else None
                )

            with nvtx.range("batch_extract_D2H"):
                all_pre_t = all_pre.to(torch.bfloat16).cpu()
                all_post_t = all_post.to(torch.bfloat16).cpu() if extract_post else None

        results = []
        for i in range(batch_size):
            results.append(
                {
                    "hidden_states_pre": all_pre_t[i],
                    "hidden_states_post": all_post_t[i] if extract_post else None,
                    "input_tokens": seq_lens[i],
                }
            )

        if self.verbose and self.rank == 0:
            avg_len = sum(seq_lens) / batch_size
            print(
                f"[DeepSpeed] Batch extract: {batch_size} items, avg {avg_len:.0f} tokens"
            )

        return results

    def generate_batch(
        self, messages_batch: list[list[dict]]
    ) -> list[tuple[str, dict]] | None:
        """
        Extract features for a batch of message sequences using forward hooks.

        Compatible interface with DeepSeekTorchrunWrapper.generate_batch
        for use with extract_features.py.

        Args:
            messages_batch: List of message sequences (chat format)

        Returns:
            List of (empty_string, context_info) on rank 0, None on other ranks.
            context_info contains 'hidden_states' (num_layers+1, hidden_dim) tensor,
            and optionally 'sublayer_attn'/'sublayer_mlp' (num_layers, hidden_dim) tensors.
        """
        import torch

        batch_size = len(messages_batch)

        # Apply chat template and tokenize
        prompts = [
            self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
        self.tokenizer.padding_side = "right"
        tok_kwargs = dict(return_tensors="pt", padding=True, truncation=True)
        if self.max_prompt_tokens:
            tok_kwargs["max_length"] = self.max_prompt_tokens
        inputs = self.tokenizer(prompts, **tok_kwargs)
        inputs = {k: v.to(f"cuda:{self.local_rank}") for k, v in inputs.items()}

        # Get actual sequence lengths (without padding)
        seq_lens = inputs["attention_mask"].sum(dim=1).tolist()

        # Last real token per item (right-padded tokenizer above).
        positions = [int(s) - 1 for s in seq_lens]

        self.hook_extractor.prepare_extraction(batch_size, positions)

        base_model = self._get_base_model()

        with torch.inference_mode():
            # Forward pass -- hooks capture activations at target positions
            # No output_hidden_states needed (hooks are more memory-efficient)
            _ = base_model(
                **inputs,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )

            # Get results from hooks
            hook_results = self.hook_extractor.get_results()

        # Only rank 0 returns results
        if self.rank == 0:
            if self.extract_sublayers:
                if not isinstance(hook_results, dict):
                    raise RuntimeError(
                        f"Expected dict from hook extractor with extract_sublayers=True, "
                        f"got {type(hook_results).__name__}"
                    )
                all_hidden = hook_results["hidden"]
                all_sublayer_attn = hook_results["sublayer_attn"]
                all_sublayer_mlp = hook_results["sublayer_mlp"]
            else:
                # hook_results is a (batch, num_layers+1, hidden_dim) tensor
                all_hidden = hook_results

            results = []
            for i in range(batch_size):
                context_info = {
                    "hidden_states": all_hidden[i],  # (num_layers+1, hidden_dim)
                    "input_tokens": int(seq_lens[i]),
                    "model_type": self.model_type,
                }
                if self.extract_sublayers:
                    context_info["sublayer_attn"] = all_sublayer_attn[
                        i
                    ]  # (num_layers, hidden_dim)
                    context_info["sublayer_mlp"] = all_sublayer_mlp[
                        i
                    ]  # (num_layers, hidden_dim)
                results.append(("", context_info))
            return results
        return None

    def cleanup(self):
        """Remove hooks and cleanup distributed process group."""
        if hasattr(self, "hook_extractor"):
            self.hook_extractor.remove_hooks()
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
