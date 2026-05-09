# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
HuggingFace Transformers wrapper for LLM feature extraction.

Provides LLM generation with hidden state extraction using HuggingFace Transformers.
This is needed because vLLM doesn't expose internal hidden states.

Key optimizations:
- Vectorized D2H memory transfers (single CUDA sync per inference instead of 32+)
- Dual hidden state extraction (pre and post generation) without double forward pass
- Optional verbose mode to reduce logging overhead in hot paths
- NVTX markers for fine-grained profiling with nsys/Nsight

Usage:
    # Feature extraction only (forward pass, no generation)
    wrapper = TransformersFeatureWrapper(
        model_path='meta-llama/Llama-3.2-1B-Instruct',
        extract_features=True,
        generate_text=False,
    )
    text, info = wrapper.generate(messages)
    hidden_states = info['hidden_states']  # shape: (num_layers+1, hidden_dim)

    # Both generation and feature extraction (dual hidden states)
    wrapper = TransformersFeatureWrapper(
        model_path='meta-llama/Llama-3.2-1B-Instruct',
        extract_features=True,
        generate_text=True,
    )
    text, info = wrapper.generate(messages)
    pre_hidden = info['hidden_states_pre']   # At end of prompt
    post_hidden = info['hidden_states_post']  # After generation
"""

import json
import numpy as np
from contextlib import nullcontext

from src.llm_eval.shared.llm_wrapper import LLMWrapperBase, LLAMA_31_CHAT_TEMPLATE


# DeepSeek V3 chat template (simplified for non-tool-call use)
# Reference: DeepSeek-V3 tokenizer assets
DEEPSEEK_V3_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% set ns = namespace(system_prompt='', is_first_sp=true, is_last_user=false) %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% if ns.is_first_sp %}{% set ns.system_prompt = message['content'] %}{% set ns.is_first_sp = false %}"
    "{% else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{{ bos_token }}{{ ns.system_prompt }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{% set ns.is_last_user = true %}{{ '<｜User｜>' + message['content'] }}{% endif %}"
    "{% if message['role'] == 'assistant' %}{% if ns.is_last_user %}{{ '<｜Assistant｜></think>' }}{% endif %}{% set ns.is_last_user = false %}{{ message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and ns.is_last_user %}{{ '<｜Assistant｜></think>' }}{% endif %}"
)


class HookBasedFeatureExtractor:
    """
    Extracts hidden states at a single token position per batch item using
    forward hooks.

    Instead of materializing full hidden states (layers x batch x seq_len x hidden),
    this only captures activations at the target position (layers x batch x hidden).

    Memory reduction: ~1000x for typical sequences.
    """

    def __init__(
        self,
        model,
        num_layers: int,
        hidden_dim: int,
        primary_device,
        extract_sublayers: bool = False,
    ):
        import torch

        self.torch = torch  # Store reference for use in methods
        self.model = model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.primary_device = (
            primary_device  # Device for final aggregation (usually cuda:0)
        )
        self.extract_sublayers = extract_sublayers
        self.hook_handles = []

        # Will be set before each forward pass
        self.target_positions = None  # (batch,) tensor of per-item target position
        self.layer_outputs = None  # List of (batch, hidden_dim) per layer

        # Sublayer outputs (pre-residual attention and MLP outputs)
        self.sublayer_outputs = None  # dict of {'attn': [...], 'mlp': [...]}

    def _gather_at_target(self, hidden):
        """Gather (batch, hidden) at self.target_positions on hidden's device."""
        device = hidden.device
        positions = self.target_positions.to(device)
        batch_indices = self.torch.arange(hidden.size(0), device=device)
        return hidden[batch_indices, positions]

    def _create_hook(self, layer_idx: int):
        """Create a hook that extracts activations at target position."""

        def hook(module, input, output):
            # Skip if not in extraction mode (e.g., during model init)
            if self.target_positions is None:
                return

            hidden = output[0] if isinstance(output, tuple) else output
            extracted = self._gather_at_target(hidden)  # (batch, hidden)
            self.layer_outputs[layer_idx] = extracted.to(
                dtype=self.torch.bfloat16, device=self.primary_device
            )

        return hook

    def _create_sublayer_hook(self, layer_idx: int, sublayer_type: str):
        """Create a hook that extracts sublayer (attn/mlp) activations at target position."""

        def hook(module, input, output):
            if self.target_positions is None:
                return

            # Attention returns (output, weights, cache) tuple; MLP returns plain tensor.
            # In both cases output[0] / output is the tensor that gets added back to
            # the residual stream -- exactly the pre-residual sublayer output we want.
            hidden = output[0] if isinstance(output, tuple) else output
            extracted = self._gather_at_target(hidden)  # (batch, hidden)
            self.sublayer_outputs[sublayer_type][layer_idx] = extracted.to(
                dtype=self.torch.bfloat16, device=self.primary_device
            )

        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer decoder layers.

        The embedding output is intentionally not hooked -- it's a pure table
        lookup with no contextualisation and every fMRI decoding paper in
        this line of work uses transformer-layer outputs only.
        """
        self.remove_hooks()

        base_model = self.model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        # Hook on each transformer block output (layers 0..num_layers-1)
        for i, layer in enumerate(base_model.layers):
            handle = layer.register_forward_hook(self._create_hook(i))
            self.hook_handles.append(handle)

            # Sublayer hooks: capture attention and MLP outputs before residual addition.
            # Standard decoder layers name the attention submodule `self_attn`; hybrid
            # models (e.g. Qwen3.5) tag linear-attention layers as `linear_attn` instead.
            # Both variants feed an additive residual, so hooking either one captures
            # the pre-residual attn-equivalent output.
            if self.extract_sublayers:
                if hasattr(layer, "self_attn"):
                    attn_mod = layer.self_attn
                elif hasattr(layer, "linear_attn"):
                    attn_mod = layer.linear_attn
                else:
                    raise AttributeError(
                        f"Layer {i} missing attention submodule. Expected "
                        f"'self_attn' or 'linear_attn'; found "
                        f"{[n for n, _ in layer.named_children()]}"
                    )
                if not hasattr(layer, "mlp"):
                    raise AttributeError(
                        f"Layer {i} missing 'mlp' submodule (found "
                        f"{[n for n, _ in layer.named_children()]})"
                    )
                handle = attn_mod.register_forward_hook(
                    self._create_sublayer_hook(i, "attn")
                )
                self.hook_handles.append(handle)
                handle = layer.mlp.register_forward_hook(
                    self._create_sublayer_hook(i, "mlp")
                )
                self.hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def prepare_extraction(self, batch_size: int, positions: list[int]):
        """
        Prepare for extraction by setting target positions and initializing storage.

        Args:
            batch_size: Number of items in batch
            positions: Per-item target token position (one per batch item).
                       Typically the last real (non-padding) token of the input.
        """
        # Create target positions tensor (will be moved to each layer's device in hook)
        self.target_positions = self.torch.tensor(
            positions, dtype=self.torch.long, device=self.primary_device
        )

        # Initialize per-layer storage list (hooks will populate)
        self.layer_outputs = [None] * self.num_layers

        # Initialize sublayer storage (no embedding layer entry)
        if self.extract_sublayers:
            self.sublayer_outputs = {
                "attn": [None] * self.num_layers,
                "mlp": [None] * self.num_layers,
            }

    def get_results(self):
        """
        Get extracted features after forward pass.

        Returns:
            When extract_sublayers=False:
                Tensor shape (batch, num_layers, hidden_dim) on CPU, bf16.
            When extract_sublayers=True:
                Dict with:
                  - "hidden":        (batch, num_layers, hidden_dim)
                  - "sublayer_attn": (batch, num_layers, hidden_dim)
                  - "sublayer_mlp":  (batch, num_layers, hidden_dim)
        """
        # Stack all layers: (num_layers, batch, hidden_dim) -> (batch, layers, hidden)
        stacked = self.torch.stack(self.layer_outputs, dim=0).permute(1, 0, 2)
        all_hidden_cpu = stacked.to("cpu", non_blocking=True)

        if not self.extract_sublayers:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            return all_hidden_cpu

        attn_stacked = self.torch.stack(self.sublayer_outputs["attn"], dim=0).permute(
            1, 0, 2
        )
        mlp_stacked = self.torch.stack(self.sublayer_outputs["mlp"], dim=0).permute(
            1, 0, 2
        )
        sublayer_attn = attn_stacked.to("cpu", non_blocking=True)
        sublayer_mlp = mlp_stacked.to("cpu", non_blocking=True)

        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

        return {
            "hidden": all_hidden_cpu,
            "sublayer_attn": sublayer_attn,
            "sublayer_mlp": sublayer_mlp,
        }


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


class TransformersFeatureWrapper(LLMWrapperBase):
    """HuggingFace Transformers wrapper with hidden state extraction.

    Features:
    - Extracts all hidden layer representations
    - Uses last token pooling for sequence dimension
    - Optional text generation
    - Supports auto device detection (cuda > mps > cpu)
    - FP8 quantization support for reduced memory on H100+ GPUs
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "auto",
        extract_features: bool = True,
        generate_text: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.95,
        verbose: bool = False,
        batch_size: int = 1,
        extract_sublayers: bool = False,
        max_prompt_tokens: int | None = None,
        compile_model: bool = True,
    ):
        """
        Initialize Transformers wrapper.

        Args:
            model_path: HuggingFace model path or local path
            torch_dtype: Data type ('auto', 'bfloat16', 'float16', 'float32')
            extract_features: Whether to extract hidden states
            generate_text: Whether to generate text response
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            verbose: Whether to print token counts for each inference
            batch_size: Batch size for feature extraction
            extract_sublayers: Whether to also extract pre-residual attention/MLP outputs
            max_prompt_tokens: If set, truncate prompts to this many tokens
        """
        import os
        import torch
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.extract_features = extract_features
        self.generate_text = generate_text
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.batch_size = batch_size
        self.extract_sublayers = extract_sublayers
        self.max_prompt_tokens = max_prompt_tokens

        # Torchrun detection: when WORLD_SIZE > 1 we let transformers
        # tensor-parallelize across ranks via `tp_plan='auto'` (the model must
        # declare `base_model_tp_plan` in its config; Qwen3.5 and most recent
        # HF decoder models do). Single-rank / no-torchrun launches still
        # fall back to `device_map='auto'` (pipeline parallelism via accelerate).
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_distributed = self.world_size > 1
        self.is_main = self.rank == 0

        if self.is_distributed:
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cuda"
        # Keep 'auto' to let model use native dtype from config (e.g., FP8 for DeepSeek V3.2)
        self.torch_dtype = (
            "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)
        )

        print("\n=== Loading Transformers Model ===")
        print(f"Model: {model_path}")
        print(f"Dtype: {self.torch_dtype}")
        print(f"Extract features: {extract_features}")
        print(f"Generate text: {generate_text}")
        print(f"GPUs: {torch.cuda.device_count()}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set chat template if tokenizer doesn't have one
        if not self.tokenizer.chat_template:
            if "deepseek" in model_path.lower():
                print(
                    "Warning: No chat template found for DeepSeek, using DeepSeek V3 template"
                )
                self.tokenizer.chat_template = DEEPSEEK_V3_CHAT_TEMPLATE
            else:
                print("Warning: No chat template found, using Llama 3.1 template")
                self.tokenizer.chat_template = LLAMA_31_CHAT_TEMPLATE

        # Detect best attention implementation
        try:
            import flash_attn

            attn_impl = "flash_attention_2"
            print(f"Using flash_attention_2 (flash_attn v{flash_attn.__version__})")
        except ImportError:
            attn_impl = "sdpa"
            print("flash_attn not available, using sdpa (scaled dot-product attention)")

        # Handle DeepSeek-V3.2 which uses model_type="deepseek_v32" not yet in transformers
        # Check config first to detect this case
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            use_deepseek_v3_fallback = False
        except ValueError as e:
            if "deepseek_v32" in str(e):
                use_deepseek_v3_fallback = True
                print("Detected DeepSeek-V3.2, using DeepseekV3ForCausalLM fallback")
            else:
                raise

        if use_deepseek_v3_fallback:
            # Load DeepSeek-V3.2 using the compatible DeepseekV3 architecture
            from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
            from huggingface_hub import hf_hub_download
            import json as json_module

            config_path = hf_hub_download(model_path, "config.json")
            with open(config_path) as f:
                config_dict = json_module.load(f)
            config_dict["model_type"] = "deepseek_v3"
            # Keep quantization_config for FP8 loading
            config = DeepseekV3Config(**config_dict)

            # Create explicit balanced device map (FP8 models can't use device_map='auto' with CPU)
            n_gpus = torch.cuda.device_count()
            n_layers = config.num_hidden_layers
            device_map = {
                "model.embed_tokens": 0,
                "model.norm": n_gpus - 1,
                "lm_head": n_gpus - 1,
            }
            layers_per_gpu = n_layers // n_gpus
            remainder = n_layers % n_gpus
            layer_idx = 0
            for gpu in range(n_gpus):
                n_layers_this_gpu = layers_per_gpu + (1 if gpu < remainder else 0)
                for _ in range(n_layers_this_gpu):
                    device_map[f"model.layers.{layer_idx}"] = gpu
                    layer_idx += 1

            self.model = DeepseekV3ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        else:
            load_kwargs = dict(
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            if self.is_distributed:
                # tp_plan="auto" requires the model config to declare
                # `base_model_tp_plan`; transformers raises a clear error if
                # the plan is missing rather than silently degrading.
                load_kwargs["tp_plan"] = "auto"
                if self.is_main:
                    print(
                        f"Loading with tp_plan='auto' across {self.world_size} rank(s); "
                        f"this rank is local_rank={self.local_rank}"
                    )
            else:
                load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if self.is_main:
            print("Model loaded successfully")

        # Print device distribution info
        if hasattr(self.model, "hf_device_map") and self.is_main:
            unique_devices = set(self.model.hf_device_map.values())
            print(f"Model distributed across devices: {unique_devices}")

        self.model.eval()

        # Get model info before compilation (config access may fail after compile)
        self.model_type = type(self.model).__name__
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

        # Initialize hook-based feature extractor BEFORE torch.compile
        # (hooks may not work correctly with compiled models).
        # Under tp_plan, each rank owns a cuda:local_rank device and layer
        # outputs land there post-allreduce; keep aggregation local to avoid
        # cross-rank D2D copies.
        if self.is_distributed:
            primary_device = f"cuda:{self.local_rank}"
        else:
            primary_device = "cuda:0" if self.device == "cuda" else self.device
        self.hook_extractor = HookBasedFeatureExtractor(
            model=self.model,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            primary_device=primary_device,
            extract_sublayers=extract_sublayers,
        )
        self.hook_extractor.register_hooks()
        num_hooks = self.num_layers + (self.num_layers * 2 if extract_sublayers else 0)
        if self.is_main:
            print(
                f"Hook-based feature extractor initialized ({num_hooks} hooks registered"
                f"{', with sublayer extraction' if extract_sublayers else ''})"
            )

        # Compile model for faster inference (PyTorch 2.0+).
        # Can be disabled via compile_model=False for dev iteration (the first
        # batch at a new seq-length shape otherwise pays a ~1-2 min compile
        # cost before any tokens leave the forward).
        self.compiled = False
        if compile_model:
            try:
                if self.is_main:
                    print("Compiling model with torch.compile (mode=default)...")
                self.model = torch.compile(self.model, mode="default")
                self.compiled = True
                if self.is_main:
                    print("Model compiled successfully")
            except Exception as e:
                if self.is_main:
                    print(f"torch.compile failed, using eager mode: {e}")
        elif self.is_main:
            print("Skipping torch.compile (compile_model=False)")

        if self.is_main:
            print(f"Model type: {self.model_type}")
            print(f"Hidden layers: {self.num_layers}")
            print(f"Hidden dim: {self.hidden_dim}")
            print(f"Compiled: {self.compiled}")
            print("=" * 30 + "\n")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate response and/or extract features from chat messages.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Tuple of (generated_text, context_info) where context_info has:
            - input_tokens: number of input tokens
            - output_tokens: number of output tokens (if generate_text=True)
            - model_type: model class name

            If extract_features=True and generate_text=True:
            - hidden_states_pre: numpy array (num_layers+1, hidden_dim) at end of prompt
            - hidden_states_post: numpy array (num_layers+1, hidden_dim) after generation

            If extract_features=True and generate_text=False:
            - hidden_states: numpy array (num_layers+1, hidden_dim)
        """
        import torch

        # Apply chat template and tokenize
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]

        context_info = {
            "input_tokens": input_tokens,
            "model_type": self.model_type,
        }

        generated_text = ""
        output_tokens = 0

        with torch.no_grad():
            if self.generate_text:
                # Generate with hidden states from all steps
                with nvtx.range("model.generate"):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_hidden_states=self.extract_features,
                        return_dict_in_generate=True,
                    )

                # Decode generated text (excluding input)
                with nvtx.range("decode"):
                    generated_ids = outputs.sequences[0][input_tokens:]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    output_tokens = len(generated_ids)

                # Extract dual hidden states from generation (no separate forward pass)
                if self.extract_features:
                    with nvtx.range("hidden_state_extraction"):
                        # outputs.hidden_states is a tuple over generation steps
                        # Each step has (embedding, layer_0, ..., layer_N)
                        # Step 0 = prefill (prompt processing)
                        # Step -1 = last decode step

                        # Pre-generation: hidden states at end of prompt (step 0, last prompt token)
                        pre_gen_hidden = outputs.hidden_states[
                            0
                        ]  # tuple of layer tensors

                        # Post-generation: hidden states after last generated token
                        post_gen_hidden = outputs.hidden_states[
                            -1
                        ]  # tuple of layer tensors

                        # Vectorized extraction: stack on GPU first, then single D2H transfer
                        pre_stacked = torch.stack(
                            [layer[0, -1, :] for layer in pre_gen_hidden], dim=0
                        )
                        post_stacked = torch.stack(
                            [layer[0, -1, :] for layer in post_gen_hidden], dim=0
                        )

                    # Single D2H transfer for each (keep bf16 for storage efficiency)
                    with nvtx.range("D2H_transfer"):
                        context_info["hidden_states_pre"] = pre_stacked.to(
                            torch.bfloat16
                        ).cpu()
                        context_info["hidden_states_post"] = post_stacked.to(
                            torch.bfloat16
                        ).cpu()

            else:
                # Forward pass only (no generation)
                with nvtx.range("model.forward"):
                    forward_outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                if self.extract_features:
                    with nvtx.range("hidden_state_extraction"):
                        hidden_states_tuple = forward_outputs.hidden_states
                        # Vectorized extraction: stack on GPU, then single D2H transfer
                        stacked = torch.stack(
                            [layer[0, -1, :] for layer in hidden_states_tuple], dim=0
                        )
                    with nvtx.range("D2H_transfer"):
                        context_info["hidden_states"] = stacked.to(torch.bfloat16).cpu()

        context_info["output_tokens"] = output_tokens

        if self.verbose:
            print(
                f"[Transformers] Input: {input_tokens} tokens, Output: {output_tokens} tokens"
            )

        return generated_text, context_info

    def get_hidden_states(self, messages: list[dict]) -> dict:
        """
        Standalone method for feature extraction only.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Dict with:
            - hidden_states: numpy array (num_layers+1, hidden_dim)
            - input_tokens: number of input tokens
        """
        # Temporarily enable feature extraction
        old_extract = self.extract_features
        old_generate = self.generate_text
        self.extract_features = True
        self.generate_text = False

        _, context_info = self.generate(messages)

        # Restore settings
        self.extract_features = old_extract
        self.generate_text = old_generate

        return {
            "hidden_states": context_info["hidden_states"],
            "input_tokens": context_info["input_tokens"],
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    @property
    def context_length(self) -> int:
        return int(
            getattr(self.model.config, "max_position_embeddings", None)
            or self.tokenizer.model_max_length
        )

    def extract_features_batch(
        self,
        full_texts: list[str],
    ) -> list[dict]:
        """
        Batch feature extraction using forward hooks (optimized).

        Uses hooks to extract hidden states only at the last real token per
        sequence, avoiding full hidden state materialization.

        Args:
            full_texts: List of prompt texts

        Returns:
            List of dicts, each with hidden_states, input_tokens
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        seq_lens = inputs["attention_mask"].sum(dim=1).tolist()
        positions = [int(sl) - 1 for sl in seq_lens]

        with nvtx.range("batch_extract_prepare_hooks"):
            self.hook_extractor.prepare_extraction(batch_size, positions)

        with torch.inference_mode():
            with nvtx.range("batch_extract_forward"):
                _ = self.model(
                    **inputs,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )

            with nvtx.range("batch_extract_D2H"):
                results_obj = self.hook_extractor.get_results()

        if isinstance(results_obj, dict):
            all_hidden = results_obj["hidden"]
            all_attn = results_obj["sublayer_attn"]
            all_mlp = results_obj["sublayer_mlp"]
        else:
            all_hidden = results_obj
            all_attn = None
            all_mlp = None

        results = []
        for i in range(batch_size):
            entry = {
                "hidden_states": all_hidden[i],
                "input_tokens": int(seq_lens[i]),
            }
            if all_attn is not None:
                entry["sublayer_attn"] = all_attn[i]
                entry["sublayer_mlp"] = all_mlp[i]
            results.append(entry)

        if self.verbose:
            avg_len = sum(seq_lens) / batch_size
            print(
                f"[Transformers] Batch extract: {batch_size} items, avg {avg_len:.0f} tokens"
            )

        return results

    def cleanup(self):
        """Remove hooks and free resources."""
        if hasattr(self, "hook_extractor"):
            self.hook_extractor.remove_hooks()

    def generate_batch(
        self, messages_batch: list[list[dict]]
    ) -> list[tuple[str, dict]]:
        """
        Process a batch of message sequences.

        Args:
            messages_batch: List of message sequences, each is list of {'role': str, 'content': str}

        Returns:
            List of (generated_text, context_info) tuples
        """
        import torch

        # Apply chat template to all sequences
        with nvtx.range("batch_tokenization"):
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages_batch
            ]

            # Tokenize with padding
            self.tokenizer.padding_side = "left"  # For generation, pad on left
            tok_kwargs = dict(return_tensors="pt", padding=True, truncation=True)
            if self.max_prompt_tokens:
                tok_kwargs["max_length"] = self.max_prompt_tokens
            inputs = self.tokenizer(prompts, **tok_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        batch_size = inputs["input_ids"].shape[0]
        seq_lens = inputs["attention_mask"].sum(dim=1).tolist()

        results = []

        with torch.no_grad():
            if self.generate_text:
                # Generate for the batch
                with nvtx.range("batch_model.generate"):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_hidden_states=self.extract_features,
                        return_dict_in_generate=True,
                    )

                # Decode each sequence
                with nvtx.range("batch_decode"):
                    generated_texts = []
                    for i in range(batch_size):
                        generated_ids = outputs.sequences[i][
                            inputs["input_ids"].shape[1] :
                        ]
                        text = self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        )
                        generated_texts.append(text)

                # Extract dual hidden states from generation (no separate forward pass)
                if self.extract_features:
                    with nvtx.range("batch_hidden_state_extraction"):
                        # outputs.hidden_states is a tuple over generation steps
                        # Step 0 = prefill, Step -1 = last decode step
                        pre_gen_hidden = outputs.hidden_states[
                            0
                        ]  # tuple of layer tensors
                        post_gen_hidden = outputs.hidden_states[
                            -1
                        ]  # tuple of layer tensors
                        padded_len = inputs["input_ids"].shape[1]

                        # Build batch tensors on GPU: (batch, num_layers+1, hidden)
                        batch_pre_hidden = []
                        batch_post_hidden = []
                        for i in range(batch_size):
                            # For pre-generation: last token position in padded input
                            actual_pos = padded_len - seq_lens[i] + seq_lens[i] - 1
                            pre_layers = torch.stack(
                                [layer[i, actual_pos, :] for layer in pre_gen_hidden],
                                dim=0,
                            )
                            # For post-generation: always last position (-1)
                            post_layers = torch.stack(
                                [layer[i, -1, :] for layer in post_gen_hidden], dim=0
                            )
                            batch_pre_hidden.append(pre_layers)
                            batch_post_hidden.append(post_layers)

                    # Stack all batches, single D2H transfer each (keep bf16)
                    with nvtx.range("batch_D2H_transfer"):
                        all_pre_hidden = (
                            torch.stack(batch_pre_hidden, dim=0)
                            .to(torch.bfloat16)
                            .cpu()
                        )
                        all_post_hidden = (
                            torch.stack(batch_post_hidden, dim=0)
                            .to(torch.bfloat16)
                            .cpu()
                        )

            else:
                # Forward pass only -- always go through the hook path so peak
                # memory is bounded. Setting output_hidden_states=True makes
                # HF retain every layer's full (batch, seq, hidden) tensor
                # until the forward returns; the hook path captures a single
                # (batch, hidden) slice per layer as each layer's forward
                # completes, then lets the full tensor be freed.
                generated_texts = [""] * batch_size
                all_sublayer_attn = None
                all_sublayer_mlp = None

                if self.extract_features:
                    # generate_batch tokenizes with left padding (for generation
                    # consistency), so the last real token is at padded_len - 1
                    # for every item in the batch.
                    padded_len = inputs["input_ids"].shape[1]
                    positions = [padded_len - 1] * batch_size

                    with nvtx.range("batch_extract_prepare_hooks"):
                        self.hook_extractor.prepare_extraction(batch_size, positions)

                    with nvtx.range("batch_model.forward"):
                        _ = self.model(
                            **inputs,
                            output_hidden_states=False,
                            use_cache=False,
                            return_dict=True,
                        )

                    with nvtx.range("batch_D2H_transfer"):
                        hook_results = self.hook_extractor.get_results()
                    if self.extract_sublayers:
                        all_hidden = hook_results["hidden"]
                        all_sublayer_attn = hook_results["sublayer_attn"]
                        all_sublayer_mlp = hook_results["sublayer_mlp"]
                    else:
                        all_hidden = hook_results  # plain tensor

            # Extract results for each item in batch
            for i in range(batch_size):
                context_info = {
                    "input_tokens": seq_lens[i],
                    "output_tokens": len(self.tokenizer.encode(generated_texts[i]))
                    if generated_texts[i]
                    else 0,
                    "model_type": self.model_type,
                }

                if self.extract_features:
                    if self.generate_text:
                        # Dual hidden states for generation mode
                        context_info["hidden_states_pre"] = all_pre_hidden[i]
                        context_info["hidden_states_post"] = all_post_hidden[i]
                    else:
                        # Single hidden states for forward-only mode
                        context_info["hidden_states"] = all_hidden[i]
                        if all_sublayer_attn is not None:
                            context_info["sublayer_attn"] = all_sublayer_attn[i]
                            context_info["sublayer_mlp"] = all_sublayer_mlp[i]

                results.append((generated_texts[i], context_info))

        if self.verbose:
            print(
                f"[Transformers] Batch of {batch_size}, avg input: {sum(seq_lens) / batch_size:.0f} tokens"
            )

        return results


class MockFeatureExtractor(LLMWrapperBase):
    """Mock feature extractor for testing without GPU.

    Mimics TransformersFeatureWrapper interface with random features.
    Useful for testing the pipeline without loading a real model.
    """

    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 4096,
        generate_text: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize mock feature extractor.

        Args:
            num_layers: Number of hidden layers to simulate
            hidden_dim: Hidden dimension size
            generate_text: Whether to generate mock text
            verbose: Whether to print token counts
        """
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.generate_text = generate_text
        self.verbose = verbose
        self.model_type = "MockModel"

        print("\n=== Mock Feature Extractor ===")
        print(f"Layers: {num_layers}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Generate text: {generate_text}")
        print("=" * 30 + "\n")

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Return mock response with mock hidden states."""
        # Estimate input tokens from message content
        total_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = total_chars // 4  # Rough estimate

        # Generate mock hidden states: (num_layers+1, hidden_dim)
        # +1 for embedding layer
        mock_hidden_pre = np.random.randn(self.num_layers + 1, self.hidden_dim).astype(
            np.float32
        )

        context_info = {
            "input_tokens": input_tokens,
            "output_tokens": 50 if self.generate_text else 0,
            "model_type": self.model_type,
        }

        if self.generate_text:
            # Dual hidden states for generation mode
            mock_hidden_post = np.random.randn(
                self.num_layers + 1, self.hidden_dim
            ).astype(np.float32)
            context_info["hidden_states_pre"] = mock_hidden_pre
            context_info["hidden_states_post"] = mock_hidden_post
            response = json.dumps(
                {
                    "reasoning": "This is a mock response for testing.",
                    "interpretation": "Mock interpretation of the action.",
                    "predicted_outcome": "Unknown - mock mode",
                }
            )
        else:
            # Single hidden states for forward-only mode
            context_info["hidden_states"] = mock_hidden_pre
            response = ""

        if self.verbose:
            print(f"[Mock] Input: ~{input_tokens} tokens (estimated)")

        return response, context_info

    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text) // 4

    @property
    def context_length(self) -> int:
        return 131072

    def generate_batch(
        self, messages_batch: list[list[dict]]
    ) -> list[tuple[str, dict]]:
        """Process a batch of message sequences."""
        results = []
        for messages in messages_batch:
            result = self.generate(messages)
            results.append(result)
        if self.verbose:
            print(f"[Mock] Batch of {len(messages_batch)} processed")
        return results
