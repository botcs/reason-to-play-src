# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
DeepSeek V3.2 feature extraction wrapper using torchrun tensor parallelism.

Loads the DeepSeek demo model format (converted sharded safetensors) and extracts
hidden states at specified token positions using forward hooks.

Prerequisites:
    1. Copy DeepSeek-V3 inference/ folder to deepseek_inference/ in this repo
    2. Convert HF checkpoint to sharded format:
       cd deepseek_inference && python convert.py --hf-ckpt-path $HF --save-path $SAVE --n-experts 256 --model-parallel $MP

Usage:
    torchrun --nproc-per-node 8 -m src.llm_eval.human_replay.extract_features \
        --deepseek-ckpt-path /path/to/converted/shards \
        --deepseek-config deepseek_inference/config_671B_v3.2.json \
        --prompts out/prompts/*.jsonl
"""

import os
import sys
import json
from pathlib import Path

# Add deepseek_inference to path for its internal imports
_deepseek_path = Path(__file__).parent.parent.parent / "deepseek_inference"
if str(_deepseek_path) not in sys.path:
    sys.path.insert(0, str(_deepseek_path))


import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from load_utils import load_sharded_model

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


class DeepSeekHookExtractor:
    """Extracts hidden states at target positions using forward hooks."""

    def __init__(self, model, num_layers: int, hidden_dim: int, device: torch.device):
        self.model = model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.hooks = []
        self.target_positions = None  # (batch,) tensor of positions
        self.layer_outputs = None  # List[Tensor] per layer

    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            if self.target_positions is None:
                return
            # DeepSeek layers return (h, residual) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            # Gather at target positions: (batch, hidden_dim)
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            extracted = hidden[batch_indices, self.target_positions]
            self.layer_outputs[layer_idx] = extracted.to(dtype=torch.bfloat16)

        return hook

    def register_hooks(self):
        """Register hooks on all transformer layers (embedding output skipped -- see transformers_wrapper.py)."""
        self.remove_hooks()
        for i, layer in enumerate(self.model.layers):
            h = layer.register_forward_hook(self._hook_fn(i))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def prepare(self, positions: list[int]):
        """Set target positions for extraction."""
        self.target_positions = torch.tensor(
            positions, dtype=torch.long, device=self.device
        )
        self.layer_outputs = [None] * self.num_layers

    def get_results(self) -> torch.Tensor:
        """Return (batch, num_layers, hidden_dim) tensor."""
        stacked = torch.stack(self.layer_outputs, dim=1)  # (batch, layers, hidden)
        return stacked.cpu()


def broadcast_tensors(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Broadcast tokenized batch from rank 0 to all ranks."""
    if world_size == 1:
        return input_ids, attention_mask

    # Broadcast shape first
    if rank == 0:
        shape = torch.tensor(input_ids.shape, dtype=torch.long, device=device)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=device)
    dist.broadcast(shape, src=0)

    batch_size, seq_len = shape.tolist()

    # Broadcast tensors
    if rank != 0:
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=device
        )

    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    return input_ids, attention_mask


class DeepSeekTorchrunWrapper:
    """
    DeepSeek V3.2 wrapper for torchrun-based tensor parallel feature extraction.

    Only rank 0 returns results; other ranks return None.
    """

    def __init__(self, ckpt_path: str, config_path: str, verbose: bool = False):
        """
        Initialize the wrapper.

        Args:
            ckpt_path: Path to converted shards (contains model0-mp8.safetensors, etc.)
            config_path: Path to config JSON (e.g., inference/config_671B_v3.2.json)
            verbose: Print debug info
        """
        self.verbose = verbose

        # Distributed setup
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        torch.set_default_dtype(torch.bfloat16)

        # Load config
        with open(config_path) as f:
            config_dict = json.load(f)

        # Import DeepSeek model (deepseek_inference/ added to sys.path above)
        from model import Transformer, ModelArgs

        args = ModelArgs(**config_dict)

        self.num_layers = args.n_layers
        self.hidden_dim = args.dim
        self.batch_size = config_dict.get("max_batch_size", 1)

        if self.rank == 0:
            print(
                f"Loading DeepSeek V3.2: {self.num_layers} layers, dim={self.hidden_dim}, batch_size={self.batch_size}"
            )
            print(f"World size: {self.world_size}, Rank: {self.rank}")

        # Initialize model on GPU
        with torch.device("cuda"):
            self.model = Transformer(args)

        # Load shard for this rank
        load_sharded_model(self.model, ckpt_path, self.rank, self.world_size)
        self.model.eval()

        if self.rank == 0:
            print(f"Loaded shards for rank {self.rank}")

        # Tokenizer from HF cache or ckpt_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.tokenizer.chat_template:
            if self.rank == 0:
                print("No chat template found, using DeepSeek V3 template")
            self.tokenizer.chat_template = DEEPSEEK_V3_CHAT_TEMPLATE

        # Hook extractor
        self.hook_extractor = DeepSeekHookExtractor(
            self.model, self.num_layers, self.hidden_dim, self.device
        )
        self.hook_extractor.register_hooks()

        self.model_type = "DeepSeekV3"

        if self.rank == 0:
            print(f"DeepSeek V3.2 ready ({self.num_layers} hooks registered)")
            print("Using DeepGEMM pre-compiled kernels (no JIT)")

    def generate_batch(
        self, messages_batch: list[list[dict]]
    ) -> list[tuple[str, dict]] | None:
        """
        Extract features for a batch of message sequences.

        Args:
            messages_batch: List of message sequences

        Returns:
            List of (empty_string, context_info) on rank 0, None on other ranks.
            context_info contains 'hidden_states' (num_layers+1, hidden_dim) tensor.
        """
        batch_size = len(messages_batch)

        # Rank 0 tokenizes
        if self.rank == 0:
            prompts = [
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in messages_batch
            ]
            self.tokenizer.padding_side = "right"
            tok = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tok["input_ids"].to(self.device)
            attention_mask = tok["attention_mask"].to(self.device)
        else:
            input_ids, attention_mask = None, None

        # Broadcast to all ranks
        input_ids, attention_mask = broadcast_tensors(
            input_ids, attention_mask, self.rank, self.world_size, self.device
        )

        # Compute target positions (last real token per sample)
        seq_lens = attention_mask.sum(dim=1).tolist()
        positions = [int(sl - 1) for sl in seq_lens]

        # Prepare hooks and run forward
        self.hook_extractor.prepare(positions)

        with torch.inference_mode():
            _ = self.model(input_ids, start_pos=0)

        # Only rank 0 collects and returns results
        if self.rank == 0:
            hidden_states = self.hook_extractor.get_results()  # (batch, layers, hidden)
            results = []
            for i in range(batch_size):
                results.append(
                    (
                        "",
                        {
                            "hidden_states": hidden_states[i],  # (layers, hidden)
                            "input_tokens": seq_lens[i],
                            "model_type": self.model_type,
                        },
                    )
                )
            return results
        return None

    def cleanup(self):
        """Remove hooks and cleanup distributed."""
        self.hook_extractor.remove_hooks()
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
