# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Tensor-parallel patches for Qwen3.5 MoE.

The default tp_plan in ``transformers.models.qwen3_5_moe.configuration_qwen3_5_moe``
shards ``self_attn.k_proj`` and ``self_attn.v_proj`` ``colwise``. With
``num_key_value_heads=2`` and ``head_dim=256``, the unsharded KV-projection
output is 512; for any TP size > 2 this shards below ``head_dim`` and the
``view(*input_shape, -1, head_dim)`` call inside
``Qwen3_5MoeAttention.forward`` (line 667 of modeling_qwen3_5_moe.py)
raises ``RuntimeError: shape '[1, S, -1, 256]' is invalid for input of
size ...``.

Two patches are applied together:

1. Override the ``base_model_tp_plan`` entries for ``k_proj``/``v_proj`` to
   ``replicated_with_grad_allreduce`` so every rank holds the full
   ``num_kv_heads * head_dim = 512`` projection.  Weight memory cost per
   layer is trivial (~3 MB); activation cost at 78k tokens is ~80 MB per
   full-attention layer (12 of 48 in this model).
2. Replace ``Qwen3_5MoeAttention.forward`` with a version that derives
   ``num_key_value_groups`` from local Q/KV head counts at run time
   instead of the unsharded config values set at ``__init__``.  ``q_proj``
   stays ``colwise``; KV is now replicated, so local groups
   = ``local_q_heads / num_kv_heads`` (e.g. 4/2=2 with TP=8).

Required for any deployment where ``num_kv_heads < world_size`` -- i.e.
TP>=4 on Qwen3.5-122B-A10B.  Idempotent and a no-op once applied.
"""

from __future__ import annotations

import os

import torch


_PATCHED = False


def apply() -> None:
    """Apply the tp_plan + attention-forward patch.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeAttention,
    )

    plan = Qwen3_5MoeTextConfig.base_model_tp_plan
    plan["layers.*.self_attn.k_proj"] = "replicated_with_grad_allreduce"
    plan["layers.*.self_attn.v_proj"] = "replicated_with_grad_allreduce"

    Qwen3_5MoeAttention.forward = _patched_forward

    _PATCHED = True

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[qwen3_5_moe_torchrun] tp_plan: k_proj/v_proj -> "
            "replicated_with_grad_allreduce; "
            "Qwen3_5MoeAttention.forward replaced (local KV groups)",
            flush=True,
        )


def _patched_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """Replacement for Qwen3_5MoeAttention.forward with local-shape KV groups.

    Structurally identical to upstream (line ~650 of
    modeling_qwen3_5_moe.py) except:

    - ``num_local_q_heads`` and ``num_local_kv_heads`` are read from the
      tensor shapes after the projections, so a colwise-sharded ``q_proj``
      and replicated ``k_proj``/``v_proj`` work together.
    - ``module.num_key_value_groups`` is temporarily swapped to
      ``num_local_q_heads // num_local_kv_heads`` for the duration of the
      attention call so ``repeat_kv`` (used by both the eager and SDPA
      backends) fans out by the local ratio, not the config ratio.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    num_local_q_heads = query_states.shape[-2]

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    num_local_kv_heads = key_states.shape[-3]
    num_local_kv_groups = num_local_q_heads // num_local_kv_heads

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    original_groups = self.num_key_value_groups
    self.num_key_value_groups = num_local_kv_groups
    try:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    finally:
        self.num_key_value_groups = original_groups

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
