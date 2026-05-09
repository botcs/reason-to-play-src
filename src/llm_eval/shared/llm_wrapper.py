# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Abstract base class for LLM wrappers used by the llm_eval scaffolding.
"""

from abc import ABC, abstractmethod


# Llama 3.1 chat template for models without one configured
# Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
# Used by feature extraction wrappers (DeepSpeed, Transformers).
LLAMA_31_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


class LLMWrapperBase(ABC):
    """Abstract base class for LLM wrappers.

    All LLM backends (OpenRouter, etc.) must implement this interface.
    """

    @abstractmethod
    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate response from chat messages.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Tuple of (generated_text, context_info) where context_info has:
            - input_tokens: number of input tokens
            - output_tokens: number of output tokens (if available)
            - max_tokens: max generation tokens
            - pct_capacity: percentage of context used
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Model context length in tokens."""
        pass
