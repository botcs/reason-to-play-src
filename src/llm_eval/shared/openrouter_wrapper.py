# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
OpenRouter LLM wrapper for the llm_eval scaffolding.

Provides LLM generation through OpenRouter API using the OpenAI client.
Fetches model context_length from the OpenRouter API at init.
"""

import json
import os
import time

import httpx
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

from src.llm_eval.shared.llm_wrapper import LLMWrapperBase


class OpenRouterWrapper(LLMWrapperBase):
    """OpenRouter API wrapper for game scaffolding.

    Uses the OpenAI Python client to communicate with OpenRouter's API.
    Requires OPENROUTER_API_KEY environment variable to be set.

    Features:
    - Fetches model context_length from OpenRouter API at init
    - Compatible with any model available on OpenRouter
    - Token usage tracking from API response
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        top_p: float = 0.95,
        reasoning_effort: str | None = None,
        seed: int = 0,
    ):
        """
        Initialize OpenRouter wrapper.

        Args:
            model: OpenRouter model name (e.g., 'deepseek/deepseek-r1-distill-llama-70b')
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature (default: 0.5)
            top_p: Top-p sampling parameter (default: 0.95)
            reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh).
                'none' explicitly disables reasoning tokens via the API.
            seed: Random seed for deterministic sampling.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your API key from https://openrouter.ai/keys"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort or "none"
        self.seed = seed

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Fetch model info (context length + pricing + supported_parameters) from API
        model_info = self._fetch_model_info(model, api_key)
        self._context_length = model_info["context_length"]
        pricing = model_info.get("pricing", {})
        # Pricing is USD per token (string); convert to float
        self._price_per_input_token = float(pricing.get("prompt", 0))
        self._price_per_output_token = float(pricing.get("completion", 0))
        # supported_parameters is used by require_reasoning_support(); if it
        # ever disappears from the response, we must fail loudly rather than
        # silently defaulting to an empty list (which would report all models
        # as non-reasoning).
        if "supported_parameters" not in model_info:
            raise ValueError(
                f"OpenRouter /models response for {model!r} has no "
                f"'supported_parameters' field. The API contract changed; "
                f"capability checks cannot proceed. Got keys: "
                f"{sorted(model_info.keys())}"
            )
        self._supported_parameters = list(model_info["supported_parameters"])

        print("\n=== OpenRouter Backend ===")
        print(f"Model: {model}")
        print(f"Context length: {self._context_length:,} tokens")
        print(f"Max generation tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
        print(f"Reasoning effort: {self.reasoning_effort}")
        print(f"Supports reasoning: {self.supports_reasoning}")
        print(f"Seed: {seed}")
        print(
            f"Pricing: ${self._price_per_input_token * 1e6:.2f}/M input, "
            f"${self._price_per_output_token * 1e6:.2f}/M output"
        )
        print("=" * 30 + "\n")

    @staticmethod
    def _fetch_model_info(model: str, api_key: str) -> dict:
        """Fetch model info (context_length, pricing) from OpenRouter API.

        Retries up to 5 times with exponential backoff on transient errors.
        Raises ValueError if model is not found after all retries succeed.
        """
        for attempt in range(5):
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            if resp.status_code in (408, 429, 502, 503, 504):
                delay = 2 ** (attempt + 1)
                print(
                    f"[OpenRouter] {resp.status_code} fetching models, retrying in {delay}s"
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            for m in resp.json().get("data", []):
                if m["id"] == model:
                    return m
            raise ValueError(
                f"Model {model!r} not found on OpenRouter. "
                f"Check the model ID at https://openrouter.ai/models"
            )
        # Final attempt -- let it raise on failure
        resp.raise_for_status()
        raise ValueError(f"Failed to fetch model info for {model!r} after 5 attempts")

    @property
    def context_length(self) -> int:
        """Model context length in tokens (fetched from OpenRouter API)."""
        return self._context_length

    @property
    def supported_parameters(self) -> list[str]:
        """List of parameters the model accepts (from OpenRouter /models)."""
        return list(self._supported_parameters)

    @property
    def supports_reasoning(self) -> bool:
        """True if OpenRouter lists 'reasoning' as a supported parameter."""
        return "reasoning" in self._supported_parameters

    def require_reasoning_support(self) -> None:
        """Raise if the model does not list 'reasoning' as supported.

        Call this before a run that depends on hidden reasoning traces
        (e.g. rationale_mode='copied-reasoning') so the failure surfaces at
        wrapper init rather than after the first costly API round-trip.
        """
        if not self.supports_reasoning:
            raise ValueError(
                f"Model {self.model!r} does not list 'reasoning' in its OpenRouter "
                f"supported_parameters={self._supported_parameters}. "
                f"Use rationale_mode='action-only' or 'prompted-rationale' for this model, "
                f"or pick a model with native reasoning support."
            )

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate response from chat messages using OpenRouter API.

        Args:
            messages: List of {'role': str, 'content': str} dicts

        Returns:
            Tuple of (generated_text, context_info) where context_info has:
            - input_tokens: number of input tokens
            - output_tokens: number of output tokens
            - max_tokens: max generation tokens
            - pct_capacity: percentage of context used (approximate)
        """
        # Always send reasoning effort -- omitting it lets thinking models
        # reason by default.
        #
        # DO NOT also pass ``reasoning.max_tokens``.  OpenRouter returns
        # 400 when both are set ("Only one of 'reasoning.effort' and
        # 'reasoning.max_tokens' can be specified").  We would have
        # picked max_tokens instead of effort, except that probing on
        # 2026-04-22 showed ``reasoning.max_tokens`` is only
        # best-effort:
        #   - qwen3.5-27b:   respected cap=64 (emitted 64 reasoning
        #                    tokens) but ignored cap=1024 (emitted 1677).
        #   - qwen3.5-9b:    ignored the cap at both 64 and 1024
        #                    (emitted 1614 and 1454 respectively).
        # And even the top-level ``max_tokens`` parameter in the
        # chat.completions.create call is not a hard total-output cap
        # for reasoning models:
        #   - qwen3.5-9b:    top-level max_tokens IS respected as a
        #                    strict reasoning+content cap.
        #   - qwen3.5-27b:   top-level max_tokens only caps VISIBLE
        #                    content; reasoning escapes it (seen
        #                    completion=1129 with cap=200).
        #   - deepseek-v3.2: same -- reasoning escapes the top-level cap.
        #
        # Upshot: the only reliable way to bound the tokens a single
        # call consumes is client-side guardrails (repair-turn echo
        # trimming, harness.truncate_if_needed with
        # context_usage_fraction).  Never trust the provider to respect
        # any reasoning-token cap.  See ``.claude/skills/
        # openrouter-usage.md`` for the full probe history.
        kwargs = {
            "extra_body": {
                "reasoning": {"effort": self.reasoning_effort},
            }
        }

        # Retry with exponential backoff on transient provider / rate-limit errors.
        # Backoff: 2s, 4s, 8s, ... capped at 5 min per wait.
        # Overall budget: 10 h.  No fixed retry count.
        base_delay = 2.0
        max_delay = 300.0  # 5 minutes
        max_total_wait = 36000  # 10 hours
        total_waited = 0.0
        attempt = 0
        retry_errors: list[dict] = []

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                    **kwargs,
                )
            except (
                RateLimitError,
                APIConnectionError,
                APIStatusError,
                json.JSONDecodeError,
            ) as e:
                if isinstance(e, RateLimitError):
                    reason = "Rate limited (429)"
                    code = 429
                elif isinstance(e, json.JSONDecodeError):
                    reason = f"Malformed API response (JSONDecodeError at char {e.pos})"
                    code = -1
                elif isinstance(e, APIStatusError):
                    reason = f"{type(e).__name__}: {e}"
                    code = getattr(e, "status_code", -1)
                    # Context length exceeded is deterministic -- retrying will never help
                    err_str = str(e).lower()
                    if "context_length_exceeded" in err_str or (
                        "context length" in err_str and "token" in err_str
                    ):
                        raise
                else:
                    reason = f"{type(e).__name__}: {e}"
                    code = -1
                delay = min(base_delay * (2**attempt), max_delay)
                if total_waited + delay > max_total_wait:
                    raise
                retry_errors.append(
                    {"attempt": attempt, "reason": reason, "code": code}
                )
                print(
                    f"[LLM] {reason}, retrying in {delay:.0f}s (waited {total_waited:.0f}s total)"
                )
                time.sleep(delay)
                total_waited += delay
                attempt += 1
                continue

            # Check for provider errors returned as empty responses or
            # missing usage metadata.  OpenRouter occasionally returns a
            # response with valid choices but no ``usage`` block (observed
            # on qwen3.5-397b-a17b via some providers); accessing
            # ``response.usage.prompt_tokens`` would crash the run.
            # Treat both cases as transient provider errors.
            if not response.choices or response.usage is None:
                if not response.choices:
                    error_obj = getattr(response, "error", None) or {}
                    error_msg = (
                        error_obj.get("message", "unknown")
                        if isinstance(error_obj, dict)
                        else str(error_obj)
                    )
                    code = (
                        error_obj.get("code", -1) if isinstance(error_obj, dict) else -1
                    )
                    reason = f"Provider error: {error_msg}"
                else:
                    error_obj = {"message": "response.usage is None"}
                    code = -1
                    reason = "Provider returned no usage metadata"
                delay = min(base_delay * (2**attempt), max_delay)
                if total_waited + delay > max_total_wait:
                    raise ValueError(
                        f"OpenRouter returned invalid response for model {self.model} "
                        f"after {attempt + 1} attempts ({total_waited:.0f}s waited). "
                        f"Last error: {error_obj}"
                    )
                retry_errors.append(
                    {"attempt": attempt, "reason": reason, "code": code}
                )
                print(
                    f"[LLM] {reason}, retrying in {delay:.0f}s (waited {total_waited:.0f}s total)"
                )
                time.sleep(delay)
                total_waited += delay
                attempt += 1
                continue

            break

        generated_text = response.choices[0].message.content
        if generated_text is None:
            generated_text = ""

        # Extract hidden reasoning tokens from response (DeepSeek models)
        hidden_reasoning = getattr(response.choices[0].message, "reasoning", None)

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Reasoning tokens (OpenAI-compatible: completion_tokens_details)
        details = getattr(response.usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        # Character counts and ratio
        # Note: completion_tokens already includes reasoning_tokens as a subset
        # (OpenAI API spec), so input_tokens + output_tokens covers everything.
        input_chars = sum(len(m.get("content", "")) for m in messages)
        output_chars = len(generated_text)
        reasoning_chars = len(hidden_reasoning) if hidden_reasoning else 0
        total_chars = input_chars + output_chars + reasoning_chars
        total_tokens = input_tokens + output_tokens
        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0.0

        pct_capacity = (input_tokens / self._context_length) * 100

        # Use the actual billed cost from the OpenRouter API response.
        # The base pricing from the /models endpoint is the *minimum* across
        # providers, but OpenRouter routes to whichever provider has capacity
        # and each provider has its own rates (e.g. Together charges 2x Venice
        # for qwen3.5-9b input tokens).  The response.usage dict includes the
        # real billed amount under the "cost" key.
        raw_usage = response.model_dump().get("usage", {})
        api_cost = raw_usage.get("cost")
        if api_cost is not None:
            call_cost = float(api_cost)
        else:
            # Fallback: estimate from base pricing (will undercount if the
            # request was served by a more expensive provider).
            call_cost = (
                input_tokens * self._price_per_input_token
                + output_tokens * self._price_per_output_token
            )

        print(f"[LLM] Tokens: {input_tokens:,} in, {output_tokens:,} out")
        if reasoning_tokens:
            print(f"[LLM] Reasoning tokens: {reasoning_tokens:,}")
        if hidden_reasoning:
            print(f"[LLM] Hidden reasoning: {len(hidden_reasoning)} chars")
        print(f"[LLM] Call cost: ${call_cost:.4f}")

        context_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "call_cost": call_cost,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "chars_per_token": chars_per_token,
            "max_tokens": self.max_tokens,
            "pct_capacity": pct_capacity,
            "reasoning": hidden_reasoning,
            "retries": attempt,
            "retry_errors": retry_errors,
            "retry_total_wait": total_waited,
        }

        return generated_text, context_info

    def count_tokens(self, text: str) -> int:
        """Approximate token count using rough character-based estimate.

        Note: This is a rough approximation. For accurate counts,
        consider using tiktoken or the model's actual tokenizer.
        """
        # Rough estimate: ~4 characters per token on average
        return len(text) // 4
