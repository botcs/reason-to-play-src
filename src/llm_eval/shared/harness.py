# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Unified harness for VGDL LLM evaluation.

Owns all prompt construction, response parsing, and state management logic.
The harness operates in multi-turn mode only: each turn is a message in an
ongoing conversation. `rationale_mode` controls what (if anything) is kept
as the assistant's rationale in the conversation history:

- action-only        : assistant message is just ``{"action": ...}``
- prompted-rationale : assistant writes rationale inline, full response kept
- copied-reasoning   : rationale is copied from the API's hidden reasoning
                       trace into a synthesized ``{"rationale": ..., "action": ...}``
"""

import json
import random

from src.llm_eval.shared.config import (
    HarnessConfig,
    VALID_RATIONALE_MODE,
    rationale_mode_carries_rationale,
)
from src.llm_eval.shared.prompt_loader import PromptLoader
from src.llm_eval.shared.response_parser import ResponseParser


class Harness:
    """Prompt/response/state manager for multi-turn harness runs."""

    def __init__(
        self,
        config: HarnessConfig,
        prompt_loader: PromptLoader,
        prompt_subdir: str | None = None,
        max_output_tokens: int | None = None,
        game_name: str | None = None,
    ):
        """Initialize harness with config and prompt loader.

        Loads the system prompt from file via prompt_loader.
        Initializes ResponseParser.

        Args:
            config: HarnessConfig dataclass instance.
            prompt_loader: PromptLoader instance for loading prompt text files.
            prompt_subdir: Optional subdirectory to search first (e.g. 'replay').
            max_output_tokens: Per-turn output token cap enforced by the LLM
                backend.  Substituted into the ``{max_output_tokens}``
                placeholder in the loaded prompt text and also used by the
                repair-prompt builder to detect budget-exhaustion parse
                failures.  Required when the loaded prompt contains the
                placeholder; otherwise optional.
            game_name: Game identifier (e.g. ``bait_vgfmri4``).  Required when
                ``suggestion_level='oracle'`` -- the corresponding rule file
                is loaded from ``prompts/game_rules/{game_name}.txt`` and
                substituted into the ``{game_rules}`` placeholder.
        """
        if config.rationale_mode not in VALID_RATIONALE_MODE:
            raise ValueError(
                f"Invalid rationale_mode={config.rationale_mode!r}. "
                f"Must be one of {VALID_RATIONALE_MODE}."
            )
        self._config = config
        self._prompt_loader = prompt_loader
        self._parser = ResponseParser()
        self._max_output_tokens = max_output_tokens

        self._prompt_name = f"{config.rationale_mode}_{config.suggestion_level}"
        raw_prompt = prompt_loader.load(self._prompt_name, subdir=prompt_subdir)
        if "{max_output_tokens}" in raw_prompt:
            if max_output_tokens is None:
                raise ValueError(
                    f"Prompt {self._prompt_name!r} contains a "
                    f"{{max_output_tokens}} placeholder but Harness was "
                    f"constructed without max_output_tokens. Pass "
                    f"max_output_tokens=<cfg.llm.max_tokens>."
                )
            raw_prompt = raw_prompt.replace(
                "{max_output_tokens}", str(max_output_tokens)
            )
        if "{game_rules}" in raw_prompt:
            if game_name is None:
                raise ValueError(
                    f"Prompt {self._prompt_name!r} contains a "
                    f"{{game_rules}} placeholder but Harness was "
                    f"constructed without game_name. Pass "
                    f"game_name=<cfg.game.game>."
                )
            game_rules = prompt_loader.load(game_name, subdir="game_rules")
            raw_prompt = raw_prompt.replace("{game_rules}", game_rules)
        self._system_prompt = raw_prompt

        # Conversation message list: [{"role": ..., "content": ...}, ...]
        self._conversation: list[dict] = [
            {"role": "system", "content": self._system_prompt}
        ]
        # Parallel list of per-message input-token estimates, kept in
        # lockstep with ``_conversation`` (same length, same index).
        # Prefers API-reported counts (``output_tokens + reasoning_tokens``
        # from the call that produced an assistant message; copied-reasoning
        # inlines the reasoning trace into the committed content, so both
        # components contribute to the next turn's prompt_tokens).
        # Falls back to ``len(content)`` as a 1-char:1-token proxy for the
        # system prompt, observation turns, and repair turns where no API
        # count is available.
        self._msg_tokens: list[int] = [len(self._system_prompt)]

        # Fixed action queue for deterministic mock testing (list of action strings).
        # When non-empty, mock_response() pops from here instead of randomizing.
        self.mock_actions: list[str] = []

        # Last valid parsed response (for fallback on parse errors)
        self._last_valid_response: dict | None = None

    def reset(self) -> None:
        """Reset all mutable state for a new play/run."""
        self._conversation = [{"role": "system", "content": self._system_prompt}]
        self._msg_tokens = [len(self._system_prompt)]
        self._last_valid_response = None

    def restore_conversation(self, steps: list[dict]) -> None:
        """Rebuild conversation from saved replay steps (for resume).

        Replays the user/assistant message pairs that build_messages() and
        commit_assistant() would have produced, using the stored per-step
        fields.  Skips crash steps and _level_advance markers.
        """
        self.reset()
        mode = self._config.rationale_mode
        prev_level: int | None = None
        prev_attempt: int | None = None

        for step in steps:
            action = step.get("action", "")
            if action == "_level_advance" or step.get("crash_reason"):
                continue
            response = step.get("response") or {}
            if not response.get("action"):
                continue

            lvl = step["level"]
            attempt = step["attempt"]
            step_num = step["step"]
            formatted_obs = step.get("formatted_obs", "")

            lines: list[str] = []
            is_new_level = prev_level is not None and lvl != prev_level
            is_restart = (
                prev_level is not None and lvl == prev_level and attempt != prev_attempt
            )
            if is_restart or is_new_level:
                lines.append(f"--- NEW TRIAL (Level {lvl}, Attempt {attempt}) ---")
                lines.append("")
            lines.append(f"# Step {step_num} (Level {lvl}, Attempt {attempt})")
            lines.append("")
            lines.append(formatted_obs)
            user_content = "\n".join(lines)

            self._conversation.append({"role": "user", "content": user_content})
            self._msg_tokens.append(len(user_content))

            if mode == "action-only":
                assistant_content = json.dumps({"action": response["action"]})
            elif mode == "prompted-rationale":
                assistant_content = step.get("raw_response", json.dumps(response))
            else:
                rationale = response.get("rationale", step.get("rationale", ""))
                assistant_content = json.dumps(
                    {"rationale": rationale, "action": response["action"]}
                )
            self._conversation.append(
                {"role": "assistant", "content": assistant_content}
            )
            out_tokens = step.get("output_tokens")
            self._msg_tokens.append(
                out_tokens if out_tokens else len(assistant_content)
            )

            prev_level = lvl
            prev_attempt = attempt

    # -------------------------------------------------------------------------
    # build_messages
    # -------------------------------------------------------------------------

    def build_messages(
        self,
        step_num: int,
        formatted_obs: str,
        level: int,
        attempt: int,
        is_restart: bool = False,
        is_new_level: bool = False,
        prev_level: int | None = None,
        prev_outcome: str = "",
        prev_score: int = 0,
    ) -> list[dict]:
        """Append observation to conversation and return full conversation.

        Args:
            step_num: Current global step number.
            formatted_obs: Pre-formatted observation string from ObservationFormatter.
            level: Current level number.
            attempt: Current attempt number (for this level).
            is_restart: True if this step follows a death restart.
            is_new_level: True if this is the first step of a new level.
            prev_level: Previous level number (used in new-level transition message).
            prev_outcome: Outcome of previous trial ('won', 'died', 'timeout', '').
            prev_score: Score at end of previous trial.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        lines: list[str] = []

        # Transition markers: close previous trial, then open new one
        if is_restart or is_new_level:
            if prev_outcome:
                lines.append(
                    f"--- TRIAL ENDED outcome: {prev_outcome}, score: {prev_score} ---"
                )
                lines.append("")
            lines.append(f"--- NEW TRIAL (Level {level}, Attempt {attempt}) ---")
            lines.append("")

        # Header
        lines.append(f"# Step {step_num} (Level {level}, Attempt {attempt})")
        lines.append("")

        # Observation
        lines.append(formatted_obs)

        observation = "\n".join(lines)
        self._conversation.append({"role": "user", "content": observation})
        # User observations have no API-reported token count yet.  They
        # are short structured strings (grid + coordinates), so 1 char =
        # 1 token is a safe proxy -- if anything a slight over-estimate
        # which nudges truncation to fire a touch earlier.
        self._msg_tokens.append(len(observation))

        return list(self._conversation)

    # -------------------------------------------------------------------------
    # parse_strict / commit_assistant / repair helpers
    # -------------------------------------------------------------------------

    def parse_strict(self, raw_response: str, context_info: dict | None = None) -> dict:
        """Parse LLM response, raising on any parse or schema error.

        Does NOT modify the conversation state.  The caller is responsible
        for calling ``commit_assistant`` on success or ``append_repair_turn``
        on failure before regenerating.

        Args:
            raw_response: Raw string response from the LLM.
            context_info: Optional dict with metadata from the LLM API call,
                may contain 'reasoning' key with hidden chain-of-thought.

        Returns:
            Parsed dict with at least the 'action' key.  For copied-reasoning
            mode, the rationale is resolved from either the API reasoning
            trace (preferred) or the response body.

        Raises:
            ValueError / json.JSONDecodeError: on any parse, schema, or
                missing-rationale failure.
        """
        if context_info is None:
            context_info = {}

        parsed = self._parser.parse(raw_response)

        hidden = context_info.get("reasoning")

        mode = self._config.rationale_mode
        if mode == "copied-reasoning":
            # Preferred source: the API's hidden reasoning trace.  For mock
            # backends and other cases where the API does not return one,
            # fall back to a 'rationale' field in the response body.
            rationale = hidden or parsed.get("rationale", "")
            if not rationale:
                raise ValueError(
                    "rationale_mode='copied-reasoning' requires a rationale source: "
                    "either a non-empty 'reasoning' field in context_info (from "
                    "native reasoning models) or a 'rationale' field in the "
                    "response body (for mocks). Neither was present."
                )
            # Surface the resolved rationale so commit_assistant can reuse it
            parsed["_resolved_rationale"] = rationale
        elif mode not in ("action-only", "prompted-rationale"):
            raise ValueError(f"Invalid rationale_mode: {mode!r}")

        return parsed

    def commit_assistant(
        self, parsed: dict, raw_response: str, context_info: dict | None = None
    ) -> None:
        """Append the final assistant message to the conversation.

        Must be called after a successful ``parse_strict``.  Chooses the
        assistant payload based on ``rationale_mode``:

          - action-only        : ``{"action": ...}``
          - prompted-rationale : raw response verbatim
          - copied-reasoning   : ``{"rationale": <resolved>, "action": ...}``
        """
        mode = self._config.rationale_mode
        if mode == "action-only":
            assistant_content = json.dumps({"action": parsed["action"]})
        elif mode == "prompted-rationale":
            assistant_content = raw_response
        else:  # copied-reasoning
            rationale = parsed.get("_resolved_rationale")
            if not rationale:
                # Should not happen -- parse_strict raises if rationale is missing.
                raise ValueError(
                    "commit_assistant(copied-reasoning): parsed dict is missing "
                    "'_resolved_rationale'. Did you skip parse_strict?"
                )
            assistant_content = json.dumps(
                {"rationale": rationale, "action": parsed["action"]}
            )
        self._conversation.append({"role": "assistant", "content": assistant_content})
        # Per-message token estimate.  Prefer the API's own count.
        # ``output_tokens`` maps to OpenRouter's ``completion_tokens``,
        # which INCLUDES the reasoning portion by OpenAI spec
        # (confirmed 2026-04-22 by live probe on qwen3.5-27b: effort=
        # medium returned completion_tokens=1518 with reasoning_tokens
        # =1505 and content ~4 tokens -- reasoning is a subset).  For
        # copied-reasoning the committed ``content`` inlines the
        # reasoning into the JSON; the delta vs ``output_tokens`` is
        # just the JSON-wrapper overhead (~10-20 tokens).  Using
        # ``output_tokens`` alone is therefore accurate within <1%
        # without double-counting.  Fall back to the raw content
        # length when no context_info is available (mock backends).
        if context_info:
            msg_tokens = int(context_info.get("output_tokens") or 0)
            if msg_tokens == 0:
                msg_tokens = len(assistant_content)
        else:
            msg_tokens = len(assistant_content)
        self._msg_tokens.append(msg_tokens)
        # Promote the internal marker into the canonical key so the parsed
        # dict the caller sees carries the resolved rationale (the hidden
        # trace for copied-reasoning) under 'rationale'.  No-op for other
        # modes, which never set _resolved_rationale.
        if "_resolved_rationale" in parsed:
            parsed["rationale"] = parsed.pop("_resolved_rationale")
        self._last_valid_response = parsed

    def append_repair_turn(
        self,
        raw_response: str,
        error: Exception,
        context_info: dict | None = None,
    ) -> None:
        """Append a malformed assistant + mode-specific repair-prompt user msg.

        Used when ``parse_strict`` raised: the agent wants to feed the error
        back to the model for a retry.  The appended pair is meant to be
        stripped (via ``clip_from``) once a valid response is obtained.

        Args:
            raw_response: The malformed raw LLM output (shown back verbatim).
            error: The exception produced by ``parse_strict``.
            context_info: Optional dict from the LLM API call.  When it
                reports ``output_tokens >= max_tokens`` the repair prompt
                additionally tells the model it hit the output cap and must
                shorten its response.
        """
        self._conversation.append({"role": "assistant", "content": raw_response})
        # Failed assistant attempt: raw_response is typically empty (cap
        # saturation) so its token contribution is dominated by the
        # hidden reasoning trace.  ``output_tokens`` already covers
        # both content and reasoning (OpenAI spec -- see commit_assistant
        # for the probe data).
        if context_info:
            failed_tokens = int(context_info.get("output_tokens") or 0)
            if failed_tokens == 0:
                failed_tokens = len(raw_response)
        else:
            failed_tokens = len(raw_response)
        self._msg_tokens.append(failed_tokens)

        repair_prompt = self._build_repair_prompt(error, context_info)
        self._conversation.append({"role": "user", "content": repair_prompt})
        self._msg_tokens.append(len(repair_prompt))

    def clip_from(self, start_idx: int) -> None:
        """Truncate the conversation so messages ``[start_idx:]`` are removed."""
        if start_idx < 1:
            raise ValueError(
                f"clip_from: start_idx={start_idx} would remove the system prompt"
            )
        if start_idx > len(self._conversation):
            raise ValueError(
                f"clip_from: start_idx={start_idx} exceeds conversation length "
                f"{len(self._conversation)}"
            )
        del self._conversation[start_idx:]
        del self._msg_tokens[start_idx:]

    def _build_repair_prompt(
        self, error: Exception, context_info: dict | None = None
    ) -> str:
        """Build the user-facing repair prompt for a parse failure.

        The schema reminder is mode-specific so the model gets the exact
        shape it was asked to produce.  When ``context_info`` reports the
        response was truncated at the output cap, the prompt additionally
        names the cap and asks the model to shorten its reply -- otherwise
        the retry just burns the same tokens and hits the same wall.

        For copied-reasoning, the failed attempt's hidden reasoning trace
        is echoed back verbatim so the model has continuity with what it
        was just thinking.  The echo is part of the ephemeral repair turn
        and gets stripped via ``clip_from`` on a successful parse, so the
        prompt explicitly instructs the model not to reference the echoed
        text in its new rationale (the reference would dangle after clip).
        """
        mode = self._config.rationale_mode
        if mode == "action-only":
            schema = '{"action": "<up|down|left|right|action|reset|wait>"}'
            extra = ""
        elif mode == "prompted-rationale":
            schema = (
                '{"rationale": "<brief notes>", '
                '"action": "<up|down|left|right|action|reset|wait>"}'
            )
            extra = ""
        else:  # copied-reasoning
            schema = '{"action": "<up|down|left|right|action|reset|wait>"}'
            extra = " Your reasoning is captured from your hidden thinking tokens automatically."

        budget_warning = ""
        reasoning_echo = ""
        if context_info:
            out_tokens = context_info.get("output_tokens")
            cap = context_info.get("max_tokens")
            if out_tokens is not None and cap is not None and out_tokens >= cap:
                if mode == "copied-reasoning":
                    cause = (
                        "your hidden reasoning consumed the entire output budget, "
                        "leaving no tokens for the JSON"
                    )
                    fix = (
                        "think far more concisely (a few hundred reasoning tokens at most) "
                        "so the response has room to emit"
                    )
                elif mode == "prompted-rationale":
                    cause = "your response was truncated mid-rationale"
                    fix = "keep the rationale to a couple of short sentences"
                else:  # action-only
                    cause = "your response was truncated mid-content"
                    fix = (
                        "emit only the single-action JSON object, nothing else "
                        "(no commentary, no preamble)"
                    )
                budget_warning = (
                    f" You hit the per-turn output-token cap of {cap} "
                    f"(completion_tokens={out_tokens}); {cause}. "
                    f"On this retry, {fix}."
                )

            reasoning = context_info.get("reasoning")
            if reasoning and mode == "copied-reasoning":
                # Guardrail: trim the echo to ``4 * max_tokens`` chars
                # (1 char = 1 token proxy).  The provider occasionally
                # returns a monster reasoning trace that exceeds
                # ``reasoning.max_tokens`` despite our request, and a
                # faithful verbatim echo would push the retry off a
                # cliff (context-window overflow or a repeat of the
                # same cap saturation on the next call).  4x headroom
                # beyond the generation cap lets the model see enough
                # of its failure pattern to course-correct without
                # giving up the entire context window to self-
                # referencing slop.  Keep the TAIL (most recent
                # chain-of-thought) since that is where the model was
                # when it ran out of budget.
                echo_budget_chars = 4 * (cap if cap is not None else 8192)
                trimmed_note = ""
                if len(reasoning) > echo_budget_chars:
                    n_trimmed = len(reasoning) - echo_budget_chars
                    reasoning = reasoning[-echo_budget_chars:]
                    trimmed_note = (
                        f" [earlier {n_trimmed} chars of reasoning trimmed "
                        f"to fit the echo budget]"
                    )
                reasoning_echo = (
                    f" Your previous hidden reasoning was:{trimmed_note} "
                    f"{reasoning!r}. Take it into account so you do not "
                    f"start from scratch, but be much less verbose this "
                    f"turn and rephrase any conclusions more concisely. "
                    f"Do not refer to this echoed reasoning in your next "
                    f"rationale -- it is about to be discarded from the "
                    f"conversation, so any reference to it will dangle."
                )

        return (
            f"Your previous response could not be parsed: {error}. "
            f"Please respond with valid JSON only, matching this schema: "
            f"{schema}.{extra}{budget_warning}{reasoning_echo}"
        )

    def _parse_replay_response(self, raw_response: str) -> dict:
        """Parse replay generation response (no action required).

        In human replay, the model returns rationale only.
        The action comes from the participant's behavioral data.
        """
        try:  # noqa: S110 - LLM output boundary
            return self._parser.parse_replay(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[WARN] Replay parse error: {e}\nRaw response: {raw_response!r}")
            return {}

    # -------------------------------------------------------------------------
    # inject_known_action -- action-only replay path
    # -------------------------------------------------------------------------

    def inject_known_action(self, action: str) -> None:
        """Inject a known action as an assistant message (action-only mode).

        Args:
            action: Action string (e.g. 'up', 'down', 'left', 'right', 'wait').

        Raises:
            ValueError: If called in any mode other than 'action-only'.
        """
        if self._config.rationale_mode != "action-only":
            raise ValueError(
                f"inject_known_action is only valid in rationale_mode='action-only' "
                f"(got {self._config.rationale_mode!r}). Other modes must go through "
                f"record_replay_step with a generation response."
            )

        assistant_content = json.dumps({"action": action})
        self._conversation.append({"role": "assistant", "content": assistant_content})
        self._msg_tokens.append(len(assistant_content))

    # -------------------------------------------------------------------------
    # mock_response
    # -------------------------------------------------------------------------

    def mock_response(self) -> dict:
        """Generate a mock response matching the expected schema.

        If mock_actions is set (a list of action strings), pops the next
        action from it.  Otherwise falls back to a random choice.

        Returns:
            Dict matching the expected response format for the current mode.
        """
        if self.mock_actions:
            action = self.mock_actions.pop(0)
        else:
            actions = ["up", "down", "left", "right", "wait", "reset"]
            action = random.choice(actions)

        mode = self._config.rationale_mode
        if mode == "action-only":
            return {"action": action}
        if mode == "prompted-rationale":
            return {"rationale": "Mock rationale", "action": action}
        # copied-reasoning: mock imitates hidden reasoning via a synthetic rationale
        return {"rationale": "Mock hidden reasoning", "action": action}

    def record_mock_response(self, response_dict: dict) -> None:
        """Record a mock response, updating the conversation as the live path would.

        Args:
            response_dict: The dict returned by mock_response().
        """
        mode = self._config.rationale_mode
        if mode == "action-only":
            assistant_content = json.dumps({"action": response_dict["action"]})
        elif mode == "prompted-rationale":
            assistant_content = json.dumps(response_dict)
        else:  # copied-reasoning -- require a rationale rather than silently
            #                       writing an empty one
            if "rationale" not in response_dict or not response_dict["rationale"]:
                raise ValueError(
                    "record_mock_response: rationale_mode='copied-reasoning' "
                    "requires the mock response dict to include a non-empty "
                    f"'rationale'. Got: {response_dict!r}"
                )
            assistant_content = json.dumps(
                {
                    "rationale": response_dict["rationale"],
                    "action": response_dict["action"],
                }
            )
        self._conversation.append({"role": "assistant", "content": assistant_content})
        self._msg_tokens.append(len(assistant_content))

    # -------------------------------------------------------------------------
    # truncate_if_needed
    # -------------------------------------------------------------------------

    def truncate_if_needed(self, context_length: int, max_tokens: int) -> None:
        """Drop oldest message pairs if approaching context limit.

        Sums per-message token counts from ``_msg_tokens`` -- assistant
        messages carry the API-reported ``output_tokens + reasoning_tokens``
        from the call that produced them (copied-reasoning inlines the
        reasoning trace, so both components contribute); the system
        prompt, observation turns, and ephemeral repair turns fall back
        to a 1-char:1-token proxy.

        Reserves ``max_tokens`` for generation, then enforces that the
        input prompt occupies at most ``CONTEXT_USAGE_FRACTION`` of the
        remainder.  Keeps the system message and at least the last 10
        user/assistant pairs.

        Args:
            context_length: Model context window in tokens (from API).
            max_tokens: Generation budget in tokens (reserved from context).
        """
        assert len(self._msg_tokens) == len(self._conversation), (
            f"_msg_tokens ({len(self._msg_tokens)}) out of sync with "
            f"_conversation ({len(self._conversation)})"
        )

        total_tokens = sum(self._msg_tokens)
        # ``context_usage_fraction`` is the slice-1..10 effective regime
        # (0.5 default).  Raised by the context-window sweep to probe
        # high-fill degradation curves; the config validates (0, 1].
        input_limit = int(
            (context_length - max_tokens) * self._config.context_usage_fraction
        )

        if total_tokens < input_limit:
            return

        # Keep system message (index 0) and at least the last 10 message pairs
        min_keep = 21  # system + 10 user/assistant pairs
        if len(self._conversation) <= min_keep:
            return

        # Calculate how many messages to drop from index 1
        n_to_drop = 0
        running = total_tokens
        i = 1
        while i < len(self._conversation) - min_keep + 1 and running > input_limit:
            running -= self._msg_tokens[i]
            n_to_drop += 1
            i += 1

        # Ensure we drop in pairs (user+assistant) to maintain alternation
        if n_to_drop % 2 != 0:
            n_to_drop += 1

        if n_to_drop > 0:
            dropped_count = n_to_drop
            self._conversation = [self._conversation[0]] + self._conversation[
                1 + n_to_drop :
            ]
            self._msg_tokens = [self._msg_tokens[0]] + self._msg_tokens[1 + n_to_drop :]
            print(
                f"[TRUNCATE] Dropped {dropped_count} messages "
                f"({total_tokens - running} tokens) from conversation. "
                f"Remaining: {len(self._conversation)} messages, "
                f"~{running} tokens (limit {input_limit})."
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def conversation_length(self) -> int:
        """Number of messages in conversation."""
        return len(self._conversation)

    @property
    def prompt_name(self) -> str:
        """The derived prompt filename (e.g. 'prompted-rationale_elaborate')."""
        return self._prompt_name

    @property
    def system_prompt(self) -> str:
        """The loaded system prompt text."""
        return self._system_prompt

    @property
    def rationale_mode(self) -> str:
        """The rationale_mode this harness is operating in."""
        return self._config.rationale_mode

    @property
    def carries_rationale(self) -> bool:
        """Whether assistant messages include a 'rationale' field."""
        return rationale_mode_carries_rationale(self._config.rationale_mode)

    # -------------------------------------------------------------------------
    # Replay methods (human replay pipeline)
    # -------------------------------------------------------------------------

    def build_replay_generation_messages(
        self,
        step_num: int,
        formatted_obs: str,
        level: int,
        attempt: int,
        action_taken: str,
        is_restart: bool = False,
        is_new_level: bool = False,
        prev_level: int | None = None,
        prev_outcome: str = "",
        prev_score: int = 0,
    ) -> list[dict]:
        """Build messages for the replay generation (imputation) phase.

        For 'action-only' mode, this is identical to build_messages() because
        no LLM generation happens in that mode.

        For 'copied-reasoning' mode, the 'Your action: X' hint is appended to
        the last user message so the model's hidden reasoning converges on
        the human's actual action.

        'prompted-rationale' is not supported in the replay pipeline.

        Args:
            step_num: Current global step number.
            formatted_obs: Pre-formatted observation string from ObservationFormatter.
            level: Current level number.
            attempt: Current attempt number.
            action_taken: Human action string (e.g. 'RIGHT', 'UP').
            is_restart: True if this step follows a death restart.
            is_new_level: True if first step of a new level.
            prev_level: Previous level number.
            prev_outcome: Outcome of previous trial.
            prev_score: Score at end of previous trial.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        mode = self._config.rationale_mode
        if mode == "prompted-rationale":
            raise ValueError(
                "rationale_mode='prompted-rationale' is not supported in the replay "
                "pipeline. Use 'action-only' or 'copied-reasoning'."
            )

        messages = self.build_messages(
            step_num=step_num,
            formatted_obs=formatted_obs,
            level=level,
            attempt=attempt,
            is_restart=is_restart,
            is_new_level=is_new_level,
            prev_level=prev_level,
            prev_outcome=prev_outcome,
            prev_score=prev_score,
        )

        if mode == "action-only":
            # No generation needed -- action will be injected directly
            return messages

        # copied-reasoning: hint the model with the human's action
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            messages[last_user_idx]["content"] += f"\n\nYour action: {action_taken}"

        return messages

    def record_replay_step(
        self,
        action_taken: str,
        generation_response: str | None = None,
        generation_context: dict | None = None,
    ) -> None:
        """Record a completed replay step, updating internal state.

        - action-only       : injects the human action as the assistant message.
        - copied-reasoning  : takes the rationale from the API's hidden reasoning
                              trace (generation_context['reasoning']) and builds
                              a ``{"rationale", "action"}`` assistant message.
                              The 'Your action: X' hint is stripped from the
                              last user message so the extraction-phase
                              conversation looks like actual gameplay.

        Args:
            action_taken: Human action string (e.g. 'RIGHT', 'UP').
            generation_response: Optional raw LLM response from the generation
                phase.  Only used for diagnostics in copied-reasoning mode.
            generation_context: Optional dict from the LLM call, contains
                'reasoning' with the hidden CoT for copied-reasoning mode.
        """
        mode = self._config.rationale_mode
        action_lower = action_taken.lower()

        if mode == "action-only":
            self.inject_known_action(action_lower)
            return

        if mode == "prompted-rationale":
            raise ValueError(
                "rationale_mode='prompted-rationale' is not supported in the replay "
                "pipeline. Use 'action-only' or 'copied-reasoning'."
            )

        # copied-reasoning: rationale must come from somewhere -- either the
        # API's hidden reasoning trace or (for mock backends) the response
        # body.  An empty rationale would silently corrupt the imputed trace.
        hidden = None
        if generation_context is not None:
            hidden = generation_context.get("reasoning")
        rationale = hidden
        if not rationale and generation_response:
            parsed = self._parse_replay_response(generation_response)
            rationale = parsed.get("rationale")
        if not rationale:
            raise ValueError(
                "rationale_mode='copied-reasoning' replay step produced no "
                "rationale: generation_context['reasoning'] was empty and the "
                "response body had no 'rationale' field. This would corrupt "
                "the imputed trace. Check that the imputation model actually "
                "supports reasoning and is returning traces."
            )
        assistant_dict = {"rationale": rationale, "action": action_lower}
        assistant_content = json.dumps(assistant_dict)
        self._conversation.append({"role": "assistant", "content": assistant_content})
        # Replay path: prefer the API-reported counts from the imputation
        # call.  ``output_tokens`` already includes reasoning (OpenAI spec).
        if generation_context:
            replay_tokens = int(generation_context.get("output_tokens") or 0)
            if replay_tokens == 0:
                replay_tokens = len(assistant_content)
        else:
            replay_tokens = len(assistant_content)
        self._msg_tokens.append(replay_tokens)

        # Strip "Your action: X" from the last user message so the
        # extraction-phase conversation looks like actual gameplay.
        self._strip_action_taken_from_last_user()

    def _strip_action_taken_from_last_user(self) -> None:
        """Remove 'Your action: X' suffix from the last user message."""
        for i in range(len(self._conversation) - 1, -1, -1):
            if self._conversation[i]["role"] == "user":
                content = self._conversation[i]["content"]
                for marker in ("\n\nYour action: ", "\n\nAction taken: "):
                    idx = content.rfind(marker)
                    if idx != -1:
                        new_content = content[:idx]
                        self._conversation[i]["content"] = new_content
                        self._msg_tokens[i] = len(new_content)
                        break
                break

    def get_replay_extraction_messages(self) -> list[dict]:
        """Return the current conversation for feature extraction.

        For both supported replay modes the conversation has already been
        restructured (action-only: inject_known_action; copied-reasoning:
        synthesized rationale + action with user hint stripped).
        """
        return list(self._conversation)
