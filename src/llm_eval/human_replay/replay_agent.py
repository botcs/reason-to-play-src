# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
ReplayAgent: Run human replay through the unified Harness pipeline.

Two-phase pipeline:

Phase 1 (Imputation): For 'copied-reasoning' mode, an LLM generates hidden
reasoning for each human action (hinted with the known action). The hidden
trace is copied into context as the player's stated rationale. 'action-only'
mode skips generation entirely -- the human's action is injected directly.
'prompted-rationale' is not supported in the replay pipeline.

Phase 2 (Feature Extraction): The saved traces are encoded through the LLM
to extract hidden features. Extraction messages use the gameplay system
prompt (not the imputation prompt) so the trace looks like actual gameplay.
"""

from __future__ import annotations

from src.llm_eval.shared.config import HarnessConfig
from src.llm_eval.shared.event_logger import EventLogger
from src.llm_eval.shared.harness import Harness
from src.llm_eval.shared.llm_wrapper import LLMWrapperBase
from src.llm_eval.shared.prompt_loader import PromptLoader
from src.llm_eval.shared.observation_formatter import ObservationFormatter
from src.llm_eval.human_replay.zstate_adapter import (
    ZstateAdapter,
    is_action_frame,
)


class ReplayAgent:
    """Run human replay data through the unified Harness pipeline.

    Processes zstates from behavioral data and produces prompt records
    in the same format as GameplayAgent's generative play.
    """

    def __init__(
        self,
        harness_config: HarnessConfig,
        prompt_loader: PromptLoader,
        block_size: int,
        llm: LLMWrapperBase | None = None,
        max_output_tokens: int | None = None,
        game_name: str | None = None,
    ):
        """Initialize ReplayAgent.

        Args:
            harness_config: Harness configuration (mode, suggestion_level, etc.).
            prompt_loader: PromptLoader for loading system prompts.
            block_size: Pixel block size for grid conversion (20 or 35).
            llm: Optional LLM wrapper for copied-reasoning generation.
                Required for 'copied-reasoning' mode to produce meaningful
                rationale. If None with 'copied-reasoning', rationale will
                be empty.
            max_output_tokens: Per-turn output token cap; forwarded to the
                Harness so the imputation prompt and repair prompts can
                reference it.  Required when the loaded prompt contains the
                ``{max_output_tokens}`` placeholder.
            game_name: Game identifier (e.g. ``bait_vgfmri4``); forwarded to
                the Harness for oracle-level rule injection.
        """
        if harness_config.rationale_mode == "prompted-rationale":
            raise ValueError(
                "rationale_mode='prompted-rationale' is not supported in the replay "
                "pipeline. Use 'action-only' or 'copied-reasoning'."
            )
        # Phase 1 prompts live in prompts/replay/ for copied-reasoning mode;
        # action-only does not generate, so prompt choice is cosmetic
        # (extraction swaps to the gameplay prompt anyway).
        self.harness = Harness(
            harness_config,
            prompt_loader,
            prompt_subdir="replay",
            max_output_tokens=max_output_tokens,
            game_name=game_name,
        )
        self.obs_formatter = ObservationFormatter()
        self.event_logger = EventLogger()
        self.adapter = ZstateAdapter(block_size=block_size)
        self._config = harness_config
        self._llm = llm

        # Phase 2: load the corresponding gameplay prompt for extraction.
        # This replaces the imputation/replay system prompt in the saved
        # messages so the trace looks like actual gameplay.
        gameplay_prompt_name = (
            f"{harness_config.rationale_mode}_{harness_config.suggestion_level}"
        )
        gp = prompt_loader.load(gameplay_prompt_name)
        if "{max_output_tokens}" in gp and max_output_tokens is not None:
            gp = gp.replace("{max_output_tokens}", str(max_output_tokens))
        if "{game_rules}" in gp and game_name is not None:
            gp = gp.replace(
                "{game_rules}", prompt_loader.load(game_name, subdir="game_rules")
            )
        self._gameplay_system_prompt = gp

    def run_replay(
        self,
        states: list[dict],
        level: int = 0,
        attempt: int = 0,
        action_frames_only: bool = False,
        stride: int = 1,
    ) -> list[dict]:
        """Process all frames of a play, returning prompt records.

        Each record contains the messages that would be sent to the LLM
        at that step, along with metadata about the action taken.

        Args:
            states: List of zstate dicts from decompressed play.
            level: Level number for this play.
            attempt: Attempt number for this level (1-based, matching generative agent).
            action_frames_only: If True, only process frames with keyPressType != None.
            stride: Process every Nth frame (1 = all frames).

        Returns:
            List of prompt record dicts, one per processed frame.
        """
        if not states:
            return []

        # Reset adapter for new play (new object IDs etc.)
        self.adapter.reset()

        # Pre-register objects from first frame for deterministic ID ordering
        self.adapter.register_objects(states[0])

        records: list[dict] = []
        step_num = 0
        prev_zstate = None

        # Clean causal model: a "step" represents (obs_i, action_i) where the
        # participant SAW states[i] and pressed a key (captured in the
        # keystate field of the next raw zstate) that produced states[i+1].
        # Because the Tomov engine writes keystate[i] into zstate[i] (i.e.,
        # retrospective to the state it produced), the raw frame whose
        # is_action_frame() returns True at index `frame_idx` encodes the
        # action the user pressed while viewing states[frame_idx - 1].  We
        # therefore expose the step as frame_idx - 1 (= state_index).  Frame
        # 0 is never an action step because there is no pre-action state.
        total_steps = sum(
            1
            for i, z in enumerate(states)
            if i > 0 and self._should_process_frame(i, z, action_frames_only, stride)
        )

        for frame_idx, zstate in enumerate(states):
            if not self._should_process_frame(
                frame_idx, zstate, action_frames_only, stride
            ):
                prev_zstate = zstate
                continue

            # Skip keypresses recorded at frame 0: there is no preceding state
            # that the user could have seen before pressing, so no step exists.
            if frame_idx == 0:
                prev_zstate = zstate
                continue

            print(f"      step {step_num}/{total_steps}", flush=True)

            # adapt_frame(prev_zstate, zstate): prev_observation is what the
            # user saw, observation is the post-action state, action_idx is
            # the keypress at zstate (= the action the user pressed).
            adapted = self.adapter.adapt_frame(prev_zstate, zstate)

            curr_obs = adapted["observation"]
            prev_obs = adapted["prev_observation"]
            events = adapted["events"]
            action_idx = adapted["action_idx"]
            action_name = adapted["action_name"]
            won = adapted["won"]
            lose = adapted["lose"]
            timeout = adapted["timeout"]
            score = adapted["score"]
            pre_action_score = adapted["prev_score"]

            # The Tomov engine writes ended/win on the frame AFTER the
            # triggering action.  Attribute the terminal outcome (and
            # any lagged score) to the action that caused it.
            if not (won or lose or timeout) and frame_idx + 1 < len(states):
                nxt = states[frame_idx + 1]
                if nxt.get("ended"):
                    nxt_win = nxt.get("win")
                    won = nxt_win is True or nxt_win == 1
                    lose = nxt_win is False or nxt_win == -1
                    timeout = nxt_win is None
                    nxt_score = nxt.get("score")
                    if nxt_score is not None and int(nxt_score) != score:
                        score = int(nxt_score)

            reward = score - pre_action_score

            resources = self._get_avatar_resources(prev_obs)

            formatted_obs = self.obs_formatter.format_observation(
                obs=prev_obs,
                score=pre_action_score,
                reward=0,
                resources=resources,
            )

            gen_messages = self.harness.build_replay_generation_messages(
                step_num=step_num,
                formatted_obs=formatted_obs,
                level=level,
                attempt=attempt,
                action_taken=action_name,
            )

            generation_response = None
            generation_context = None
            if (
                self._config.rationale_mode == "copied-reasoning"
                and self._llm is not None
            ):
                raw, ctx = self._llm.generate(gen_messages)
                generation_response = raw
                generation_context = ctx

            imputation_messages = None
            if self._config.rationale_mode == "copied-reasoning":
                imputation_messages = [dict(m) for m in gen_messages]
                if generation_response:
                    imputation_messages.append(
                        {"role": "assistant", "content": generation_response}
                    )

            self.harness.record_replay_step(
                action_name,
                generation_response=generation_response,
                generation_context=generation_context,
            )

            extraction_messages = self._build_extraction_messages(
                gen_messages, generation_response
            )

            action_log = self.event_logger.log_action(
                step_num=step_num,
                action=action_idx,
                prev_obs=prev_obs,
                curr_obs=curr_obs,
                events=events,
                reward=reward,
                won=won,
                lose=lose,
                prev_score=pre_action_score,
                curr_score=score,
                level=level,
                attempt=attempt,
                timeout=timeout,
            )

            record = {
                "step_num": step_num,
                # state_index: the frame where the participant SAW the state.
                # Under the clean causal model (obs_i, action_i), this is
                # frame_idx - 1 because the keystate of the raw zstate at
                # frame_idx was captured during the tick that transformed
                # states[frame_idx-1] into states[frame_idx].
                "frame_idx": frame_idx - 1,
                # Wall-clock Unix timestamp of the observation the participant
                # saw before pressing this step's action.  zstate["ts"] is
                # preserved by realign_zstate_positions (it only patches
                # positions), so prev_zstate["ts"] is the pre-action frame's
                # recording time.
                "realworld_ts": prev_zstate["ts"],
                "action_taken": action_name,
                "action_idx": action_idx,
                "action_log": action_log,
                "reward": reward,
                "score": score,
                "won": won,
                "lose": lose,
                "timeout": timeout,
                "events": events,
                "prev_obs": prev_obs,
                "messages": extraction_messages,
                "imputation_messages": imputation_messages,
                "formatted_obs": formatted_obs,
                "generation_response": generation_response,
                "generation_context": generation_context,
            }
            records.append(record)

            step_num += 1
            prev_zstate = zstate

        # A play can end on an idle frame long after the last action
        # (especially for timeouts).  If the terminal zstate flags that
        # case, stamp the outcome onto the last action record so the
        # downstream prompt reconstruction emits TRIAL ENDED and the
        # JS viewer shows [WIN]/[LOSE]/[LEVEL TIMEOUT] next to the
        # final step.
        if records and states:
            terminal = states[-1]
            last = records[-1]
            if terminal.get("ended") and not (
                last["won"] or last["lose"] or last["timeout"]
            ):
                win_value = terminal.get("win")
                last["won"] = win_value is True or win_value == 1
                last["lose"] = win_value is False or win_value == -1
                last["timeout"] = win_value is None
                if last["won"]:
                    last["action_log"] += " [WIN]"
                elif last["lose"]:
                    last["action_log"] += " [LOSE]"
                elif last["timeout"]:
                    last["action_log"] += " [LEVEL TIMEOUT]"

        return records

    def _should_process_frame(
        self,
        frame_idx: int,
        zstate: dict,
        action_frames_only: bool,
        stride: int,
    ) -> bool:
        """Determine if a frame should be processed based on sampling config."""
        if action_frames_only:
            return is_action_frame(zstate)

        if stride > 1:
            return frame_idx % stride == 0

        return True

    def _build_extraction_messages(
        self,
        gen_messages: list[dict],
        generation_response: str | None,
    ) -> list[dict]:
        """Build extraction-ready messages from the current conversation.

        The extraction messages use the gameplay system prompt (not the
        imputation/replay prompt) so the trace looks like actual gameplay.

        For ``action-only``, the record snapshots ``gen_messages`` -- the
        pre-decision context shown to the model (system + user turn, plus
        any prior user/assistant pairs).

        For ``copied-reasoning``, the record snapshots the full conversation
        after the synthesized assistant message was appended.  This matches
        the legacy behaviour feature extraction relies on.
        """
        if self._config.rationale_mode == "action-only":
            result = [dict(m) for m in gen_messages]
        else:
            result = list(self.harness._conversation)
        return self._swap_system_prompt(result)

    def _swap_system_prompt(self, messages: list[dict]) -> list[dict]:
        """Replace the system prompt with the gameplay prompt for extraction."""
        if messages and messages[0]["role"] == "system":
            messages[0] = {"role": "system", "content": self._gameplay_system_prompt}
        return messages

    @staticmethod
    def _get_avatar_resources(obs: list) -> dict | None:
        """Extract avatar resources from adapted observation grid."""
        for column in obs:
            for cell in column:
                for obj in cell:
                    if obj["name"] == "avatar":
                        resources = obj.get("resources", {})
                        return resources if resources else None
        return None
