# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
GameplayAgent: Main game loop orchestrating all LLM scaffolding components.

Handles:
- Running VGDL games and capturing state/events
- Delegating prompt construction and response parsing to the Harness
- Auto-restart on death, auto-advance on level win
- Idle frame injection to match human reaction time
- Incremental JSON logging with replay state capture

Terminology
-----------
Two clocks run through the system:

  frame -- one game-engine update (one env.step() call), 50 ms at 20 Hz.
           Config/variables: level_frame_budget/total_frame_budget,
           idle_frames, level_frames, total_frames.
  step  -- one LLM decision that produces one game move.
           Config/variables: step_num, total_steps, level_steps.

One step costs (1 + idle_frames) frames.

Idle frame injection
--------------------
Human participants in the Tomov et al. 2023 fMRI experiment played at 20 Hz
(50 ms per frame). Empirical analysis of the behavioral data (7034 plays,
32 subjects, 13 games) shows that humans are idle on ~71-87% of frames
depending on the game, with a median inter-action interval of ~150 ms
(3 frames at 20 Hz). The idle rate varies by game type:

    Game              Idle%   Median RT
    chase variants    ~71%    ~150 ms   (reactive)
    zelda/bait/avoid  ~79-81% ~150 ms   (moderate)
    helper/sokoban    ~84-87% ~200-250 ms (deliberate)

Without idle injection, an LLM agent using the same 1200-frame budget gets
1200 meaningful actions vs ~240 for humans, while NPCs advance the same
number of frames. This gives the LLM ~5x more control over the game.

To match human dynamics, after each LLM step we inject idle_frames NO_OP
frames into the engine (configured via game.idle_frames). With idle_frames=4,
each LLM step costs 5 game frames (1 action + 4 NO_OPs), so 1200 frames
yield ~240 LLM steps -- matching human action counts while preserving
identical NPC dynamics.
"""

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import gym

from src.utils import register_game, get_repo_path
from src.llm_eval.generative_gameplay.export_replay import (
    capture_state,
    write_replay_file,
)
from src.llm_eval.shared.event_logger import EventLogger
from src.llm_eval.shared.observation_formatter import ObservationFormatter
from src.llm_eval.shared.response_parser import ResponseParser
from src.llm_eval.shared.config import (
    BlockedCurriculaAdvancement,
    Config,
    FixedBudgetAdvancement,
    HarnessConfig,
    LLMConfig,
    GameConfig,
    LoggingConfig,
    rationale_mode_to_effort,
    validate_config,
)
from src.llm_eval.shared.harness import Harness
from src.llm_eval.shared.prompt_loader import PromptLoader


# Prefixes to strip from Hydra CLI override keys for shorter filenames
_STRIP_PREFIXES = ("game.", "llm.", "logging.", "harness.")


def _build_overrides_slug(argv: list[str]) -> str:
    """Build a filename slug from Hydra CLI overrides.

    Each key=value arg becomes key-value with group prefixes stripped.
    Args without '=' are kept as-is. Result joined with '_'.
    Falls back to 'default' if no overrides.
    """
    parts = []
    for arg in argv:
        # Skip Hydra internal flags
        if arg.startswith("--") or arg.startswith("+") or arg.startswith("~"):
            continue
        # Replace = with - to form key-value slug
        slug = arg.replace("=", "-")
        # Strip common group prefixes
        for prefix in _STRIP_PREFIXES:
            if slug.startswith(prefix):
                slug = slug[len(prefix) :]
                break
        # Sanitize: replace / with -, strip anything not alphanumeric/dash/underscore/dot
        slug = slug.replace("/", "-")
        slug = re.sub(r"[^a-zA-Z0-9._-]", "", slug)
        if slug:
            parts.append(slug)
    return "_".join(parts) if parts else "default"


def build_interactions_from_game(game) -> set[tuple[str, str, str]]:
    """Build the set of known (color1, effect, color2) interaction tuples
    directly from a VGDL game's domain and sprite registry.

    Iterates over ``domain.collision_eff``, resolves each effect's
    actor/actee sprite types to their leaf colors (expanding parent types),
    and returns the full set.  EOS interactions are excluded because the
    tracker cannot resolve them to a colour (no state entry for EOS).
    """
    registry = game.sprite_registry
    domain = game.domain

    def _leaf_keys(stype: str) -> list[str]:
        """Return all registered leaf sprite keys that match *stype*."""
        if stype in registry.sprite_keys:
            return [stype]
        return [key for key in registry.sprite_keys if stype in registry.stypes[key]]

    def _color(key: str) -> str | None:
        """Extract the colour string from a sprite type's class args."""
        img = registry.class_args[key].get("img")
        if img and "/" in img:
            return img.split("/")[1].upper()
        return None

    interactions: set[tuple[str, str, str]] = set()
    for effect in domain.collision_eff:
        if effect.actee_stype == "EOS":
            continue
        for actor_key in _leaf_keys(effect.actor_stype):
            c1 = _color(actor_key)
            if c1 is None:
                continue
            for actee_key in _leaf_keys(effect.actee_stype):
                c2 = _color(actee_key)
                if c2 is None:
                    continue
                interactions.add((c1, effect.name, c2))
    return interactions


class InteractionTracker:
    """Track unique interactions discovered from game events."""

    def __init__(self, known_interactions: set[tuple[str, str, str]] | None = None):
        """
        Initialize InteractionTracker.

        Args:
            known_interactions: The full set of (color1, effect, color2)
                tuples that the game can produce.  Built automatically via
                ``build_interactions_from_game`` after the env is created.
                If *None*, ``coverage`` returns 0.0 until it is set.
        """
        self.known: set[tuple[str, str, str]] = known_interactions or set()
        self.seen: set[tuple[str, str, str]] = set()  # (color1, effect, color2)
        self.discovery_order: list[
            tuple[int, tuple[str, str, str]]
        ] = []  # (step, interaction)
        self.all_instances: list[
            tuple[int, tuple[str, str, str], tuple[str, str]]
        ] = []  # (step, interaction, (obj_id1, obj_id2))

    def record_events(
        self,
        events: list,
        prev_obs: list,
        step: int,
    ) -> None:
        """Extract and record unique interactions from game events.

        Args:
            events: List of event tuples from game step
            prev_obs: Observation before the action (to look up colors)
            step: Current step number
        """
        for event in events:
            effect_name = event[0]
            sprite1 = event[1]  # (name, obj_id)
            sprite2 = event[2]  # (name, obj_id)

            color1 = self._get_color_from_obs(prev_obs, sprite1[1])
            color2 = self._get_color_from_obs(prev_obs, sprite2[1])

            if color1 is None or color2 is None:
                continue

            # Normalize colors to uppercase to avoid duplicate interactions with different casing
            interaction = (color1.upper(), effect_name, color2.upper())

            # Track all instances (for detailed logging)
            self.all_instances.append((step, interaction, (sprite1[1], sprite2[1])))

            # Track unique interactions (for coverage)
            if interaction not in self.seen:
                self.seen.add(interaction)
                self.discovery_order.append((step, interaction))
                if self.known and interaction not in self.known:
                    print(
                        f"[InteractionTracker] WARNING: discovered interaction "
                        f"{interaction} not in known set (step {step}). "
                        f"Adding to known -- build_interactions_from_game may be incomplete."
                    )
                    self.known.add(interaction)

    def _get_color_from_obs(self, obs: list, obj_id: str) -> str | None:
        """Get color name for an object by ID."""
        if obs is None:
            return None
        for column in obs:
            for cell in column:
                for obj in cell:
                    if obj["obj_id"] == obj_id:
                        color = obj.get("color")
                        if color is None:
                            return None
                        if isinstance(color, str):
                            return color
                        # Convert RGB tuple to color name
                        from src.utils import COLOR_DICT

                        rgb_to_color = {v: k for k, v in COLOR_DICT.items()}
                        rgb_tuple = tuple(color) if isinstance(color, list) else color
                        return rgb_to_color.get(rgb_tuple)
        return None

    @property
    def total(self) -> int:
        """Number of known interactions (for backward compat)."""
        return len(self.known)

    @property
    def coverage(self) -> float:
        """Return fraction of known interactions discovered."""
        return len(self.seen) / len(self.known) if self.known else 0.0

    @property
    def interactions_discovered(self) -> int:
        """Return number of unique interactions discovered."""
        return len(self.seen)

    def reset(self) -> None:
        """Reset tracker for a new run (but keep known interactions)."""
        self.seen.clear()
        self.discovery_order.clear()
        self.all_instances.clear()


class ParseRetryBudgetExhausted(ValueError):
    """Raised when the parse-retry budget has been fully consumed.

    Carries the full ``attempts`` list (including the terminal failed
    attempt) so the caller can persist it into the step_log before
    re-raising.  Without this, the terminal retries live only in
    ``output.log`` and the replay JSON shows an abrupt end.  Inherits
    from ValueError so existing ``except ValueError`` callers keep
    working.
    """

    def __init__(self, message: str, attempts: list[dict]) -> None:
        super().__init__(message)
        self.attempts = attempts


class ConsecutiveSaturatedRetriesExceeded(ValueError):
    """Raised when N consecutive LLM calls within a single step return with
    ``output_tokens >= max_tokens``.

    Prevents the "saturate-retry-saturate" loop first observed on slice 11
    lemmings (``<RUN-ID>`` -- 122b-a10b elaborate, max=440,069 reasoning
    tokens on a single step = ~10 provider-ceiling-saturated retries before
    the parse-retry budget caught up).  The saturate-retry-saturate loop
    costs ~N x max_tokens in reasoning on a single decision, far more than
    the intended per-step budget.  Checked after every LLM call within
    ``_generate_with_parse_retry``; the counter resets at the start of each
    step and on any non-saturated call within a step.

    Carries the full ``attempts`` list (including the terminal saturated
    attempt) so the caller can persist it into the step_log before
    re-raising.  Inherits from ``ValueError`` to match the existing
    ``except ValueError`` paths around parse-retry handling.
    """

    def __init__(self, message: str, attempts: list[dict]) -> None:
        super().__init__(message)
        self.attempts = attempts


class ApiBudgetExhausted(Exception):
    """Raised when cumulative run cost has crossed ``llm.api_budget``.

    Checked after every LLM call (including parse-retry attempts).
    Carries the failing cost / cap pair plus any in-progress parse-retry
    attempts so the caller can write a forensic crash step before
    re-raising.  Deliberately NOT a ValueError -- this is a cost-control
    abort, not malformed data.
    """

    def __init__(
        self,
        message: str,
        total_cost: float,
        api_budget: float,
        attempts: list[dict],
    ) -> None:
        super().__init__(message)
        self.total_cost = total_cost
        self.api_budget = api_budget
        self.attempts = attempts


class GameplayAgent:
    """Main scaffolding agent that orchestrates LLM gameplay.

    Accepts a Config object and delegates prompt construction, response parsing,
    and conversation management to a Harness instance.
    """

    def __init__(
        self, cfg: Config, llm: object | None = None, timestamp: str | None = None
    ):
        """
        Initialize GameplayAgent from a Config object.

        Args:
            cfg: Structured Config instance (from load_config or built manually).
            llm: Optional pre-initialized LLM instance. If None, lazily loaded
                from cfg.llm settings when first needed.
        """
        self.cfg = cfg

        # Create Harness instance -- owns prompt/response/conversation state
        prompt_loader = PromptLoader()
        self.harness = Harness(
            cfg.harness,
            prompt_loader,
            max_output_tokens=cfg.llm.max_tokens,
            game_name=cfg.game.game,
        )

        # Components
        self.event_logger = EventLogger()
        self.obs_formatter = ObservationFormatter()
        self.response_parser = ResponseParser()
        self._llm = llm

        # Interaction tracker (known interactions populated after first env creation)
        self.interaction_tracker = InteractionTracker()

        # Compute system prompt hash for logging
        self.system_prompt_hash = hashlib.sha256(
            self.harness.system_prompt.encode()
        ).hexdigest()[:8]

        # Precompute filename: timestamp + hash of full CLI args
        self._args_hash = hashlib.sha256(" ".join(sys.argv).encode()).hexdigest()[:8]
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_filename = (
            f"{self._timestamp}_{self._args_hash}.generative.replay.json.gz"
        )
        self._log_filepath = os.path.join(cfg.logging.output_dir, self._log_filename)

        # S3 incremental sync (configured via set_s3_sync)
        self._s3_client = None  # boto3 S3 client (from ExpDB._s3)
        self._s3_bucket = None
        self._s3_key = None

        # W&B integration
        self._wandb_run = None
        self._wandb_save_interval = 20  # Save JSON file every N steps
        self._last_wandb_save_step = 0
        # _log_filepath already set above from overrides slug + timestamp
        self._interaction_table = None  # W&B table for interaction logging
        self._provider_error_table = None  # W&B table for provider errors
        if cfg.logging.wandb_project:
            import wandb

            self._wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                config={
                    "game": cfg.game.game,
                    "model": cfg.llm.model,
                    "prompt_name": self.harness.prompt_name,
                    "system_prompt_hash": self.system_prompt_hash,
                    "rationale_mode": cfg.harness.rationale_mode,
                    "suggestion_level": cfg.harness.suggestion_level,
                    "max_levels": cfg.game.max_levels,
                    "advancement_strategy": cfg.game.advancement.strategy,
                    "seed": cfg.seed,
                    "backend": cfg.llm.backend,
                    "temperature": cfg.llm.temperature,
                    "top_p": cfg.llm.top_p,
                },
            )
            # Initialize interaction table
            self._interaction_table = wandb.Table(
                columns=[
                    "step",
                    "level",
                    "attempt",
                    "color1",
                    "effect",
                    "color2",
                    "obj_id1",
                    "obj_id2",
                    "first_occurrence",
                ]
            )
            # Initialize provider error table
            self._provider_error_table = wandb.Table(
                columns=[
                    "step",
                    "level",
                    "attempt",
                    "retry_num",
                    "error_code",
                    "reason",
                ]
            )

        # State (reset in run())
        self.total_score: int = 0
        self.step_num: int = 0
        self.current_level: int = 0
        self.attempt: int = 0

        # Track resource name -> color mapping (for inventory display)
        self.resource_colors: dict[str, str] = {}

        # Logging
        self.log_data: dict = {}

        # Track last context info from LLM (token usage, hidden reasoning)
        self._last_context_info: dict | None = None

        # Game environment
        self.env = None
        self._game_registered = False

    @property
    def llm(self):
        """Lazy-load LLM when first needed."""
        if self._llm is None and self.cfg.llm.model and self.cfg.llm.backend != "mock":
            if self.cfg.llm.backend == "openrouter":
                from src.llm_eval.shared.openrouter_wrapper import OpenRouterWrapper

                effort = rationale_mode_to_effort(self.cfg.harness.rationale_mode)
                self._llm = OpenRouterWrapper(
                    model=self.cfg.llm.model,
                    max_tokens=self.cfg.llm.max_tokens,
                    temperature=self.cfg.llm.temperature,
                    top_p=self.cfg.llm.top_p,
                    reasoning_effort=effort,
                    seed=self.cfg.seed,
                )
                if self.cfg.harness.rationale_mode == "copied-reasoning":
                    self._llm.require_reasoning_support()
            else:
                raise ValueError(f"Unknown backend: {self.cfg.llm.backend}")
        return self._llm

    def set_s3_sync(self, s3_client: object, bucket: str, key: str) -> None:
        """Enable incremental S3 uploads alongside local file saves.

        Args:
            s3_client: A boto3 S3 client (e.g. ExpDB._s3).
            bucket: S3 bucket name.
            key: S3 object key for the replay file.
        """
        self._s3_client = s3_client
        self._s3_bucket = bucket
        self._s3_key = key

    def _register_and_make_env(self, level: int = 0) -> None:
        """Register game with gym and create environment."""
        repo_path = get_repo_path()
        game_name = self.cfg.game.game
        game_path = f"{repo_path}games/{game_name}_v0/"

        # Check if game exists
        game_file = os.path.join(game_path, f"{game_name}.txt")
        level_file = os.path.join(game_path, f"{game_name}_lvl{level}.txt")

        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Game file not found: {game_file}")
        if not os.path.exists(level_file):
            raise FileNotFoundError(f"Level file not found: {level_file}")

        game_id = f"{game_name}-v0"

        # Register game (gym handles re-registration gracefully)
        if not self._game_registered:
            register_game(game_name, level=level)
            self._game_registered = True

        # Create environment
        self.env = gym.make(game_id, level_file=level_file)

        # Seed the game engine's random generator for reproducibility
        self.env.game.set_seed(self.cfg.seed)

        # Build known interactions from the game definition (once)
        if not self.interaction_tracker.known:
            self.interaction_tracker.known = build_interactions_from_game(self.env.game)

            # Resolve {sprite_type} placeholders in oracle system prompts
            # using canonical colors from the game's sprite registry.
            registry = self.env.game.sprite_registry
            canon_colors: dict[str, str] = {}
            for key in registry.sprite_keys:
                img = registry.class_args[key].get("img")
                if img and "/" in img:
                    canon_colors[key] = img.split("/")[1].upper()
            if canon_colors:
                sp = self.harness._system_prompt
                for sprite_type, color in canon_colors.items():
                    sp = sp.replace("{" + sprite_type + "}", color)
                self.harness._system_prompt = sp
                self.harness._conversation[0]["content"] = sp

    def _append_state(self, lvl: int, action_log: str | None = None) -> None:
        """Capture the current env state and stamp it with (level, attempt).

        The replay viewer's frame-mode iterator requires ``level`` and
        ``attempt`` on every saved state so it can draw per-play
        separators in the action log; generative gameplay knows both at
        capture time.  Matches the human-replay pipeline's
        ``viewer_states`` stamping in ``run_replay.process_game``.

        ``action_log`` is the per-frame action-log string describing the
        transition OUT of this state (``states[i] -> states[i+1]``).  It
        is stamped onto the state so the replay viewer's frame-mode
        iterator can show intermediate NPC/spawn ticks between LLM
        decisions, mirroring ``run_replay.process_game``'s per-frame
        stamping.  The initial state captured at ``env.reset()`` has no
        incoming transition, so it is stored without an ``action_log``.
        """
        state = capture_state(self.env)
        state["level"] = lvl
        state["attempt"] = self.attempt
        if action_log is not None:
            state["action_log"] = action_log
        self._replay_states.append(state)

    def _count_levels(self) -> int:
        """Count available levels for this game."""
        repo_path = get_repo_path()
        game_name = self.cfg.game.game
        game_path = f"{repo_path}games/{game_name}_v0/"

        level = 0
        while os.path.exists(os.path.join(game_path, f"{game_name}_lvl{level}.txt")):
            level += 1
        return level

    def restore_from_replay(self, replay_data: dict) -> tuple[int, dict]:
        """Restore agent state from a saved replay for level-boundary resume.

        Returns (resume_level, resume_state) where resume_state is a dict
        of counter overrides to pass to run().
        """
        steps = replay_data.get("steps", [])
        if not steps:
            raise ValueError("Replay has no steps -- nothing to resume from")

        resume_level = 0
        for step in reversed(steps):
            if step.get("action") == "_level_advance":
                resume_level = step["level"] + 1
                break

        self.harness.restore_conversation(steps)

        total_retries = sum(len(s.get("parse_retries", [])) for s in steps)

        seen: set[tuple[str, str, str]] = set()
        discovery_order: list[tuple[int, tuple[str, str, str]]] = []
        if "discovery_order" in replay_data:
            for entry in replay_data["discovery_order"]:
                interaction = tuple(entry["interaction"])
                seen.add(interaction)
                discovery_order.append((entry["step"], interaction))

        last_real_step = None
        for step in reversed(steps):
            if step.get("action") not in ("_level_advance", "") and not step.get(
                "crash_reason"
            ):
                last_real_step = step
                break

        prev_outcome = ""
        prev_score = 0
        if last_real_step:
            if last_real_step.get("won"):
                prev_outcome = "won"
            elif last_real_step.get("lose"):
                prev_outcome = "died"

        total_input = sum(s.get("input_tokens") or 0 for s in steps)
        total_output = sum(s.get("output_tokens") or 0 for s in steps)
        total_reasoning = sum(s.get("reasoning_tokens") or 0 for s in steps)
        total_cost = sum(s.get("call_cost") or 0 for s in steps)

        real_steps = [
            s
            for s in steps
            if s.get("action") != "_level_advance" and not s.get("crash_reason")
        ]
        step_num = len(real_steps)
        total_frames = step_num * (1 + self.cfg.game.idle_frames)

        cumulative_wins = sum(1 for s in real_steps if s.get("won"))
        cumulative_losses = sum(1 for s in real_steps if s.get("lose"))

        resume_state = {
            "step_num": step_num,
            "total_frames": total_frames,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_reasoning_tokens": total_reasoning,
            "total_cost": total_cost,
            "cumulative_wins": cumulative_wins,
            "cumulative_losses": cumulative_losses,
            "cumulative_success": cumulative_wins - cumulative_losses,
            "parse_retries_consumed": total_retries,
            "prev_outcome": prev_outcome,
            "prev_score": prev_score,
            "interaction_seen": seen,
            "interaction_discovery_order": discovery_order,
            "prior_steps": steps,
            "prior_states": replay_data.get("states", []),
        }

        print(
            f"Resume: restored {step_num} steps, {cumulative_wins}W/{cumulative_losses}L, "
            f"${total_cost:.2f}, resuming at level {resume_level}"
        )
        return resume_level, resume_state

    def run(self, start_level: int = 0, resume_state: dict | None = None) -> dict:
        """
        Run game from start_level until max_levels complete or all levels done.

        Args:
            start_level: Level to start from
            resume_state: If provided, override counters with restored values
                from a prior run (see restore_from_replay).

        Returns:
            Dict with final results including outcome, total_steps, final_level
        """
        game_name = self.cfg.game.game
        max_levels = self.cfg.game.max_levels
        advancement = self.cfg.game.advancement

        # Initialize state
        self.total_score = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost = 0.0
        self.step_num = 0
        self.current_level = start_level
        self.attempt = 0
        self.level_frames = (
            0  # frames used on the current level (resets on level advance)
        )
        self.level_steps = (
            0  # steps used on the current level (resets on level advance)
        )
        self.total_frames = 0  # global frame counter (never resets)
        self.consecutive_level_wins = 0  # consecutive wins on current level
        self.consecutive_run_losses = 0  # deaths since last win (run-wide)
        self.cumulative_wins = 0  # total level completions across entire run
        self.cumulative_losses = 0  # total deaths across entire run
        self.cumulative_success = 0  # cumulative_wins - cumulative_losses
        self._prev_outcome = ""  # outcome of previous trial (for transition markers)
        self._prev_score = 0  # score at end of previous trial
        # Per-run parse-retry budget.  Decremented each time the model
        # produces a malformed response that we repair via a follow-up
        # prompt.  Exhaustion aborts the run.
        self._parse_retry_remaining = self.cfg.llm.parse_retry_budget

        # Initialize log -- reuse the timestamp from __init__
        timestamp = self._timestamp

        # Build meta field with run configuration
        import subprocess

        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_repo_path(),
        ).stdout.strip()
        self.log_data = {
            "meta": {
                "game": game_name,
                "model": self.cfg.llm.model,
                "backend": self.cfg.llm.backend,
                "start_level": start_level,
                "rationale_mode": self.cfg.harness.rationale_mode,
                "max_levels": max_levels,
                "advancement_strategy": advancement.strategy,
                "prompt_name": self.harness.prompt_name,
                "system_prompt_hash": self.system_prompt_hash,
                "suggestion_level": self.cfg.harness.suggestion_level,
                "temperature": self.cfg.llm.temperature,
                "top_p": self.cfg.llm.top_p,
                "max_tokens": self.cfg.llm.max_tokens,
                "run_token_budget": self.cfg.llm.run_token_budget,
                "api_budget": self.cfg.llm.api_budget,
                "seed": self.cfg.seed,
                "git_hash": git_hash,
                "command": " ".join(sys.argv),
                "started_at": timestamp,
                "wandb_run_url": self._wandb_run.url if self._wandb_run else None,
            },
            "game": game_name,
            "model": self.cfg.llm.model,
            "start_level": start_level,
            "max_levels": max_levels,
            "prompt_name": self.harness.prompt_name,
            "system_prompt": self.harness.system_prompt,
            "started_at": timestamp,
            "steps": [],
            "final_level": start_level,
            "total_steps": 0,
            "outcome": "running",
        }

        # Inline replay state capture -- states captured during the run
        self._replay_states = []
        self._replay_save_interval = 10  # Save .replay.json.gz every N steps
        self._last_replay_save_step = 0

        if resume_state is not None:
            self.step_num = resume_state["step_num"]
            self.total_frames = resume_state["total_frames"]
            self.total_input_tokens = resume_state["total_input_tokens"]
            self.total_output_tokens = resume_state["total_output_tokens"]
            self.total_reasoning_tokens = resume_state["total_reasoning_tokens"]
            self.total_cost = resume_state["total_cost"]
            self.cumulative_wins = resume_state["cumulative_wins"]
            self.cumulative_losses = resume_state["cumulative_losses"]
            self.cumulative_success = resume_state["cumulative_success"]
            self._prev_outcome = resume_state["prev_outcome"]
            self._prev_score = resume_state["prev_score"]
            self._parse_retry_remaining -= resume_state["parse_retries_consumed"]
            self.interaction_tracker.seen = resume_state["interaction_seen"]
            self.interaction_tracker.discovery_order = resume_state[
                "interaction_discovery_order"
            ]
            self.log_data["steps"] = list(resume_state["prior_steps"])
            self._replay_states = list(resume_state["prior_states"])
            self._last_replay_save_step = self.step_num
            self.log_data["meta"]["resumed_at_level"] = start_level
            self.log_data["meta"]["resumed_from_steps"] = resume_state["step_num"]

        # Determine number of levels
        num_levels = self._count_levels()
        level_cap = (
            num_levels if max_levels == 0 else min(num_levels, start_level + max_levels)
        )
        print(
            f"Game {game_name} has {num_levels} levels (playing {level_cap - start_level})"
        )

        # Main game loop
        while self.current_level < level_cap:
            print(f"\n=== Level {self.current_level}, Attempt {self.attempt} ===")

            # Compute remaining frames for this attempt based on strategy
            if isinstance(advancement, FixedBudgetAdvancement):
                remaining_frames = advancement.level_frame_budget - self.level_frames
                print(
                    f"  Level budget: {self.level_frames}/{advancement.level_frame_budget} used, "
                    f"{remaining_frames} remaining"
                )
            else:
                remaining_frames = advancement.total_frame_budget - self.total_frames
                print(
                    f"  Global budget: {self.total_frames}/{advancement.total_frame_budget} used, "
                    f"{remaining_frames} remaining"
                )

            if remaining_frames <= 0:
                break

            if self.total_output_tokens >= self.cfg.llm.run_token_budget:
                break

            attempt_start_step = self.step_num

            won, died = self._run_level(self.current_level, remaining_frames)

            attempt_decisions = self.step_num - attempt_start_step
            attempt_frames = attempt_decisions * (1 + self.cfg.game.idle_frames)
            self.level_frames += attempt_frames
            self.level_steps += attempt_decisions
            self.total_frames += attempt_frames

            # Update cumulative performance counters
            if won:
                self.cumulative_wins += 1
                self.cumulative_success += 1
                self.consecutive_level_wins += 1
                self.consecutive_run_losses = 0
            if died:
                self.cumulative_losses += 1
                self.cumulative_success -= 1
                self.consecutive_level_wins = 0
                self.consecutive_run_losses += 1

            # Determine level outcome for logging
            token_budget_exhausted = (
                self.total_output_tokens >= self.cfg.llm.run_token_budget
            )
            if isinstance(advancement, FixedBudgetAdvancement):
                level_timed_out = self.level_frames >= advancement.level_frame_budget
            else:
                level_timed_out = self.total_frames >= advancement.total_frame_budget

            # Consecutive-losses early stop (blocked_curricula only).
            # mcl=0 disables the check; mcl>0 aborts after that many deaths
            # without an intervening win, to avoid burning credits on a run
            # that is clearly stuck.
            mcl = (
                advancement.max_consecutive_losses
                if isinstance(advancement, BlockedCurriculaAdvancement)
                else 0
            )
            consecutive_losses_abort = mcl > 0 and self.consecutive_run_losses >= mcl

            if token_budget_exhausted:
                print(
                    f"Level {self.current_level} TOKEN BUDGET EXHAUSTED "
                    f"({self.total_output_tokens:,} output tokens / "
                    f"{self.cfg.llm.run_token_budget:,} budget, "
                    f"${self.total_cost:.2f} spent)"
                )
                level_outcome = "token_budget"
                break
            elif consecutive_losses_abort:
                print(
                    f"Level {self.current_level} CONSECUTIVE LOSSES ABORT "
                    f"({self.consecutive_run_losses} deaths in a row, "
                    f"limit {mcl})"
                )
                level_outcome = "consecutive_losses_abort"
                break
            elif won:
                print(
                    f"Level {self.current_level} WON! ({self.level_frames} level frames, "
                    f"{self.total_frames} total frames)"
                )
                level_outcome = "won"
            elif level_timed_out:
                print(
                    f"Level {self.current_level} BUDGET EXHAUSTED "
                    f"({self.attempt} attempts)"
                )
                level_outcome = "frame_timeout"
            elif died:
                print(f"Level {self.current_level} - DIED, restarting...")
                level_outcome = "died"
            else:
                level_outcome = "incomplete"

            # Log level outcome to W&B
            if self._wandb_run:
                import wandb

                wandb.log(
                    {
                        "level_completed": self.current_level,
                        "attempt_steps": attempt_decisions,
                        "level_steps_total": self.level_steps,
                        "level_frames_total": self.level_frames,
                        "total_frames": self.total_frames,
                        "level_attempts": self.attempt,
                        "level_outcome": level_outcome,
                    }
                )

            # Advancement decision depends on strategy
            if isinstance(advancement, FixedBudgetAdvancement):
                # fixed_budget: advance only on per-level timeout (Tomov 2023)
                should_advance = level_timed_out
            else:
                # blocked_curricula: advance on consecutive wins OR global timeout
                mastery_reached = (
                    won
                    and self.consecutive_level_wins
                    >= advancement.consecutive_wins_required
                )
                if mastery_reached:
                    level_outcome = "mastery"
                    print(
                        f"  MASTERY on level {self.current_level} "
                        f"({self.consecutive_level_wins} consecutive wins)"
                    )
                should_advance = mastery_reached
                # Global timeout ends the run, NOT advances the level
                if level_timed_out and not mastery_reached:
                    break

            if should_advance:
                advance_log = (
                    f"[{self.step_num}] LEVEL_ADVANCE -> level {self.current_level} "
                    f"advanced ({level_outcome}: "
                    f"{self.level_frames} level frames, {self.attempt} attempts)"
                )
                step_log = {
                    "step": self.step_num,
                    "level": self.current_level,
                    "attempt": self.attempt,
                    "action": "_level_advance",
                    "action_log": advance_log,
                    "response": {},
                    "state_summary": f"Level advanced due to {level_outcome}",
                    "reward": 0,
                    "won": False,
                    "lose": False,
                    "timeout": False,
                    "level_outcome": level_outcome,
                    "level_frames": self.level_frames,
                }
                self.log_data["steps"].append(step_log)

                self._prev_outcome = "won"
                self._prev_score = self.total_score
                self.current_level += 1
                self.attempt = 0
                self.level_frames = 0  # reset per-level frame counter
                self.level_steps = 0  # reset per-level step counter
                self.consecutive_level_wins = 0  # reset for next level
                self.total_score = 0

            elif won:
                # Won but not advancing -- replay same level
                print(f"  Replaying level {self.current_level}...")
                self._prev_outcome = "won"
                self._prev_score = self.total_score
                self.attempt += 1
                self.total_score = 0

            elif died:
                # Death -- restart same level
                self._prev_outcome = "died"
                self._prev_score = self.total_score
                self.attempt += 1
                self.total_score = 0

        # Determine final outcome
        if self.current_level >= level_cap:
            self.log_data["outcome"] = "completed"
        else:
            self.log_data["outcome"] = "incomplete"

        self.log_data["final_level"] = self.current_level
        self.log_data["total_steps"] = self.step_num
        self.log_data["cumulative_wins"] = self.cumulative_wins
        self.log_data["cumulative_losses"] = self.cumulative_losses
        self.log_data["cumulative_success"] = self.cumulative_success
        self.log_data["total_input_tokens"] = self.total_input_tokens
        self.log_data["total_output_tokens"] = self.total_output_tokens
        self.log_data["total_reasoning_tokens"] = self.total_reasoning_tokens
        self.log_data["total_cost"] = self.total_cost
        self.log_data["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add interaction coverage to log
        self.log_data["interactions_discovered"] = (
            self.interaction_tracker.interactions_discovered
        )
        self.log_data["interaction_coverage"] = self.interaction_tracker.coverage
        self.log_data["discovery_order"] = [
            {"step": step, "interaction": list(interaction)}
            for step, interaction in self.interaction_tracker.discovery_order
        ]

        self._save_log_final()

        # Finish W&B run
        if self._wandb_run:
            import wandb

            wandb.log(
                {
                    "final_outcome": self.log_data["outcome"],
                    "final_level": self.current_level,
                    "total_steps": self.step_num,
                    "final_cumulative_wins": self.cumulative_wins,
                    "final_cumulative_losses": self.cumulative_losses,
                    "final_cumulative_success": self.cumulative_success,
                    "final_interactions_discovered": self.interaction_tracker.interactions_discovered,
                    "final_interaction_coverage": self.interaction_tracker.coverage,
                    "final_input_tokens": self.total_input_tokens,
                    "final_output_tokens": self.total_output_tokens,
                    "final_reasoning_tokens": self.total_reasoning_tokens,
                    "final_cost": self.total_cost,
                }
            )
            # Log the complete tables
            if self._interaction_table is not None:
                wandb.log({"interaction_sequence": self._interaction_table})
            if self._provider_error_table is not None:
                wandb.log({"provider_errors": self._provider_error_table})
            wandb.finish()

        return {
            "outcome": self.log_data["outcome"],
            "final_level": self.current_level,
            "total_steps": self.step_num,
            "cumulative_wins": self.cumulative_wins,
            "cumulative_losses": self.cumulative_losses,
            "cumulative_success": self.cumulative_success,
        }

    def _run_level(self, lvl: int, remaining_frames: int) -> tuple[bool, bool]:
        """
        Run single level (unified for single-turn and multi-turn).

        Delegates prompt construction and response parsing to self.harness.

        Args:
            lvl: Level number to run
            remaining_frames: How many more frames this attempt may consume.
                For fixed_budget: level_frame_budget - level_frames.
                For blocked_curricula: total_frame_budget - total_frames.

        Returns:
            Tuple of (won, died)
        """

        # Reset resource colors for new level
        self.resource_colors = {}

        # Setup environment for this level
        self._register_and_make_env(lvl)
        obs, info = self.env.unwrapped.reset()
        self._append_state(lvl)

        prev_obs = info["state"]
        prev_score = 0
        won = False
        died = False
        attempt_frames = 0

        # Track if this is the first step for transition markers
        is_first_step = True
        is_restart = self.attempt > 0
        is_new_level = self.step_num > 0  # Not the very first step of the game
        prev_level = lvl - 1 if is_new_level else None

        while not won and not died:
            # Check remaining frame budget
            if remaining_frames > 0 and attempt_frames >= remaining_frames:
                break

            # Check per-run output token budget
            if self.total_output_tokens >= self.cfg.llm.run_token_budget:
                print(
                    f"[TOKEN BUDGET] Run output token budget exhausted: "
                    f"{self.total_output_tokens:,} / {self.cfg.llm.run_token_budget:,}"
                )
                break

            # Each step costs 1 + idle_frames game frames
            attempt_frames += 1 + self.cfg.game.idle_frames

            # Get current observation
            obs = info["state"]

            # Wall-clock timestamp of the observation the LLM is about to
            # decide from.  Mirrors the human-replay pipeline's realworld_ts
            # (sourced from zstate["ts"]) so downstream feature extraction can
            # tag features with the decision's real time regardless of
            # gameplay source.
            realworld_ts = time.time()

            # Get avatar resources if available
            resources = self._get_avatar_resources(obs)

            # Format observation for prompt
            formatted_obs = self.obs_formatter.format_observation(
                obs=obs,
                score=self.total_score,
                reward=0,  # Will be updated after action
                resources=resources,
                resource_colors=self.resource_colors,
            )

            # Build messages via harness
            if is_first_step:
                messages = self.harness.build_messages(
                    step_num=self.step_num,
                    formatted_obs=formatted_obs,
                    level=lvl,
                    attempt=self.attempt,
                    is_restart=is_restart,
                    is_new_level=is_new_level and not is_restart,
                    prev_level=prev_level,
                    prev_outcome=self._prev_outcome,
                    prev_score=self._prev_score,
                )
                is_first_step = False
            else:
                messages = self.harness.build_messages(
                    step_num=self.step_num,
                    formatted_obs=formatted_obs,
                    level=lvl,
                    attempt=self.attempt,
                )

            # Truncate if approaching context limit (multi-turn only, no-op for single-turn)
            if self.llm is not None:
                self.harness.truncate_if_needed(
                    self.llm.context_length, self.cfg.llm.max_tokens
                )

            # Get LLM response
            parse_retry_log: list[dict] = []
            if self.llm is None:
                # Mock mode -- mock responses are always well-formed by
                # construction; no retry needed.
                response_dict = self.harness.mock_response()
                raw_response = json.dumps(response_dict)
                context_info = {}
                self.harness.record_mock_response(response_dict)
            else:
                try:
                    response_dict, raw_response, context_info, parse_retry_log = (
                        self._generate_with_parse_retry(messages)
                    )
                except (
                    ParseRetryBudgetExhausted,
                    ApiBudgetExhausted,
                    ConsecutiveSaturatedRetriesExceeded,
                ) as exc:
                    # Persist the terminal retry burst into a crash step
                    # before re-raising.  Without this, the retries that
                    # actually killed the run live only in output.log and
                    # the replay JSON ends on the last SUCCESSFUL step.
                    if isinstance(exc, ParseRetryBudgetExhausted):
                        crash_reason = "parse_retry_budget_exhausted"
                    elif isinstance(exc, ConsecutiveSaturatedRetriesExceeded):
                        crash_reason = "consecutive_saturated_retries_exceeded"
                    else:
                        crash_reason = "api_budget_exhausted"
                    # Use empty-string / empty-dict placeholders (not None)
                    # so the crash row survives export_replay.write_replay_file,
                    # which assumes `response` is a dict and `action` is a
                    # string.
                    crash_step = {
                        "step": self.step_num,
                        "level": lvl,
                        "attempt": self.attempt,
                        "state_index": len(self._replay_states) - 1,
                        "action": "",
                        "action_log": "",
                        "formatted_obs": formatted_obs,
                        "raw_response": "",
                        "rationale": "",
                        "response": {},
                        "state_summary": "",
                        "reward": 0,
                        "won": False,
                        "lose": False,
                        "timeout": False,
                        "realworld_ts": realworld_ts,
                        "conversation_length": self.harness.conversation_length,
                        "parse_retries": exc.attempts,
                        "crash_reason": crash_reason,
                        "cumulative_cost_at_crash": self.total_cost,
                    }
                    self.log_data["steps"].append(crash_step)
                    # Force a final replay save -- the next scheduled
                    # incremental save never fires because we re-raise.
                    self._save_log_incremental(force=True)
                    raise

            # Parse action
            action_str = response_dict["action"]

            # The state the LLM just decided from is the last one captured
            # (either the initial reset state or the final idle frame of
            # the previous step).  Record its index BEFORE taking any engine
            # tick so the step record's ``state_index`` points to the
            # pre-action observation, mirroring human-replay's
            # ``frame_idx - 1`` convention at ``run_replay.py:614`` /
            # ``replay_agent.py:255``.
            pre_action_state_index = len(self._replay_states) - 1

            # Handle reset action -- skip env.step, treat as voluntary death.
            # No engine tick occurs, so no new state snapshot is appended;
            # the pre-action state is the only state associated with this
            # step.
            if action_str == "reset":
                reward = 0
                won = False
                died = True
                lose = True
                events = []
                is_over = True
                action_log = f"[{self.step_num}] RESET -> level restarted [LOSE]"
            else:
                action_idx = self.response_parser.action_to_index(action_str)

                # Execute the action frame.  Capture a state snapshot
                # immediately so the replay viewer can show the action's
                # direct effect before any idle-frame NPC motion mixes in.
                state_img, reward, is_over, truncated, info = self.env.unwrapped.step(
                    action_idx
                )
                action_frame_events = list(info["events_triggered"])
                action_frame_log = self.event_logger.log_action(
                    step_num=self.step_num,
                    action=action_idx,
                    prev_obs=prev_obs,
                    curr_obs=info["state"],
                    events=action_frame_events,
                    reward=int(reward),
                    won=self.env.unwrapped.game.won,
                    lose=self.env.unwrapped.game.lose,
                    prev_score=prev_score,
                    curr_score=prev_score + int(reward),
                    level=lvl,
                    attempt=self.attempt,
                )
                self._append_state(lvl, action_log=action_frame_log)

                # Inject idle frames (NO_OPs) to match human reaction time.
                # Each idle frame advances NPCs and game clock without player
                # input, and is captured as its own viewer state with a
                # per-frame action log.  Matches run_replay.py's per-frame
                # stamping so the two replay sources share one UX.
                idle_frames = self.cfg.game.idle_frames
                noop_idx = 4  # NO_OP action index
                idle_reward = 0
                idle_events = []
                idle_over = False
                pre_idle_obs = info["state"]
                running_score = prev_score + int(reward)
                for _ in range(idle_frames):
                    _, r, idle_over, _, idle_info = self.env.unwrapped.step(noop_idx)
                    idle_frame_events = list(idle_info["events_triggered"])
                    idle_frame_log = self.event_logger.log_action(
                        step_num=self.step_num,
                        action=noop_idx,
                        prev_obs=pre_idle_obs,
                        curr_obs=idle_info["state"],
                        events=idle_frame_events,
                        reward=int(r),
                        won=self.env.unwrapped.game.won,
                        lose=self.env.unwrapped.game.lose,
                        prev_score=running_score,
                        curr_score=running_score + int(r),
                        level=lvl,
                        attempt=self.attempt,
                    )
                    self._append_state(lvl, action_log=idle_frame_log)
                    idle_reward += int(r)
                    idle_events.extend(idle_frame_events)
                    running_score += int(r)
                    pre_idle_obs = idle_info["state"]
                    if idle_over:
                        break
                reward = int(reward) + idle_reward
                info["events_triggered"] = action_frame_events + idle_events

                # Check win/lose (after idle frames -- NPC may have killed avatar)
                self.env.unwrapped.game._check_terminations()
                won = self.env.unwrapped.game.won
                lose = self.env.unwrapped.game.lose
                died = lose
                is_over = is_over or idle_over

                # Get events (already merged above)
                events = info["events_triggered"]

                # Track interactions for coverage metrics
                interactions_before = len(self.interaction_tracker.all_instances)
                self.interaction_tracker.record_events(events, prev_obs, self.step_num)

                # Log new interactions to W&B table
                if self._wandb_run and self._interaction_table is not None:
                    self._log_interactions_to_table(lvl, interactions_before)

                # Track resource colors from collect events
                new_resources = self._get_avatar_resources(info["state"]) or {}
                self._update_resource_colors(
                    resources or {}, new_resources, events, prev_obs
                )

                # Update score
                self.total_score += int(reward)

                # Real VGDL engine events -> single action-log string used
                # by both the viewer panel and the LLM's LAST K ACTIONS.
                action_log = self.event_logger.log_action(
                    step_num=self.step_num,
                    action=action_idx,
                    prev_obs=prev_obs,
                    curr_obs=info["state"],
                    events=events,
                    reward=reward,
                    won=won,
                    lose=lose,
                    prev_score=prev_score,
                    curr_score=self.total_score,
                    level=lvl,
                    attempt=self.attempt,
                )

            step_log = {
                "step": self.step_num,
                "level": lvl,
                "attempt": self.attempt,
                # Pre-action state (what the LLM saw when choosing this
                # action).  Matches human-replay's convention so the
                # viewer canvas shows the decision input, not the
                # post-action result.  The action's direct effect is
                # available at ``state_index + 1`` and any idle-frame
                # NPC motion at ``state_index + 2 .. + idle_frames + 1``.
                "state_index": pre_action_state_index,
                "action": action_str,
                "action_log": action_log,
                "formatted_obs": formatted_obs,
                "raw_response": raw_response,
                "rationale": response_dict.get("rationale", ""),
                "response": response_dict,
                "state_summary": f"Avatar at {self._get_avatar_pos(info['state'] if action_str != 'reset' else prev_obs)}, Score: {self.total_score}",
                "reward": reward,
                "won": won,
                "lose": lose,
                # Generative mode plays through the VGDL engine directly;
                # the engine resolves each terminal as won or lost and
                # does not produce level-time-budget timeouts.  The flag
                # is emitted explicitly so the replay schema matches the
                # human-replay pipeline.
                "timeout": False,
                "realworld_ts": realworld_ts,
                "conversation_length": self.harness.conversation_length,
                # Per-step token / cost totals.  These are the sums
                # across final accepted call + all failed parse-retry
                # attempts (see agent._generate_with_parse_retry), not
                # the final-attempt-only values.  ``num_llm_calls``
                # encodes how many calls each step cost.
                "input_tokens": (context_info or {}).get("input_tokens"),
                "output_tokens": (context_info or {}).get("output_tokens"),
                "reasoning_tokens": (context_info or {}).get("reasoning_tokens"),
                "call_cost": (context_info or {}).get("call_cost"),
                "num_llm_calls": (context_info or {}).get("num_llm_calls", 1),
                # Per-attempt log of parse failures that got repaired via
                # conversational retry at this step; empty on first-shot
                # success.  See agent._generate_with_parse_retry.
                "parse_retries": parse_retry_log,
            }
            self.log_data["steps"].append(step_log)

            # Log to W&B
            if self._wandb_run:
                import wandb

                wandb_log = {
                    "step_num": self.step_num,
                    "frame_num": self.step_num * (1 + self.cfg.game.idle_frames),
                    "level": lvl,
                    "attempt": self.attempt,
                    "reward": reward,
                    "cumulative_score": self.total_score,
                    "cumulative_wins": self.cumulative_wins,
                    "cumulative_losses": self.cumulative_losses,
                    "cumulative_success": self.cumulative_success,
                    "won": won,
                    "lose": lose,
                    "interactions_discovered": self.interaction_tracker.interactions_discovered,
                    "interaction_coverage": self.interaction_tracker.coverage,
                    "num_conversation_messages": self.harness.conversation_length,
                }
                wandb_log["cumulative_input_tokens"] = self.total_input_tokens
                wandb_log["cumulative_output_tokens"] = self.total_output_tokens
                wandb_log["cumulative_reasoning_tokens"] = self.total_reasoning_tokens
                wandb_log["cumulative_cost"] = self.total_cost
                # LLM cost metrics + provider error tracking.  The token
                # / cost fields on context_info are the per-STEP totals
                # (summed over final attempt + failed parse-retries) as
                # set by _generate_with_parse_retry.
                if context_info:
                    wandb_log["input_tokens"] = context_info["input_tokens"]
                    wandb_log["output_tokens"] = context_info["output_tokens"]
                    wandb_log["reasoning_tokens"] = context_info["reasoning_tokens"]
                    wandb_log["call_cost"] = context_info.get("call_cost", 0.0)
                    wandb_log["num_llm_calls"] = context_info["num_llm_calls"]
                    wandb_log["input_chars"] = context_info["input_chars"]
                    wandb_log["output_chars"] = context_info["output_chars"]
                    wandb_log["chars_per_token"] = context_info["chars_per_token"]
                    wandb_log["pct_capacity"] = context_info["pct_capacity"]
                    reasoning = context_info.get("reasoning")
                    wandb_log["reasoning_chars"] = len(reasoning) if reasoning else 0
                    retries = context_info["retries"]
                    wandb_log["provider_retries"] = retries
                    wandb_log["provider_retry_wait_s"] = context_info[
                        "retry_total_wait"
                    ]
                    if retries and self._provider_error_table is not None:
                        for err in context_info["retry_errors"]:
                            self._provider_error_table.add_data(
                                self.step_num,
                                lvl,
                                self.attempt,
                                err["attempt"],
                                err["code"],
                                err["reason"],
                            )
                wandb.log(wandb_log)

            # Print progress
            self._print_progress(action_log, response_dict, context_info)

            # Save incrementally
            self._save_log_incremental()

            # Update for next iteration
            self.step_num += 1
            prev_obs = info["state"] if action_str != "reset" else prev_obs
            prev_score = self.total_score

            if is_over or won or lose:
                break

        return won, died

    def _generate_with_parse_retry(
        self, messages: list[dict]
    ) -> tuple[dict, str, dict, list[dict]]:
        """Generate a response and repair parse failures via conversational retry.

        The flow:
          1. Call ``self.llm.generate(messages)``.
          2. Try to parse via ``harness.parse_strict``.
          3. On parse failure: append the malformed assistant + a mode-specific
             repair-prompt user message (``harness.append_repair_turn``),
             decrement the run-level retry budget, and regenerate.
          4. On success: clip the appended repair turns so the saved
             conversation contains only ``(user_obs -> valid_assistant)``,
             then commit the assistant message via
             ``harness.commit_assistant``.
          5. If the retry budget is exhausted, raise ``ValueError``.

        Accumulates token usage and cost across all attempts (including
        failed ones -- they cost real money).

        Before returning, ``context_info``'s per-call token / cost
        fields (``input_tokens``, ``output_tokens``,
        ``reasoning_tokens``, ``call_cost``) are OVERWRITTEN with the
        sums across every LLM call this step made -- final accepted
        attempt + all failed parse-retry attempts.  Downstream per-step
        wandb metrics therefore reflect what the step actually consumed
        on the provider, not just the one call that happened to parse.
        The number of LLM calls is surfaced separately as
        ``num_llm_calls``.

        Returns:
            Tuple of (parsed_dict, final_raw_response, final_context_info,
            per_attempt_log). The per-attempt log is a list of
            ``{"raw": ..., "error": ...}`` dicts suitable for attaching to
            the step record; empty on first-shot success.
        """
        # Index where repair turns will start (used for clipping on success).
        # Captured BEFORE the first generate call, i.e. the point right after
        # the user observation was appended by harness.build_messages.
        repair_start_idx = self.harness.conversation_length

        attempts: list[dict] = []
        step_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "call_cost": 0.0,
        }
        # Per-step counter for the consecutive-saturated-retries guard.
        # Reset at the top of each step (each ``_generate_with_parse_retry``
        # invocation) so a well-behaved next step is not punished for a
        # previous step's saturation burst.
        consecutive_saturated = 0
        sat_cap = self.cfg.llm.max_consecutive_saturated_retries

        raw_response, context_info = self.llm.generate(messages)
        self._accumulate_token_usage(context_info)
        self._accumulate_step_usage(step_usage, context_info)
        self._check_api_budget(attempts)
        consecutive_saturated = self._update_saturation_counter(
            consecutive_saturated, context_info, attempts, sat_cap
        )

        while True:
            try:
                parsed = self.harness.parse_strict(raw_response, context_info)
            except (ValueError, json.JSONDecodeError) as e:
                # Record the failed attempt with its hidden reasoning trace
                # BEFORE checking the budget, so the terminal attempt shows
                # up in the step_log (the caller catches the exhaustion
                # exception and persists ``exc.attempts`` into the step).
                attempts.append(
                    {
                        "raw": raw_response,
                        "error": str(e),
                        "hidden_reasoning_failed": context_info.get("reasoning"),
                    }
                )
                if self._parse_retry_remaining <= 0:
                    raise ParseRetryBudgetExhausted(
                        f"Parse retry budget exhausted "
                        f"(llm.parse_retry_budget={self.cfg.llm.parse_retry_budget}). "
                        f"Last error: {e}. "
                        f"Last raw response: {raw_response!r}",
                        attempts=attempts,
                    ) from e
                self._parse_retry_remaining -= 1
                # Keep only the most recent failed attempt in the retry
                # prompt: clip back to ``repair_start_idx`` before appending
                # the new repair turn.  Without this, each retry would
                # concatenate another malformed response + another hidden
                # reasoning echo, blowing the context window after a few
                # copied-reasoning retries.  A no-op on the first retry
                # (nothing past repair_start_idx yet).
                self.harness.clip_from(repair_start_idx)
                self.harness.append_repair_turn(raw_response, e, context_info)
                retry_messages = list(self.harness._conversation)
                raw_response, context_info = self.llm.generate(retry_messages)
                self._accumulate_token_usage(context_info)
                self._accumulate_step_usage(step_usage, context_info)
                self._check_api_budget(attempts)
                consecutive_saturated = self._update_saturation_counter(
                    consecutive_saturated, context_info, attempts, sat_cap
                )
                continue
            else:
                break

        # Success: clip any repair turns so the in-memory conversation
        # looks like the model responded cleanly to the original observation.
        if attempts:
            self.harness.clip_from(repair_start_idx)

        self.harness.commit_assistant(parsed, raw_response, context_info)

        # Overwrite per-call token / cost fields with step totals so
        # every downstream reader (wandb logger, replay step_log, etc.)
        # sees "what this step actually consumed" across the final
        # accepted call plus any failed parse-retry attempts.
        context_info["input_tokens"] = step_usage["input_tokens"]
        context_info["output_tokens"] = step_usage["output_tokens"]
        context_info["reasoning_tokens"] = step_usage["reasoning_tokens"]
        context_info["call_cost"] = step_usage["call_cost"]
        context_info["num_llm_calls"] = 1 + len(attempts)

        self._last_context_info = context_info
        return parsed, raw_response, context_info, attempts

    def _accumulate_token_usage(self, context_info: dict) -> None:
        """Accumulate run-level token counts and cost from a single LLM call.

        Called on every call, including failed-parse attempts, because the
        provider bills for those too.
        """
        self.total_input_tokens += context_info["input_tokens"]
        self.total_output_tokens += context_info["output_tokens"]
        self.total_reasoning_tokens += context_info["reasoning_tokens"]
        self.total_cost += context_info.get("call_cost", 0.0)

    @staticmethod
    def _accumulate_step_usage(step_usage: dict, context_info: dict) -> None:
        """Accumulate per-step token / cost sums across all LLM calls for one step.

        Mirrors ``_accumulate_token_usage`` but writes into a step-scoped
        dict so the wandb logger can report "tokens this step actually
        consumed" rather than "tokens on the one accepted call".
        """
        step_usage["input_tokens"] += context_info["input_tokens"]
        step_usage["output_tokens"] += context_info["output_tokens"]
        step_usage["reasoning_tokens"] += context_info["reasoning_tokens"]
        step_usage["call_cost"] += context_info.get("call_cost", 0.0)

    def _check_api_budget(self, attempts: list[dict]) -> None:
        """Raise ApiBudgetExhausted if cumulative cost crossed the cap.

        Called immediately after every ``_accumulate_token_usage`` inside
        the parse-retry loop.  ``attempts`` is forwarded into the exception
        so the caller can persist in-progress retries into the crash step.

        ``llm.api_budget <= 0.0`` disables the check (default).
        """
        budget = self.cfg.llm.api_budget
        if budget <= 0.0:
            return
        if self.total_cost >= budget:
            raise ApiBudgetExhausted(
                f"API budget exhausted "
                f"(llm.api_budget=${budget:.4f}, "
                f"cumulative cost=${self.total_cost:.4f}).",
                total_cost=self.total_cost,
                api_budget=budget,
                attempts=list(attempts),
            )

    def _update_saturation_counter(
        self,
        current: int,
        context_info: dict,
        attempts: list[dict],
        cap: int,
    ) -> int:
        """Increment / reset the consecutive-saturated counter after an LLM call.

        A call is "saturated" when ``output_tokens >= max_tokens`` in
        ``context_info`` (OpenRouter's ``output_tokens`` includes reasoning
        tokens per the OpenAI spec, so this single check catches both visible
        truncation and hidden-reasoning-ceiling saturation).  On a saturated
        call, increments the counter; on a non-saturated call, resets to 0.
        Raises ``ConsecutiveSaturatedRetriesExceeded`` when the counter
        reaches ``cap``.

        ``cap <= 0`` disables the check (returns the current counter unchanged).
        """
        if cap <= 0:
            return current
        out_tokens = context_info.get("output_tokens") or 0
        max_tokens = context_info.get("max_tokens")
        if max_tokens is None or out_tokens < max_tokens:
            return 0
        new_count = current + 1
        if new_count >= cap:
            raise ConsecutiveSaturatedRetriesExceeded(
                f"Consecutive saturated retries exceeded "
                f"(llm.max_consecutive_saturated_retries={cap}).  "
                f"{new_count} calls in a row hit output_tokens >= "
                f"max_tokens={max_tokens} within the same step, "
                f"indicating a saturate-retry-saturate loop.",
                attempts=list(attempts),
            )
        return new_count

    @staticmethod
    def _get_avatar_pos(obs: list) -> tuple | None:
        """Get avatar (x, y) position from observation."""
        if obs is None:
            return None
        for x, column in enumerate(obs):
            for y, cell in enumerate(column):
                for obj in cell:
                    if obj["name"] == "avatar":
                        return (x, y)
        return None

    def _get_avatar_resources(self, obs: list) -> dict | None:
        """Get avatar resources from observation."""
        if obs is None:
            return None
        for column in obs:
            for cell in column:
                for obj in cell:
                    if obj["name"] == "avatar":
                        return obj.get("resources", {})
        return None

    def _update_resource_colors(
        self,
        prev_resources: dict,
        new_resources: dict,
        events: list,
        prev_obs: list,
    ) -> None:
        """Track resource colors when objects are collected.

        When a resource count increases and there's a killSprite event where
        the avatar killed something, we associate that object's color with
        the resource.
        """
        # Find resources that increased
        for res_name, new_count in new_resources.items():
            old_count = prev_resources.get(res_name, 0)
            if new_count > old_count and res_name not in self.resource_colors:
                # Resource increased - find the collected object's color
                for event in events:
                    effect_name = event[0]
                    if effect_name == "killSprite":
                        sprite1 = event[1]  # (name, obj_id) - killed sprite
                        sprite2 = event[2]  # (name, obj_id) - killer
                        name1, obj_id1 = sprite1
                        name2, obj_id2 = sprite2
                        # Check if avatar killed something (collect event)
                        if name2 == "avatar":
                            color = self._get_object_color_by_id(prev_obs, obj_id1)
                            if color:
                                self.resource_colors[res_name] = color
                                break

    def _get_object_color_by_id(self, obs: list, obj_id: str) -> str | None:
        """Get color name for an object by ID."""
        if obs is None:
            return None
        for column in obs:
            for cell in column:
                for obj in cell:
                    if obj["obj_id"] == obj_id:
                        color = obj.get("color")
                        if color is None:
                            return None
                        if isinstance(color, str):
                            return color
                        # Convert RGB tuple to color name
                        from src.utils import COLOR_DICT

                        rgb_to_color = {v: k for k, v in COLOR_DICT.items()}
                        rgb_tuple = tuple(color) if isinstance(color, list) else color
                        return rgb_to_color.get(rgb_tuple)
        return None

    def _print_progress(
        self, action_log: str, response: dict, context_info: dict | None
    ) -> None:
        """Print step progress to console."""
        print(f"\n[Step {self.step_num}]")
        if self.cfg.logging.verbose:
            if response.get("rationale"):
                print(f"  Rationale: {response['rationale'][:200]}...")
            if context_info and context_info.get("reasoning"):
                hidden = context_info["reasoning"]
                print(f"  Hidden reasoning: {hidden[:200]}...")
        print(f"  Action: {response['action']}")
        print(f"  Result: {action_log}")
        print(f"  Conversation: {self.harness.conversation_length} messages")

    def _log_interactions_to_table(self, level: int, start_idx: int) -> None:
        """Log new interactions to W&B table.

        Args:
            level: Current level number
            start_idx: Index in all_instances where new interactions start
        """
        if self._interaction_table is None:
            return

        # Add all new interaction instances to the table
        for idx, (step, interaction, obj_ids) in enumerate(
            self.interaction_tracker.all_instances[start_idx:], start=start_idx
        ):
            color1, effect, color2 = interaction
            obj_id1, obj_id2 = obj_ids

            # Check if this is the first occurrence of this interaction type
            # Look at all previous instances to see if this interaction type appeared before
            first_occurrence = interaction not in [
                inter for _, inter, _ in self.interaction_tracker.all_instances[:idx]
            ]

            self._interaction_table.add_data(
                step,
                level,
                self.attempt,
                color1,
                effect,
                color2,
                obj_id1,
                obj_id2,
                first_occurrence,
            )

    def _save_log_incremental(self, force: bool = False) -> None:
        """Save current log_data as .replay.json.gz (overwrites) and sync to W&B + S3.

        Args:
            force: If True, save regardless of interval. Used for final save.
        """
        if (
            not force
            and (self.step_num - self._last_replay_save_step)
            < self._replay_save_interval
        ):
            return

        write_replay_file(self.log_data, self._replay_states, self._log_filepath)
        self._last_replay_save_step = self.step_num

        # Sync to S3
        if self._s3_client and self._s3_bucket and self._s3_key:
            self._s3_client.upload_file(
                self._log_filepath, self._s3_bucket, self._s3_key
            )

        # Periodically save to W&B
        if self._wandb_run and (
            self.step_num - self._last_wandb_save_step >= self._wandb_save_interval
        ):
            self._save_to_wandb()
            self._last_wandb_save_step = self.step_num

    def _save_to_wandb(self) -> None:
        """Save replay file to W&B."""
        if not self._wandb_run or not self._log_filepath:
            return

        import wandb

        # Save the replay file to the run
        wandb.save(self._log_filepath, policy="live")

    def _save_log_final(self) -> None:
        """Save final log as .replay.json.gz and upload to W&B."""
        self._save_log_incremental(force=True)

        # Final W&B save
        if self._wandb_run and self._log_filepath:
            import wandb

            wandb.save(self._log_filepath, policy="end")

        print(f"\nLog saved to {self.cfg.logging.output_dir}")


def run_game(
    game_name: str,
    model_path: str | None = None,
    start_level: int = 0,
    max_levels: int = 9,
    output_dir: str = "out/",
    verbose: bool = False,
    wandb_project: str | None = None,
    seed: int = 0,
    backend: str = "openrouter",
    openrouter_max_tokens: int = 4096,
    temperature: float = 0.5,
    top_p: float = 0.95,
    rationale_mode: str = "prompted-rationale",
    suggestion_level: str = "elaborate",
    advancement: dict | None = None,
) -> dict:
    """Convenience function to run a game evaluation from individual args.

    Kept for W&B sweeps and legacy CLI scripts; the preferred entry point
    is ``src.llm_eval.generative_gameplay.run`` (Hydra).

    Args:
        game_name: Name of the game
        model_path: OpenRouter model ID (None -> mock backend)
        start_level: Level to start from
        max_levels: How many levels to play (0=all)
        output_dir: Directory for output logs
        verbose: If True, print full responses without truncation
        wandb_project: W&B project name (disables W&B if None)
        seed: Random seed
        backend: 'openrouter' or 'mock'
        openrouter_max_tokens: Max tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        rationale_mode: 'action-only' | 'prompted-rationale' | 'copied-reasoning'
        suggestion_level: 'elaborate' | 'minimal'
        advancement: Advancement strategy dict

    Returns:
        Results dict with outcome, final_level, total_steps
    """

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    cfg = Config(
        harness=HarnessConfig(
            rationale_mode=rationale_mode,
            suggestion_level=suggestion_level,
        ),
        llm=LLMConfig(
            backend="mock" if model_path is None else backend,
            model=model_path or "",
            temperature=temperature,
            top_p=top_p,
            max_tokens=openrouter_max_tokens,
        ),
        game=GameConfig(
            game=game_name,
            start_level=start_level,
            max_levels=max_levels,
            advancement=advancement
            or {
                "strategy": "blocked_curricula",
                "total_frame_budget": 2700,
                "consecutive_wins_required": 2,
            },
        ),
        logging=LoggingConfig(
            output_dir=output_dir,
            wandb_project=wandb_project or "",
            verbose=verbose,
        ),
        seed=seed,
    )

    validate_config(cfg)
    agent = GameplayAgent(cfg)
    return agent.run(start_level=start_level)
