# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Structured configuration schema for the VGDL LLM evaluation harness.

Uses Hydra with OmegaConf dataclass-backed structured configs.
All fields are flat -- override on CLI via dot notation, e.g.:
  harness.rationale_mode=prompted-rationale harness.suggestion_level=minimal
"""

from dataclasses import dataclass, field
from typing import Union


# -- Advancement strategy dataclasses -----------------------------------------
# These are NOT used as OmegaConf structured config nodes (OmegaConf doesn't
# support Union types).  GameConfig.advancement is a plain dict that gets
# parsed into the correct typed dataclass by parse_advancement(), called from
# validate_config().


@dataclass
class FixedBudgetAdvancement:
    strategy: str = "fixed_budget"
    level_frame_budget: int = 1200  # per-LEVEL frame limit


@dataclass
class BlockedCurriculaAdvancement:
    strategy: str = "blocked_curricula"
    total_frame_budget: int = 2700  # GLOBAL frame limit across all levels
    consecutive_wins_required: int = 2
    # Abort the run after this many consecutive deaths without an intervening
    # win. 0 disables the early stop (default).
    max_consecutive_losses: int = 0


@dataclass
class HarnessConfig:
    # 'action-only' | 'prompted-rationale' | 'copied-reasoning'
    rationale_mode: str = "prompted-rationale"
    suggestion_level: str = "elaborate"  # 'elaborate' | 'minimal' | 'oracle'
    # Fraction of the context remaining after the generation budget that the
    # input prompt may occupy before the oldest message pairs are dropped.
    context_usage_fraction: float = 0.5


@dataclass
class LLMConfig:
    backend: str = "openrouter"  # 'openrouter' | 'mock'
    model: str = ""  # openrouter model ID
    temperature: float = 0.5
    top_p: float = 0.95
    max_tokens: int = 4096  # generation budget
    run_token_budget: int = 5_000_000  # hard per-run cumulative token limit
    # Hard per-run cumulative cost cap in USD.  Checked after every LLM
    # call (including parse-retry attempts).  If cumulative_cost >= api_budget
    # the run raises ApiBudgetExhausted, writes a crash step, and terminates.
    # 0.0 disables (default) -- costs are uncapped and only bounded by
    # run_token_budget and total_frame_budget.  Use this as a safety net for
    # sweeps on expensive models where a single runaway run could blow past
    # the planned per-run forecast.
    api_budget: float = 0.0
    # Max number of parse-failure retries per run.  On each failure, the agent
    # feeds the error back to the model in a retry turn and requests another
    # response. On success the retry turns are clipped. 0 disables
    # retries (first parse failure aborts the run).
    parse_retry_budget: int = 10
    # Abort if this many consecutive attempts exhaust max_tokens within one
    # decision. A completed response resets the counter; 0 disables the guard.
    max_consecutive_saturated_retries: int = 3


@dataclass
class GameConfig:
    game: str = "bait_vgfmri4"
    start_level: int = 0
    resume_from: str | None = None
    max_levels: int = 9  # how many levels to play (0=all)
    idle_frames: int = 4  # NO_OP frames injected after each LLM step
    advancement: dict = field(
        default_factory=lambda: {
            "strategy": "blocked_curricula",
            "total_frame_budget": 2700,
            "consecutive_wins_required": 2,
            "max_consecutive_losses": 0,
        }
    )


@dataclass
class LoggingConfig:
    output_dir: str = "out/"
    wandb_project: str = ""  # empty string = disabled
    verbose: bool = False


@dataclass
class ReplayConfig:
    data_dir: str = ""
    subject: str = ""  # empty = all subjects
    game_filter: str = ""  # empty = all games
    action_frames_only: bool = True
    stride: int = 1


@dataclass
class ExtractionConfig:
    """Config for the sliding-window feature extraction entry point.

    Flat by design -- ``agents.lrm.features.extract`` is a
    standalone Hydra app with its own ``agents/lrm/configs/extract_features/default.yaml``
    and does not share the gameplay ``Config`` graph.
    """

    # Path to a single ``.replay.json.gz`` input file.
    prompts: str = "???"
    # HuggingFace model ID (e.g. ``Qwen/Qwen3.5-9B``).
    model: str = "???"
    # Prefer an immutable HF commit. None uses the model repository default revision.
    model_revision: str | None = None
    output_dir: str = "out/sliding_window_features"
    torch_dtype: str = "bfloat16"
    # Fraction of the model's native max context to use per window.
    # Generative gameplay truncates at ``harness.context_usage_fraction`` --
    # keep these in sync manually; they are not formally linked.
    window_fraction: float = 0.3
    # Overlap ratio between consecutive windows.  0.5 = 50% overlap.
    # With window_fraction=0.3 and overlap=0.5, stride = 15% of max context.
    overlap: float = 0.5
    # Disable torch.compile.  Default True because feature extraction uses
    # variable-length windows and re-compilation on every new shape adds
    # minutes of overhead without a steady-state payoff.
    no_compile: bool = True
    wandb_project: str = ""
    verbose: bool = False
    # When True, collapse runs of identical consecutive actions to their
    # first occurrence before building the conversation (drops ~72% of
    # turns on human replays, ~3.6x token reduction).  Kept as a comparison
    # option against the default all-decision-points extraction.  Note:
    # compressed features have non-uniform time gaps between kept turns;
    # neural alignment must account for this at analysis time.
    action_compression: bool = False
    # When True, instantiate the model architecture with random weights
    # (``from_config``) instead of loading pretrained checkpoints.  The
    # tokenizer is still loaded from the pretrained model.  Output paths
    # get an ``__ABLATION__random-init`` suffix on the model_id.
    random_init: bool = False
    # Labels output directories for explicit experimental conditions.
    ablation_tag: str = ""
    # Continue selected sessions after an error, then exit nonzero for the batch.
    continue_on_session_error: bool = False


@dataclass
class Config:
    seed: int = 0  # global RNG seed (game engine, LLM API, numpy, random)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    game: GameConfig = field(default_factory=GameConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


# -- Rationale mode <-> reasoning effort mapping -------------------------------

VALID_RATIONALE_MODE = ("action-only", "prompted-rationale", "copied-reasoning")

_RATIONALE_MODE_TO_EFFORT = {
    "action-only": "none",
    "prompted-rationale": "none",
    "copied-reasoning": "medium",
}


def rationale_mode_to_effort(mode: str) -> str:
    """Map rationale_mode to the OpenRouter reasoning effort flag.

    copied-reasoning uses 'medium' rather than 'high' so the prompt-level
    "think only when plans change" steering can still influence the model.
    """
    if mode not in _RATIONALE_MODE_TO_EFFORT:
        raise ValueError(
            f"Invalid rationale_mode={mode!r}. Must be one of {VALID_RATIONALE_MODE}."
        )
    return _RATIONALE_MODE_TO_EFFORT[mode]


def rationale_mode_carries_rationale(mode: str) -> bool:
    """Whether a mode keeps a 'rationale' field in assistant messages.

    Only 'action-only' omits it.
    """
    if mode not in VALID_RATIONALE_MODE:
        raise ValueError(
            f"Invalid rationale_mode={mode!r}. Must be one of {VALID_RATIONALE_MODE}."
        )
    return mode != "action-only"


# -- Advancement parsing -------------------------------------------------------

_VALID_ADVANCEMENT_STRATEGY = ("fixed_budget", "blocked_curricula")

_FIXED_BUDGET_KEYS = {"strategy", "level_frame_budget"}
_BLOCKED_CURRICULA_KEYS = {
    "strategy",
    "total_frame_budget",
    "consecutive_wins_required",
    "max_consecutive_losses",
}


def parse_advancement(
    raw: dict,
) -> Union[FixedBudgetAdvancement, BlockedCurriculaAdvancement]:
    """Parse an advancement dict into the correct typed dataclass.

    Raises ValueError for unknown strategy, missing fields, or extra fields.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"game.advancement must be a dict, got {type(raw).__name__}")

    strategy = raw.get("strategy")
    if strategy not in _VALID_ADVANCEMENT_STRATEGY:
        raise ValueError(
            f"game.advancement.strategy={strategy!r} is invalid. "
            f"Must be one of {_VALID_ADVANCEMENT_STRATEGY}"
        )

    if strategy == "fixed_budget":
        extra = set(raw.keys()) - _FIXED_BUDGET_KEYS
        if extra:
            raise ValueError(
                f"game.advancement has extra keys for strategy='fixed_budget': {extra}. "
                f"Allowed keys: {_FIXED_BUDGET_KEYS}"
            )
        fb = raw.get("level_frame_budget", 1200)
        if fb <= 0:
            raise ValueError(f"game.advancement.level_frame_budget={fb} must be > 0")
        return FixedBudgetAdvancement(level_frame_budget=fb)

    # blocked_curricula
    extra = set(raw.keys()) - _BLOCKED_CURRICULA_KEYS
    if extra:
        raise ValueError(
            f"game.advancement has extra keys for strategy='blocked_curricula': {extra}. "
            f"Allowed keys: {_BLOCKED_CURRICULA_KEYS}"
        )
    tb = raw.get("total_frame_budget", 2700)
    cw = raw.get("consecutive_wins_required", 2)
    mcl = raw.get("max_consecutive_losses", 0)
    if tb <= 0:
        raise ValueError(f"game.advancement.total_frame_budget={tb} must be > 0")
    if cw < 1:
        raise ValueError(
            f"game.advancement.consecutive_wins_required={cw} must be >= 1"
        )
    if mcl < 0:
        raise ValueError(
            f"game.advancement.max_consecutive_losses={mcl} must be >= 0 "
            f"(0 disables the early stop)"
        )
    return BlockedCurriculaAdvancement(
        total_frame_budget=tb,
        consecutive_wins_required=cw,
        max_consecutive_losses=mcl,
    )


# -- Validation ----------------------------------------------------------------

_VALID_BACKEND = ("openrouter", "mock")
_VALID_SUGGESTION_LEVEL = ("elaborate", "minimal", "oracle")


def validate_config(cfg: Config) -> None:
    """Validate configuration for invalid combinations.

    Raises ValueError with a descriptive message for every violation.
    """
    h = cfg.harness
    lc = cfg.llm

    # Parse and replace advancement dict with typed dataclass
    cfg.game.advancement = parse_advancement(cfg.game.advancement)

    if h.rationale_mode not in VALID_RATIONALE_MODE:
        raise ValueError(
            f"harness.rationale_mode={h.rationale_mode!r} is invalid. "
            f"Must be one of {VALID_RATIONALE_MODE}"
        )

    if h.suggestion_level not in _VALID_SUGGESTION_LEVEL:
        raise ValueError(
            f"harness.suggestion_level={h.suggestion_level!r} is invalid. "
            f"Must be one of {_VALID_SUGGESTION_LEVEL}"
        )

    if not (0.0 < h.context_usage_fraction <= 1.0):
        raise ValueError(
            f"harness.context_usage_fraction={h.context_usage_fraction} is "
            f"invalid. Must be in (0, 1]."
        )

    if lc.backend not in _VALID_BACKEND:
        raise ValueError(
            f"llm.backend={lc.backend!r} is invalid. Must be one of {_VALID_BACKEND}"
        )

    if h.rationale_mode == "copied-reasoning" and lc.backend not in (
        "openrouter",
        "mock",
    ):
        raise ValueError(
            f"harness.rationale_mode='copied-reasoning' requires llm.backend='openrouter' "
            f"(hidden reasoning traces are only exposed via the OpenRouter API) "
            f"or 'mock' for testing. Got backend={lc.backend!r}."
        )

    if lc.run_token_budget <= 0:
        raise ValueError(f"llm.run_token_budget={lc.run_token_budget} must be positive")

    if lc.parse_retry_budget < 0:
        raise ValueError(
            f"llm.parse_retry_budget={lc.parse_retry_budget} must be >= 0 "
            f"(0 disables retries)"
        )

    if lc.max_consecutive_saturated_retries < 0:
        raise ValueError(
            f"llm.max_consecutive_saturated_retries="
            f"{lc.max_consecutive_saturated_retries} must be >= 0 "
            f"(0 disables the check)"
        )
