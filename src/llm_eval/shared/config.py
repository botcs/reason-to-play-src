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
    # Fraction of the (context_length - max_tokens) budget that the input
    # prompt is allowed to occupy before ``Harness.truncate_if_needed``
    # drops the oldest message pairs.  0.5 is the slice-1..10 effective
    # regime (the pre-2026-04-22 char-based estimator fired truncation at
    # ~22-45% of real context because it over-counted natural-language
    # rationales; we kept the new honest-token-count trigger at the same
    # effective fill).  Research variable -- raise for sweeps that probe
    # high-context-fill degradation curves.
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
    # feeds the error back to the model in a repair turn and regenerates; on
    # success the repair turns are clipped from the conversation.  0 disables
    # retries (first parse failure aborts the run).
    parse_retry_budget: int = 10
    # Per-step guard against the "saturate-retry-saturate" loop first seen on
    # slice 11 lemmings (<RUN-ID>, 122b-a10b elaborate): a single step retried
    # ~10 times with every attempt hitting ``output_tokens >= max_tokens``,
    # burning ~10 x max_tokens of reasoning on one decision.  If N calls in a
    # row within the same step come back saturated (``output_tokens >=
    # max_tokens``), abort the run with
    # ``crash_reason="consecutive_saturated_retries_exceeded"``.  Counter is
    # reset per step and reset on any non-saturated call within a step.
    # 0 disables the check.
    max_consecutive_saturated_retries: int = 3


@dataclass
class GameConfig:
    game: str = "bait_vgfmri4"
    start_level: int = 0
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
    data_dir: str = "./workdir/prepare_behavioral_data"
    subject: str = ""  # empty = all subjects
    game_filter: str = ""  # empty = all games
    action_frames_only: bool = True
    stride: int = 1


@dataclass
class ExpDBConfig:
    enabled: bool = False  # set True to activate DynamoDB tracking
    region: str = "<REGION>"
    s3_bucket: str = "<ANONYMIZED-BUCKET>"
    s3_prefix: str = "llm_generative_gameplay/"
    replay_s3_prefix: str = "llm_replay/"
    # S3 prefix for sliding-window feature artifacts (.pt + prompts.jsonl.gz).
    # Mirrors the layout used by scripts/upload_features_to_s3.py so the new
    # in-process uploads land in the same S3 cohort as the manual ones.
    feature_s3_prefix: str = "extract_model_features_to_py/multi-turn/"
    # When extract_features sweeps a glob of sessions, controls whether a
    # hard error in one session aborts the whole batch (False, matches
    # gameplay/replay siblings) or marks that slot failed and continues
    # (True, default -- preserves the amortised model-load when one input
    # is bad).
    continue_on_session_error: bool = True
    # When True, reset existing slots to pending before claiming.  Allows
    # re-running an extraction that previously claimed the same slots.
    force_reclaim: bool = False
    # When True, resume a failed gameplay run from its last mastered level.
    # Downloads the prior replay from S3, restores conversation + counters,
    # and continues from where the run was killed.
    resume: bool = False


@dataclass
class ExtractionConfig:
    """Config for the sliding-window feature extraction entry point.

    Flat by design -- ``src.llm_eval.human_replay.extract_features`` is a
    standalone Hydra app with its own ``conf/extract_features/default.yaml``
    and does not share the gameplay ``Config`` graph.
    """

    # Path to a single ``.replay.json.gz`` input file.
    prompts: str = "???"
    # HuggingFace model ID (e.g. ``Qwen/Qwen3.5-9B``).
    model: str = "???"
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
    # Opaque ablation tag.  When non-empty, ExpDB routes to the ablation
    # table (``vgdl-grid-ablation``) with create-on-claim semantics, and
    # the tag is prepended to the slot_id.  Each ablation study defines
    # its own tag (e.g. ``random-init``, ``wf-0.7``).
    ablation_tag: str = ""
    # DynamoDB-backed run tracking + S3 upload.  When ``exp_db.enabled`` is
    # True, every session claims a slot in ``vgdl-grid-feature-extraction``,
    # uploads its .pt + prompts.jsonl.gz to S3, and writes the wandb
    # run id/url into the slot record.  Off by default; behavior is
    # otherwise unchanged from pre-2026-04-27.
    exp_db: ExpDBConfig = field(default_factory=ExpDBConfig)


@dataclass
class Config:
    seed: int = 0  # global RNG seed (game engine, LLM API, numpy, random)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    game: GameConfig = field(default_factory=GameConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    exp_db: ExpDBConfig = field(default_factory=ExpDBConfig)


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
