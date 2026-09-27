"""Deterministic slot ID generation for each experiment table.

Slot IDs are pipe-separated strings built from the config dimensions that
uniquely identify an experiment slot. The same config always produces the
same slot ID, enabling idempotent population and reliable claim/update.
"""

from src.llm_eval.shared.config import Config


def slot_id_gameplay(cfg: Config) -> str:
    """Build slot ID for the generative gameplay table."""
    h = cfg.harness
    return "|".join(
        [
            cfg.llm.model,
            h.rationale_mode,
            h.suggestion_level,
            cfg.game.game,
            f"seed{cfg.seed}",
        ]
    )


NO_MODEL_SENTINEL = "NO_MODEL"


def slot_id_replay(
    rationale_mode: str,
    model: str,
    suggestion_level: str,
    subject: str,
    game: str,
) -> str:
    """Build slot ID for the unified replay table.

    For 'action-only' rows (no LLM generation), pass
    model=NO_MODEL_SENTINEL ('NO_MODEL').
    """
    return "|".join([rationale_mode, model, suggestion_level, subject, game])


def slot_id_feature_extraction(
    extraction_model: str,
    prompt_config: str,
    subject: str,
    game: str,
) -> str:
    """Build slot ID for the feature-extraction table."""
    return "|".join([extraction_model, prompt_config, subject, game])


def slot_id_ablation(
    ablation_tag: str,
    extraction_model: str,
    prompt_config: str,
    subject: str,
    game: str,
) -> str:
    """Build slot ID for the ablation table.

    ``ablation_tag`` is an opaque string defined by the caller — each
    ablation study constructs its own tag from whatever parameters it
    varies.  This function only joins it into the key.
    """
    return "|".join([ablation_tag, extraction_model, prompt_config, subject, game])


NO_VARIANT_SENTINEL = "none"


def slot_id_behavioural(
    agent_type: str,
    model: str,
    rationale_mode: str,
    suggestion_level: str,
    instance_id: str,
    game: str,
    cohort: str,
    level: int,
) -> str:
    """Build slot ID for the behavioural summary table.

    For non-LLM agents, pass rationale_mode and suggestion_level as
    NO_VARIANT_SENTINEL ('none').
    """
    return "|".join(
        [
            agent_type,
            model,
            rationale_mode,
            suggestion_level,
            instance_id,
            game,
            cohort,
            str(level),
        ]
    )
