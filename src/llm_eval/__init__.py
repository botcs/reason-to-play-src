# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
LLM Evaluation Scaffolding for VGDL Games.

Subpackages:
    shared/              -- Config, harness, formatters, wrappers (used by both pipelines)
    generative_gameplay/ -- GameplayAgent, run.py (LLM plays games)
    human_replay/        -- ReplayAgent, run_replay.py (replay human data for feature extraction)

Usage:
    from src.llm_eval.generative_gameplay.agent import GameplayAgent
    from src.llm_eval.shared.config import Config
    from src.llm_eval.human_replay.replay_agent import ReplayAgent
"""

from src.llm_eval.shared.event_logger import EventLogger
from src.llm_eval.shared.observation_formatter import ObservationFormatter
from src.llm_eval.shared.harness import Harness
from src.llm_eval.shared.response_parser import ResponseParser
from src.llm_eval.generative_gameplay.agent import GameplayAgent, run_game

__all__ = [
    "EventLogger",
    "ObservationFormatter",
    "Harness",
    "ResponseParser",
    "GameplayAgent",
    "run_game",
]
