# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Sampling strategies for selecting which steps to send to the LLM for inference.

Strategies:
- non_wait: Only sample steps where the action is not WAIT
- key_moments: Sample on deaths, wins, score changes, first interaction with each color
- all: Sample all steps (expensive but complete)

Frame striding can be applied on top of any strategy via --frame-stride argument.
"""

from typing import Iterator


def sample_non_wait(steps: list[dict]) -> Iterator[dict]:
    """
    Sample only steps where the action is not WAIT.

    This filters out idle frames (80%+ of frames at 20Hz are WAIT).

    Args:
        steps: List of step dicts

    Yields:
        Step dicts where action_taken != 'WAIT' and != 'START'
    """
    for step in steps:
        action = step.get("action_taken", "WAIT")
        if action not in ("WAIT", "START"):
            yield step


def sample_key_moments(steps: list[dict]) -> Iterator[dict]:
    """
    Sample on key moments: deaths, wins, score changes, first interaction with each color.

    Key moments are identified by:
    - [WIN] or [LOSE] in action history
    - Score changes (+ or - in action log)
    - First mention of a new color in action history

    Args:
        steps: List of step dicts

    Yields:
        Step dicts at key moments
    """
    seen_colors = set()
    prev_score = None

    for step in steps:
        is_key_moment = False

        # Check action history for events
        history = step.get("action_history", [])
        if history:
            latest_action = history[0]  # Most recent action (reversed order)

            # Check for win/lose
            if "[WIN]" in latest_action or "[LOSE]" in latest_action:
                is_key_moment = True

            # Check for score change
            if ", +" in latest_action or ", -" in latest_action:
                is_key_moment = True

            # Check for new color interaction
            # Look for patterns like "collecting (COLOR)", "blocked by (COLOR)", etc.
            import re

            colors_in_action = re.findall(r"\(([A-Z]+)\)", latest_action)
            for color in colors_in_action:
                if color not in seen_colors:
                    seen_colors.add(color)
                    is_key_moment = True

        # Also check current state for score
        state_str = step.get("current_state", "")
        for line in state_str.split("\n"):
            if line.startswith("Score:"):
                # Extract score value
                import re

                match = re.search(r"Score:\s*(-?\d+)", line)
                if match:
                    curr_score = int(match.group(1))
                    if prev_score is not None and curr_score != prev_score:
                        is_key_moment = True
                    prev_score = curr_score
                break

        if is_key_moment:
            yield step


def sample_all(steps: list[dict]) -> Iterator[dict]:
    """
    Sample all steps.

    Args:
        steps: List of step dicts

    Yields:
        All step dicts
    """
    yield from steps


# Registry of sampling strategies
SAMPLING_STRATEGIES = {
    "non_wait": sample_non_wait,
    "key_moments": sample_key_moments,
    "all": sample_all,
}


def get_sampler(strategy: str):
    """
    Get a sampling function by name.

    Args:
        strategy: Name of the sampling strategy

    Returns:
        A function that takes steps and yields sampled steps
    """
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Available: {list(SAMPLING_STRATEGIES.keys())}"
        )

    return SAMPLING_STRATEGIES[strategy]


def list_strategies() -> list[str]:
    """Return list of available sampling strategies."""
    return list(SAMPLING_STRATEGIES.keys())
