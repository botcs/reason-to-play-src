# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""
Manual play mode - step into the LLM's shoes.

See EXACTLY what the LLM receives and respond with JSON as if you were the LLM.

Usage:
    python -m src.llm_eval.manual_play --game bait_vgfmri4
    python -m src.llm_eval.manual_play --game bait_vgfmri4 --level 3
"""

import argparse
import json
import sys

from src.llm_eval.generative_gameplay.agent import GameplayAgent


class ManualLLM:
    """A 'fake LLM' that prompts the user for input."""

    def __init__(self, show_system_prompt: bool = True):
        self.show_system_prompt = show_system_prompt
        self._shown_system = False

    def generate(self, messages: list[dict]) -> str:
        """Show prompt to user and get their JSON response."""
        # Show system prompt once
        if self.show_system_prompt and not self._shown_system:
            print("\n" + "=" * 80)
            print("SYSTEM PROMPT:")
            print("=" * 80)
            print(messages[0]["content"])
            print("=" * 80)
            self._shown_system = True

        # Show user prompt
        print("\n" + "=" * 80)
        print("USER PROMPT:")
        print("=" * 80)
        print(messages[1]["content"])
        print("=" * 80)

        # Get user's JSON response
        print("\nEnter your JSON response (or 'q' to quit):")
        print(
            'Example: {"reasoning": "...", "note_to_self": "...", "commitment": "...", "action": "up"}'
        )
        print()

        lines = []
        while True:
            line = input()
            if line.strip().lower() == "q":
                sys.exit(0)
            lines.append(line)
            # Try to parse as JSON - if valid, we're done
            text = "\n".join(lines)
            if text.strip():
                # Check if it looks complete (has closing brace)
                if text.rstrip().endswith("}"):
                    return text


def main():
    parser = argparse.ArgumentParser(description="Manual play - be the LLM")
    parser.add_argument("--game", required=True, help="Game name (e.g., bait_vgfmri4)")
    parser.add_argument("--level", type=int, default=0, help="Starting level")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps")
    parser.add_argument(
        "--no-system", action="store_true", help="Skip showing system prompt"
    )
    args = parser.parse_args()

    # Create agent with our manual "LLM"
    manual_llm = ManualLLM(show_system_prompt=not args.no_system)

    agent = GameplayAgent(
        game_name=args.game,
        model_path=None,
        max_steps=args.max_steps,
        llm=manual_llm,
    )

    print("\n" + "=" * 80)
    print(f"MANUAL PLAY: {args.game} level {args.level}")
    print("You are the LLM. Respond with JSON exactly as the LLM would.")
    print("=" * 80)

    result = agent.run(start_level=args.level)

    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print(json.dumps(result, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
