# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
ResponseParser: Parses LLM JSON responses into structured data.

Mode-agnostic: extracts JSON, validates that 'action' exists, and returns
all fields found. Does NOT require specific fields beyond 'action'.
"""

import json
import re


class ResponseParser:
    """Parses LLM JSON responses into structured data."""

    # Action string to index mapping
    ACTION_MAP = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
        "wait": 4,
        "action": 5,
        "reset": -1,  # sentinel; intercepted by agent before reaching env
    }

    def parse(self, response: str) -> dict:
        """
        Parse LLM response JSON.  Returns ALL fields found in the JSON dict,
        only requires 'action'.

        Args:
            response: Raw LLM response string

        Returns:
            Dict with at least 'action' key, plus any other fields present.

        Raises:
            ValueError: On invalid response format, missing action, or empty response.
        """
        if not response:
            raise ValueError("Empty response from LLM (content was None/empty)")

        response = response.strip()

        json_str = self._extract_json(response)
        data = json.loads(json_str)

        if "action" not in data:
            raise ValueError(
                f"Missing required field: action. Got keys: {list(data.keys())}"
            )

        action_str = data["action"].lower().strip()
        if action_str not in self.ACTION_MAP:
            raise ValueError(
                f"Invalid action: {data['action']}. Must be one of: {list(self.ACTION_MAP.keys())}"
            )

        # Return all fields, with action normalized
        result = {k: str(v) for k, v in data.items() if k != "action"}
        result["action"] = action_str
        return result

    def parse_replay(self, response: str) -> dict:
        """
        Parse replay generation response. Does NOT require 'action' field.

        In human replay, the model generates rationale only;
        the action comes from the participant's behavioral data.

        Args:
            response: Raw LLM response string

        Returns:
            Dict with all fields found in the JSON (e.g. rationale).

        Raises:
            ValueError: On invalid response format or empty response
        """
        if not response:
            raise ValueError("Empty response from LLM (content was None/empty)")

        response = response.strip()
        json_str = self._extract_json(response)
        data = json.loads(json_str)
        return {k: str(v) for k, v in data.items()}

    def action_to_index(self, action_str: str) -> int:
        """
        Convert action string to index.

        Args:
            action_str: Action name ('up', 'down', 'left', 'right', 'wait', 'action')

        Returns:
            Action index (0-5)

        Raises:
            ValueError: If action string is invalid
        """
        action_lower = action_str.lower().strip()
        if action_lower not in self.ACTION_MAP:
            raise ValueError(f"Invalid action: {action_str}")
        return self.ACTION_MAP[action_lower]

    def index_to_action(self, index: int) -> str:
        """
        Convert action index to string.

        Args:
            index: Action index (0-5)

        Returns:
            Action string

        Raises:
            ValueError: If index is invalid
        """
        index_to_action = {v: k for k, v in self.ACTION_MAP.items()}
        if index not in index_to_action:
            raise ValueError(f"Invalid action index: {index}")
        return index_to_action[index]

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON object from text.

        Handles cases where JSON might be wrapped in markdown code blocks
        or have extra text around it.

        Args:
            text: Raw text that may contain JSON

        Returns:
            Extracted JSON string

        Raises:
            ValueError: If no valid JSON found
        """
        # Try direct parse first
        text = text.strip()
        if text.startswith("{"):
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[: i + 1]

        # Try to find JSON in markdown code blocks
        code_block_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, text)
            if match:
                potential_json = match.group(1).strip()
                if potential_json.startswith("{"):
                    return potential_json

        # Try to find JSON object anywhere in text
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)

        # Last resort - try to find content between first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]

        raise ValueError(f"Could not extract JSON from response: {text[:200]}...")
