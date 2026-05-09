# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Coordinate utilities for LLM display layer.

Screen coordinates: X=0 at left (increases rightward),
Y=0 at top (increases downward). Matches the VGDL engine and vgdl-js viewer.
"""


def format_pos(x: int, y: int) -> str:
    """Format a grid position as '(x,y)'."""
    return f"({x},{y})"
