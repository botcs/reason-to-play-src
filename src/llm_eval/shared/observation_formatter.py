# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
ObservationFormatter: Formats current game observation into prompt-ready text.

Produces output like:
    Grid: 21x12
    Score: 15 (+5)
    Objects:
    - ORANGE at (14, 3)
    - PURPLE at (3, 10)
    Walls at: (0, 0-11), (20, 0-11), (0-20, 0), (0-20, 11)
"""

from src.utils import COLOR_DICT

# Map RGB tuples to color names (for legacy support)
RGB_TO_COLOR = {v: k for k, v in COLOR_DICT.items()}


class ObservationFormatter:
    """Formats current game observation into prompt-ready text."""

    def format_observation(
        self,
        obs: list,
        score: int,
        reward: float,
        resources: dict | None = None,
        resource_colors: dict | None = None,
    ) -> str:
        """
        Returns formatted observation string.

        Args:
            obs: 2D grid from info['state']
            score: Current total score
            reward: Last step reward (for delta display)
            resources: Optional dict of avatar resources
            resource_colors: Optional dict mapping resource names to colors

        Returns:
            Formatted observation string for prompt
        """
        lines = []

        # Grid dimensions
        width, height = self._get_grid_dimensions(obs)
        lines.append(f"Grid: {width}x{height}")

        # Collect every non-wall, non-floor sprite. Two sprites may
        # occupy the same cell (e.g. avatar standing on a goal); each
        # is emitted on its own line so the LLM sees both.
        objects: list[dict] = []
        wall_positions = []

        for x, column in enumerate(obs):
            for y, cell in enumerate(column):
                for obj in cell:
                    name = obj["name"]
                    obj_id = obj["obj_id"]
                    color = self._get_color_name(obj.get("color"))

                    if name == "wall":
                        wall_positions.append((x, y))
                    elif name not in ("floor", "background"):
                        objects.append(
                            {
                                "obj_id": obj_id,
                                "color": color or name.upper(),
                                "pos": (x, y),
                            }
                        )

        # Resources/inventory (show colors instead of internal resource names)
        if resources:
            resource_items = []
            for res_name, count in resources.items():
                if count > 0:
                    # Use color if available, otherwise fall back to resource name
                    display_name = res_name
                    if resource_colors and res_name in resource_colors:
                        display_name = resource_colors[res_name]
                    resource_items.append(f"{display_name}: {count}")
            if resource_items:
                lines.append(f"Inventory: {', '.join(resource_items)}")

        # Score with delta
        if reward != 0:
            sign = "+" if reward > 0 else ""
            lines.append(f"Score: {score} ({sign}{int(reward)})")
        else:
            lines.append(f"Score: {score}")

        # Objects (sorted by internal obj_id for deterministic ordering)
        lines.append("Objects:")
        if objects:
            objects.sort(key=lambda o: o["obj_id"])
            for obj in objects:
                lines.append(f"- {obj['color']} at ({obj['pos'][0]},{obj['pos'][1]})")
        else:
            lines.append("- none")

        # Walls (compressed)
        if wall_positions:
            wall_str = self._compress_wall_ranges(wall_positions)
            lines.append(f"Walls at: {wall_str}")

        return "\n".join(lines)

    def _get_grid_dimensions(self, obs: list) -> tuple[int, int]:
        """Returns (width, height) of grid."""
        if not obs:
            return (0, 0)
        width = len(obs)
        height = len(obs[0]) if obs[0] else 0
        return (width, height)

    def _compress_wall_ranges(self, wall_positions: list[tuple]) -> str:
        """
        Compress wall positions into ranges.

        Groups walls by rows and columns to produce compact representation
        like '(0, 0-11), (20, 0-11), (0-20, 0), (0-20, 11)'
        """
        if not wall_positions:
            return "none"

        wall_set = set(wall_positions)

        # Group by x (vertical lines)
        x_groups = {}
        for x, y in wall_positions:
            if x not in x_groups:
                x_groups[x] = []
            x_groups[x].append(y)

        # Group by y (horizontal lines)
        y_groups = {}
        for x, y in wall_positions:
            if y not in y_groups:
                y_groups[y] = []
            y_groups[y].append(x)

        ranges = []
        used = set()

        # Find vertical wall segments (same x, consecutive y)
        for x in sorted(x_groups.keys()):
            y_vals = sorted(x_groups[x])
            segments = self._find_consecutive_segments(y_vals)
            for start_y, end_y in segments:
                # Only use if this is a significant segment
                if end_y - start_y >= 2:
                    for y in range(start_y, end_y + 1):
                        used.add((x, y))
                    if start_y == end_y:
                        ranges.append(f"({x}, {start_y})")
                    else:
                        ranges.append(f"({x}, {start_y}-{end_y})")

        # Find horizontal wall segments (same y, consecutive x)
        for y in sorted(y_groups.keys()):
            x_vals = sorted(y_groups[y])
            segments = self._find_consecutive_segments(x_vals)
            for start_x, end_x in segments:
                # Check which positions haven't been used yet
                segment_positions = [(x, y) for x in range(start_x, end_x + 1)]
                unused_in_segment = [p for p in segment_positions if p not in used]

                if len(unused_in_segment) >= 2:
                    # Find the range of unused positions
                    unused_x = sorted([p[0] for p in unused_in_segment])
                    sub_segments = self._find_consecutive_segments(unused_x)
                    for sub_start, sub_end in sub_segments:
                        if sub_end - sub_start >= 1:
                            for x in range(sub_start, sub_end + 1):
                                used.add((x, y))
                            if sub_start == sub_end:
                                ranges.append(f"({sub_start}, {y})")
                            else:
                                ranges.append(f"({sub_start}-{sub_end}, {y})")

        # Add remaining individual walls
        remaining = wall_set - used
        for x, y in sorted(remaining):
            ranges.append(f"({x}, {y})")

        # Limit output length
        if len(ranges) > 20:
            return f"{len(wall_positions)} wall positions (borders)"

        return ", ".join(ranges) if ranges else "none"

    def _find_consecutive_segments(self, values: list[int]) -> list[tuple[int, int]]:
        """Find consecutive segments in a sorted list of integers."""
        if not values:
            return []

        segments = []
        start = values[0]
        end = values[0]

        for v in values[1:]:
            if v == end + 1:
                end = v
            else:
                segments.append((start, end))
                start = v
                end = v

        segments.append((start, end))
        return segments

    def _get_color_name(self, color: tuple | list | str | None) -> str | None:
        """Get color name from various formats (RGB tuple, list, or string)."""
        if color is None:
            return None
        # If already a string, return it directly
        if isinstance(color, str):
            return color
        # Otherwise try to convert from RGB tuple
        rgb_tuple = tuple(color) if isinstance(color, list) else color
        return RGB_TO_COLOR.get(rgb_tuple)
