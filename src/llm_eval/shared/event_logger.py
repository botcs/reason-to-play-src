# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
EventLogger: Formats action-log strings from real VGDL engine events.

Each log line has the form:

    "[L<lvl> A<att> #<step>] <ACTION> -> <events>; Score <+N>[WIN]|[LOSE]"

Example:

    "[L0 A0 #3] RIGHT -> bounceForward(box, avatar); Score +1"

The `<events>` segment lists ONLY real VGDL opcodes decoded from the
engine's `effectListByColor` / `events_triggered` channel.  When an
action triggers no engine opcode and no score change, the segment
reads "no change".

This module used to also emit observation-diff fabrications
(`(COLOR) moved`, `appeared`, `disappeared`) alongside or in place of
the engine events.  Those entries were removed deliberately: they
relied on UUID identity across frames, which the Tomov 2023 zstate
recording corrupts (SpawnPoint phantom cycling, sprite-position lag,
re-IDing under randomization).  A future grid-diff interpretation
layer can be added as a SEPARATE module if we ever need it.
"""

from src.utils import COLOR_DICT

# Map RGB tuples to color names (used when an engine event references
# an object whose color we want to surface in the log).
RGB_TO_COLOR = {v: k for k, v in COLOR_DICT.items()}


class EventLogger:
    """Formats action logs from real VGDL engine events."""

    def __init__(self):
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "ACTION"]

    def log_action(
        self,
        step_num: int,
        action: int,
        prev_obs: list,
        curr_obs: list,
        events: list,
        reward: float,
        won: bool,
        lose: bool,
        prev_score: int,
        curr_score: int,
        level: int = 0,
        attempt: int = 0,
        timeout: bool = False,
    ) -> str:
        """Return the formatted action-log string for a single step.

        Args:
            step_num: Step number within this attempt.
            action: Action index (0-5).
            prev_obs: 2D grid before the action (used only to resolve
                color names for sprites that the engine event mentions
                but that were destroyed in this tick).
            curr_obs: 2D grid after the action (same role as prev_obs).
            events: Engine events (`effectListByColor` -> list of
                `(effect_name, (name1, id1), (name2, id2))` tuples).
            reward: Score delta for this step.
            won: True iff the game ended in a win this step.
            lose: True iff the game ended in a loss this step.
            prev_score: Score before the action (unused in current
                output; retained for callers).
            curr_score: Score after the action (unused; retained).
            level: Level number for the log prefix.
            attempt: Attempt number for the log prefix.
            timeout: True iff the play ended because the level's time
                budget expired with no win or loss. Renders as
                `[LEVEL TIMEOUT]` (mutually exclusive with `[WIN]` /
                `[LOSE]`).

        Returns:
            Formatted action log string.
        """
        action_name = self.action_names[action]
        event_str = self._format_events(
            events, prev_obs, curr_obs, reward, won, lose, timeout
        )
        return f"[L{level} A{attempt} #{step_num}] {action_name} -> {event_str}"

    # ------------------------------------------------------------------
    # Engine-events formatter
    # ------------------------------------------------------------------

    def _format_events(
        self,
        events: list,
        prev_obs: list,
        curr_obs: list,
        reward: float,
        won: bool,
        lose: bool,
        timeout: bool,
    ) -> str:
        """Format engine events as `effect_name(actor/COLOR, actee/COLOR)`."""
        all_objects = {
            **self._index_objects(prev_obs),
            **self._index_objects(curr_obs),
        }

        parts: list[str] = []
        for event in events:
            effect_name = event[0]
            name1, obj_id1 = event[1]
            name2, obj_id2 = event[2]

            # Avatar-hits-screen-edge events are emitted by the engine
            # but carry no interesting semantic for gameplay logs.
            if name2 == "EOS":
                continue

            label1 = self._name_color_label(name1, obj_id1, all_objects)
            label2 = self._name_color_label(name2, obj_id2, all_objects)
            parts.append(f"{effect_name}({label1}, {label2})")

        return self._finalize(parts, reward, won, lose, timeout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finalize(
        self,
        parts: list[str],
        reward: float,
        won: bool,
        lose: bool,
        timeout: bool,
    ) -> str:
        """Join event parts and append score / outcome annotations."""
        body_parts = list(parts)
        if reward != 0:
            sign = "+" if reward > 0 else ""
            body_parts.append(f"Score {sign}{int(reward)}")

        has_terminal = won or lose or timeout
        if body_parts:
            result = "; ".join(body_parts)
        elif has_terminal:
            result = ""
        else:
            result = "no change"

        # Outcomes are mutually exclusive. [WIN] / [LOSE] mean the engine
        # resolved the play; [LEVEL TIMEOUT] means the level's time
        # budget expired without a resolution (see
        # VGFMRI_DB_README.md:174-176).
        if won:
            result = (result + " [WIN]").lstrip()
        elif lose:
            result = (result + " [LOSE]").lstrip()
        elif timeout:
            result = (result + " [LEVEL TIMEOUT]").lstrip()

        return result

    def _index_objects(self, obs: list) -> dict[str, dict]:
        """Index every object in obs by its obj_id for color/name lookup."""
        objects: dict[str, dict] = {}
        if obs is None:
            return objects
        for column in obs:
            for cell in column:
                for obj in cell:
                    color = self._get_color_name(obj.get("color"))
                    objects[obj["obj_id"]] = {"name": obj["name"], "color": color}
        return objects

    @staticmethod
    def _name_color_label(name: str, obj_id: str, objects: dict[str, dict]) -> str:
        """Return 'name/COLOR' if the object can be found in obs, else 'name'."""
        obj = objects.get(obj_id)
        if obj is not None and obj["color"] is not None:
            return f"{name}/{obj['color']}"
        return name

    @staticmethod
    def _get_color_name(color: tuple | list | str | None) -> str | None:
        """Normalize a color value (RGB tuple / list / name string) to a name."""
        if color is None:
            return None
        if isinstance(color, str):
            return color
        rgb_tuple = tuple(color) if isinstance(color, list) else color
        return RGB_TO_COLOR.get(rgb_tuple)
