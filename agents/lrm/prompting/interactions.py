"""Track observed game interactions for gameplay and recorded-action prompts."""


def build_interactions_from_game(game) -> set[tuple[str, str, str]]:
    """Build the set of known (color1, effect, color2) interaction tuples
    directly from a VGDL game's domain and sprite registry.

    Iterates over ``domain.collision_eff``, resolves each effect's
    actor/actee sprite types to their leaf colors (expanding parent types),
    and returns the full set.  EOS interactions are excluded because the
    tracker cannot resolve them to a colour (no state entry for EOS).
    """
    registry = game.sprite_registry
    domain = game.domain

    def _leaf_keys(stype: str) -> list[str]:
        """Return all registered leaf sprite keys that match *stype*."""
        if stype in registry.sprite_keys:
            return [stype]
        return [key for key in registry.sprite_keys if stype in registry.stypes[key]]

    def _color(key: str) -> str | None:
        """Extract the colour string from a sprite type's class args."""
        img = registry.class_args[key].get("img")
        if img and "/" in img:
            return img.split("/")[1].upper()
        return None

    interactions: set[tuple[str, str, str]] = set()
    for effect in domain.collision_eff:
        if effect.actee_stype == "EOS":
            continue
        for actor_key in _leaf_keys(effect.actor_stype):
            c1 = _color(actor_key)
            if c1 is None:
                continue
            for actee_key in _leaf_keys(effect.actee_stype):
                c2 = _color(actee_key)
                if c2 is None:
                    continue
                interactions.add((c1, effect.name, c2))
    return interactions


class InteractionTracker:
    """Track unique interactions discovered from game events."""

    def __init__(self, known_interactions: set[tuple[str, str, str]] | None = None):
        """
        Initialize InteractionTracker.

        Args:
            known_interactions: The full set of (color1, effect, color2)
                tuples that the game can produce.  Built automatically via
                ``build_interactions_from_game`` after the env is created.
                If *None*, ``coverage`` returns 0.0 until it is set.
        """
        self.known: set[tuple[str, str, str]] = known_interactions or set()
        self.seen: set[tuple[str, str, str]] = set()  # (color1, effect, color2)
        self.discovery_order: list[
            tuple[int, tuple[str, str, str]]
        ] = []  # (step, interaction)
        self.all_instances: list[
            tuple[int, tuple[str, str, str], tuple[str, str]]
        ] = []  # (step, interaction, (obj_id1, obj_id2))

    def record_events(
        self,
        events: list,
        prev_obs: list,
        step: int,
    ) -> None:
        """Extract and record unique interactions from game events.

        Args:
            events: List of event tuples from game step
            prev_obs: Observation before the action (to look up colors)
            step: Current step number
        """
        for event in events:
            effect_name = event[0]
            sprite1 = event[1]  # (name, obj_id)
            sprite2 = event[2]  # (name, obj_id)

            color1 = self._get_color_from_obs(prev_obs, sprite1[1])
            color2 = self._get_color_from_obs(prev_obs, sprite2[1])

            if color1 is None or color2 is None:
                continue

            # Normalize colors to uppercase to avoid duplicate interactions with different casing
            interaction = (color1.upper(), effect_name, color2.upper())

            # Track all instances (for detailed logging)
            self.all_instances.append((step, interaction, (sprite1[1], sprite2[1])))

            # Track unique interactions (for coverage)
            if interaction not in self.seen:
                self.seen.add(interaction)
                self.discovery_order.append((step, interaction))
                if self.known and interaction not in self.known:
                    print(
                        f"[InteractionTracker] WARNING: discovered interaction "
                        f"{interaction} not in known set (step {step}). "
                        f"Adding to known -- build_interactions_from_game may be incomplete."
                    )
                    self.known.add(interaction)

    def _get_color_from_obs(self, obs: list, obj_id: str) -> str | None:
        """Get color name for an object by ID."""
        if obs is None:
            return None
        for column in obs:
            for cell in column:
                for obj in cell:
                    if obj["obj_id"] == obj_id:
                        color = obj.get("color")
                        if color is None:
                            return None
                        if isinstance(color, str):
                            return color
                        # Convert RGB tuple to color name
                        from environments.colors import COLOR_DICT

                        rgb_to_color = {v: k for k, v in COLOR_DICT.items()}
                        rgb_tuple = tuple(color) if isinstance(color, list) else color
                        return rgb_to_color.get(rgb_tuple)
        return None

    @property
    def total(self) -> int:
        """Number of known interactions (for backward compat)."""
        return len(self.known)

    @property
    def coverage(self) -> float:
        """Return fraction of known interactions discovered."""
        return len(self.seen) / len(self.known) if self.known else 0.0

    @property
    def interactions_discovered(self) -> int:
        """Return number of unique interactions discovered."""
        return len(self.seen)

    def reset(self) -> None:
        """Reset tracker for a new run (but keep known interactions)."""
        self.seen.clear()
        self.discovery_order.clear()
        self.all_instances.clear()
