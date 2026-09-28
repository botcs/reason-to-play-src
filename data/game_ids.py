"""Game identifiers shared by human and agent data."""

import re

GAMES = {"bait", "chase", "helper", "lemmings", "zelda", "plaqueattack", "avoidgeorge"}


def game_identity(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"(vgfmri[34])_(.+)|(.+)_(vgfmri[34])", name, re.I)
    if match is None:
        raise ValueError(f"Unrecognized game variant: {name!r}")
    cohort, game = (match[1], match[2]) if match[1] else (match[4], match[3])
    return game.lower(), cohort.lower()


def canonical_game_id(name: str) -> str:
    """Use the game identifiers shared by released human and agent data."""
    game, cohort = game_identity(name)
    if game not in GAMES:
        raise ValueError(f"Unrecognized game: {name!r}")
    spelling = {"avoidgeorge": "avoidGeorge", "plaqueattack": "plaqueAttack"}
    return f"{spelling.get(game, game)}_{cohort}"
