"""Locate translated game definitions and register headless Gym environments."""

from pathlib import Path


def definitions_directory() -> Path:
    """Return the game definitions installed with this package."""
    return Path(__file__).resolve().parent / "definitions"


def game_directory(game, *, generated=False, root=None) -> Path:
    """Resolve a game in the supplied definitions directory or packaged games."""
    directory = Path(root) if root is not None else definitions_directory()
    if generated:
        directory /= "generated"
    return directory / f"{game}_v0"


def game_files(game, level=0, *, generated=False, root=None) -> tuple[Path, Path]:
    """Return the translated rules and layout for one game and level."""
    directory = game_directory(game, generated=generated, root=root)
    rules = directory / f"{game}.txt"
    layout = directory / f"{game}_lvl{level}.txt"
    for path in (rules, layout):
        if not path.is_file():
            raise FileNotFoundError(path)
    return rules, layout


def register_game(game, generated=False, level=0, fast=False, *, root=None):
    """Register the selected game and level using the object observation API."""
    from gym.envs.registration import register

    rules, layout = game_files(game, level, generated=generated, root=root)
    register(
        id=f"{game}-v0",
        entry_point="environments.vgdl.interfaces.gym:VGDLEnv",
        kwargs={
            "game_file": str(rules),
            "level_file": str(layout),
            "obs_type": "objects",
            "block_size": 1 if fast else 50,
        },
    )
