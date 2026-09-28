import os

from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings

os.environ["SDL_AUDIODRIVER"] = "dsp"
import pdb

import numpy as np
import pygame


def load_game(game_name, games_folder, level_transform=None):
    def _load_level(gameString, levelString):

        headless = True

        rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)

        # (self, gameDef, levelDef, observationType=OBSERVATION_GLOBAL, visualize=False, actionset=BASEDIRS, positions=None, **kwargs)

        rle = rleCreateFunc()
        # import pdb; pdb.set_trace()
        rle.visualize = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        # pdb.set_trace()
        pygame.init()

        return rle

    def _pad_level_to_size(levelString, target_width, target_height):
        """Pad a level string with walls to reach target dimensions"""
        lines = levelString.strip().split("\n")
        current_height = len(lines)
        current_width = max(len(line) for line in lines) if lines else 0

        # Pad each line to target width
        padded_lines = []
        for line in lines:
            padding_needed = target_width - len(line)
            padded_line = line + "w" * padding_needed
            padded_lines.append(padded_line)

        # Add rows to reach target height
        rows_needed = target_height - current_height
        for _ in range(rows_needed):
            padded_lines.append("w" * target_width)

        return "\n".join(padded_lines)

    def _gen_color():
        from vgdl.colors import colorDict

        color_list = colorDict.values()
        color_list = [c for c in color_list if c not in ["UUWSWF"]]
        for color in color_list:
            yield color

    file_list = {}
    for file in os.listdir(games_folder):
        # import pdb; pdb.set_trace()
        if "DS" not in file:

            if "expt_ee" in game_name:
                if game_name in file:
                    if "lvl" not in file:
                        level = file.split("desc_")[1][0]
                        file_list["game_{}".format(level)] = file
                    else:
                        level = file.split("_lvl")[1].split(".")[0]
                        file_list[int(level)] = file
            else:
                if (
                    game_name == file.split(".txt")[0]
                    or game_name == file.split("_lvl")[0]
                ):
                    if "lvl" not in file:
                        file_list["game"] = file
                    else:
                        level = file.split("_lvl")[1].split(".")[0]
                        file_list[int(level)] = file

    # new_doc = ''
    # with open('{}/{}'.format(games_folder, file_list['game']), 'r') as f:
    # 	new_doc = []
    # 	g = _gen_color()
    # 	for line in f.readlines():
    # 		new_line = (" ".join([string if string[:4]!="img="
    # 			else "color={}".format(next(g))
    # 			for string in line.split(" ")]))
    # 		new_doc.append(new_line)
    # 	new_doc = "\n".join(new_doc)

    # import pdb; pdb.set_trace()

    if "expt_ee" not in game_name:
        with open("{}/{}".format(games_folder, file_list["game"]), "r") as game:
            gameString = game.read()

    env_list = {}

    num_levels = len(file_list.keys()) - 1
    if "expt_ee" in game_name:
        num_levels = int(len(file_list.keys()) / 2)

    # First pass: read all level strings and find max dimensions
    level_strings = {}
    max_width = 0
    max_height = 0

    for lvl_idx in range(num_levels):
        with open("{}/{}".format(games_folder, file_list[lvl_idx]), "r") as level:
            level_strings[lvl_idx] = level.read()

        # Calculate dimensions of this level
        lines = level_strings[lvl_idx].strip().split("\n")
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0

        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # Second pass: optionally transform levels before padding them to a common size
    transformed_levels = {}
    for lvl_idx in range(num_levels):
        padded_level = _pad_level_to_size(level_strings[lvl_idx], max_width, max_height)
        if callable(level_transform):
            try:
                candidate = level_transform(
                    padded_level, lvl_idx, max_width, max_height
                )
            except TypeError:
                candidate = level_transform(padded_level, lvl_idx)
            if candidate is None:
                transformed_levels[lvl_idx] = padded_level
            elif isinstance(candidate, str):
                transformed_levels[lvl_idx] = candidate
            else:
                raise TypeError(
                    "level_transform must return either a string layout or None"
                )
        else:
            transformed_levels[lvl_idx] = padded_level

    final_max_width = 0
    final_max_height = 0
    for lvl_idx in range(num_levels):
        lines = transformed_levels[lvl_idx].split("\n")
        # Handle possible empty strings gracefully
        height = len([line for line in lines if line != ""]) if lines != [""] else 0
        if height == 0:
            continue
        width = max(len(line) for line in lines)
        final_max_width = max(final_max_width, width)
        final_max_height = max(final_max_height, height)

    if final_max_width == 0:
        final_max_width = max_width
    if final_max_height == 0:
        final_max_height = max_height

    # Third pass: pad all (potentially transformed) levels to the final dimensions and create environments
    for lvl_idx in range(num_levels):

        if "expt_ee" in game_name:
            with open(
                "{}/{}".format(games_folder, file_list["game_{}".format(lvl_idx)]), "r"
            ) as game:
                gameString = game.read()

        final_level = _pad_level_to_size(
            transformed_levels.get(lvl_idx, level_strings[lvl_idx]),
            final_max_width,
            final_max_height,
        )

        env_list[lvl_idx] = _load_level(gameString, final_level)

    return env_list
