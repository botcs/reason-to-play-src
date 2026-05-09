# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
from pygame.math import Vector2
from src.vgdl.core import Action, Color

# Canonical palette matches tsividis_vgdl/vgdl/colors.py -- the RGB values
# the fMRI participants saw. LIGHTPURPLE and LIGHTPINK aren't in tsividis
# and retain their legacy values.
GREEN = Color((129, 199, 132))
BLUE = Color((25, 118, 210))
RED = Color((211, 47, 47))
GRAY = Color((69, 90, 100))
WHITE = Color((250, 250, 250))
BROWN = Color((109, 76, 65))
BLACK = Color((55, 71, 79))
ORANGE = Color((230, 81, 0))
YELLOW = Color((255, 245, 157))
PINK = Color((255, 138, 128))
GOLD = Color((255, 196, 0))
LIGHTRED = Color((255, 82, 82))
LIGHTORANGE = Color((255, 112, 67))
LIGHTBLUE = Color((144, 202, 249))
LIGHTGREEN = Color((185, 246, 202))
LIGHTGRAY = Color((207, 216, 220))
DARKGRAY = Color((68, 90, 100))
DARKBLUE = Color((1, 87, 155))
PURPLE = Color((92, 107, 192))
LIGHTPURPLE = Color((200, 150, 220))
LIGHTPINK = Color((255, 230, 230))

UP = Vector2(0, -1)
DOWN = Vector2(0, 1)
LEFT = Vector2(-1, 0)
RIGHT = Vector2(1, 0)

BASEDIRS = [UP, LEFT, DOWN, RIGHT]
BASEDIRS_DICT = dict(UP=UP, LEFT=LEFT, DOWN=DOWN, RIGHT=RIGHT)
BASEDIRS_DICT = dict(UP=UP, LEFT=LEFT, DOWN=DOWN, RIGHT=RIGHT)


def get_str_from_vector(vector):
    if vector == UP:
        return "UP"
    elif vector == DOWN:
        return "DOWN"
    elif vector == LEFT:
        return "LEFT"
    elif vector == RIGHT:
        return "RIGHT"
    else:
        raise ValueError


NOOP = Action()
