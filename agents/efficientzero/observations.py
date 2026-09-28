"""EfficientZero images from recorded sprites and the translated game description.

This path paints recorded frames. It does not simulate the game or require a
second rule definition, an ASCII level map, or a Pygame display.
"""

from collections.abc import Mapping


def render_recorded_frames(
    play: Mapping,
    states,
    *,
    resize=84,
    num_channels=1,
    background=(207, 216, 220),
):
    """Match the preserved EZ rasterization using the released human frames.

    The historical EZ canvas uses 30 pixels per cell. Source sprite rectangles
    keep their original pixel units, even when their block size differs. These
    are separate scales; changing either changes the model's image inputs.
    The authoritative translated game description supplies sprite draw order;
    all positions, sizes and colors come from the recorded frame.
    """
    description = play.get("game_str")
    if not isinstance(description, str) or not description:
        raise ValueError("Recorded play is missing the translated game description")
    dimensions = play.get("grid_size")
    if (
        not isinstance(dimensions, (list, tuple))
        or len(dimensions) != 2
        or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 1 for n in dimensions
        )
    ):
        raise ValueError("Recorded play grid_size must be [width, height], both > 1")
    if num_channels not in (1, 3):
        raise ValueError(f"Unsupported channel count requested: {num_channels}")

    import cv2
    import numpy as np
    from src.vgdl import VGDLParser

    sprite_order = VGDLParser().parse_game(description).sprite_order
    width, height = dimensions
    frames = []
    for state in states:
        objects = state["objects"]
        unknown = set(objects) - set(sprite_order)
        if unknown:
            raise ValueError(
                f"Recorded sprite types absent from translated game description: {sorted(unknown)}"
            )
        canvas = np.empty((height * 30, width * 30, 3), dtype=np.uint8)
        canvas[:] = np.asarray(background, dtype=np.uint8)
        for sprite_type in sprite_order:
            for sprite in objects.get(sprite_type, {}).values():
                rect = sprite["rect"]
                left, top = (int(value) for value in rect["pos"])
                rect_width, rect_height = (int(value) for value in rect["size"])
                canvas[top : top + rect_height, left : left + rect_width, :] = (
                    np.asarray(sprite["color"], dtype=np.uint8)
                )
        if resize and (canvas.shape[0] != resize or canvas.shape[1] != resize):
            canvas = cv2.resize(canvas, (resize, resize), interpolation=cv2.INTER_AREA)
        if num_channels == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        frames.append(canvas.astype(np.uint8))
    if not frames:
        raise ValueError("No recorded frames were provided")
    stacked = np.stack(frames, axis=0)
    return (
        stacked[:, np.newaxis, :, :]
        if num_channels == 1
        else stacked.transpose(0, 3, 1, 2)
    )
