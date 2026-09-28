# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
import pygame
from pygame.math import Vector2

from environments.vgdl.render import SpriteLibrary
from environments.vgdl.ontology.constants import RIGHT, BLACK, GOLD

import numpy as np


class OffscreenRenderer:
    """Draw experimental image inputs without an SDL display or event loop."""

    def __init__(self, game, block_size, render_sprites=True):
        self.game = game
        # In pixels
        self.block_size = block_size
        self.screen = pygame.Surface(self.screen_dims, depth=32)
        self.render_sprites = render_sprites
        if self.render_sprites:
            self.sprite_cache = SpriteLibrary.default()

    @property
    def screen_dims(self):
        return (self.game.width * self.block_size, self.game.height * self.block_size)

    def draw_all(self):
        self.screen.fill((200, 200, 200))

        for s in self.game.sprite_registry.sprites():
            self.draw_sprite(s)

    def calculate_render_rect(self, rect, shrinkfactor=0):
        displacement_factor = self.block_size / max(rect.size)
        sprite_rect = pygame.Rect(
            Vector2(rect.topleft) * displacement_factor,
            (self.block_size, self.block_size),
        )
        if shrinkfactor != 0:
            sprite_rect = sprite_rect.inflate(
                *(-Vector2(sprite_rect.size) * shrinkfactor)
            )

        return sprite_rect

    def draw_sprite(self, sprite):
        sprite_rect = self.calculate_render_rect(sprite.rect, sprite.shrinkfactor)
        if self.render_sprites and sprite.img:
            # assert sprite.shrinkfactor == 0, 'TODO implement shrinking sprites'
            block_size = int((1 - sprite.shrinkfactor) * self.block_size)
            img = self.sprite_cache.get_sprite_of_size(sprite.img, block_size)

            if hasattr(sprite, "orientation"):
                # Assume by default images face right
                img_orientation = sprite.img_orient or RIGHT
                angle = img_orientation.angle_to(sprite.orientation)
                if abs(angle / 180) == 1:
                    # A flip will likely look nicer than a rotate
                    img = pygame.transform.flip(img, True, False)
                else:
                    img = pygame.transform.rotate(img, -angle)

            self.screen.blit(img, sprite_rect)
        else:
            self.screen.fill(sprite.color, sprite_rect)

        if sprite.resources:
            self.draw_resources(sprite, sprite_rect)

    def draw_resources(self, sprite, rect):
        """Draw progress bars on the bottom third of the sprite"""
        tot = len(sprite.resources)
        barheight = rect.height / 3.5 / tot
        offset = rect.top + 2 * rect.height / 3.0
        for r in sorted(sprite.resources.keys()):
            wiggle = rect.width / 10.0
            limit = self.game.domain.resources_limits.get(r, 1)
            prop = max(0, min(1, sprite.resources[r] / float(limit)))
            if prop != 0:
                filled = pygame.Rect(
                    rect.left + wiggle / 2,
                    offset,
                    prop * (rect.width - wiggle),
                    barheight,
                )
                rest = pygame.Rect(
                    rect.left + wiggle / 2 + prop * (rect.width - wiggle),
                    offset,
                    (1 - prop) * (rect.width - wiggle),
                    barheight,
                )
                self.screen.fill(self.game.domain.resources_colors.get(r, GOLD), filled)
                self.screen.fill(BLACK, rest)
                offset += barheight

    def get_image(self):
        return np.flipud(
            np.rot90(pygame.surfarray.array3d(self.screen).astype(np.uint8))
        )
