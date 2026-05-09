# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
from src.vgdl.state import StateObserver, KeyValueObservation
import math

from typing import Union, List, Dict
from src.vgdl.interfaces.gym.env import BG_NAME, BG_COLOR
import numpy as np


class AvatarOrientedObserver(StateObserver):
    def _get_distance(self, s1, s2):
        return math.hypot(s1.rect.x - s2.rect.x, s1.rect.y - s2.rect.y)

    def get_observation(self):
        avatars = self.game.get_avatars()
        assert avatars
        avatar = avatars[0]

        avatar_pos = avatar.rect.topleft
        resources = [avatar.resources[r] for r in self.game.domain.notable_resources]

        sprite_distances = []
        for key in self.game.sprite_registry.sprite_keys:
            dist = 100
            for s in self.game.get_sprites(key):
                dist = min(self._get_distance(avatar, s) / self.game.block_size, dist)
            sprite_distances.append(dist)

        obs = KeyValueObservation(
            position=avatar_pos,
            speed=avatar.speed,
            resources=resources,
            distances=sprite_distances,
        )
        return obs


class NotableSpritesObserver(StateObserver):
    """
    TODO: There is still a problem with games where the avatar
    transforms into a different type
    """

    def __init__(self, game, notable_sprites: Union[List, Dict] = None):
        super().__init__(game)
        self.notable_sprites = notable_sprites or [
            s[0] for s in game.sprite_registry.groups() if s[0] != "floor"
        ]
        self.colors = {BG_NAME: BG_COLOR}
        self.resources_max_info = dict(
            zip(
                self.game.domain.notable_resources,
                [
                    self.game.domain.resources_limits.get(r_name, np.inf)
                    for r_name in self.game.domain.notable_resources
                ],
            )
        )
        self.base_symb_obs = None

    def get_observation(self):
        # self.notable_sprites = [s[0] for s in self.game.sprite_registry.groups() if s[0] != 'floor']
        sprite_keys = list(self.notable_sprites)
        resource_types = self.game.domain.notable_resources
        if self.base_symb_obs is None:
            symb_obs = [
                [[] for _ in range(self.game.height)] for _ in range(self.game.width)
            ]
            if "wall" in sprite_keys:
                key = "wall"
                for s in self.game.get_sprites(key):
                    pos = self._rect_to_pos(s.rect)
                    resources_dict = dict(
                        zip(
                            resource_types,
                            [float(s.resources.get(r, 0)) for r in resource_types],
                        )
                    )
                    symb_obs[int(pos[0])][int(pos[1])].append(
                        dict(
                            name=s.key,
                            color=s.color_str,
                            obj_id=s.id,
                            pos=pos,
                            resources=resources_dict,
                            resources_max=self.resources_max_info,
                        )
                    )
            self.base_symb_obs = symb_obs
        symb_obs = [[cell.copy() for cell in row] for row in self.base_symb_obs]
        if "wall" in sprite_keys:
            sprite_keys.remove("wall")

        for i, key in enumerate(sprite_keys):
            for s in self.game.get_sprites(key):
                pos = self._rect_to_pos(s.rect)
                resources_dict = dict(
                    zip(
                        resource_types,
                        [float(s.resources.get(r, 0)) for r in resource_types],
                    )
                )
                if not (
                    int(pos[0]) >= 0
                    and int(pos[1]) >= 0
                    and int(pos[0]) < len(symb_obs)
                    and int(pos[1]) < len(symb_obs[0])
                ):
                    print("pos", pos, s, self.game.game_desc)
                    self.game.sprite_registry.kill_sprite(s)
                    continue
                symb_obs[int(pos[0])][int(pos[1])].append(
                    dict(
                        name=s.key,
                        color=s.color_str,
                        obj_id=s.id,
                        pos=pos,
                        resources=resources_dict,
                        resources_max=self.resources_max_info,
                    )
                )

        return symb_obs

        # for i, key in enumerate(sprite_keys):
        #     # class_one_hot = [float(j==i) for j in range(num_classes)]
        #
        #     # TODO this code is currently unsafe as getSprites does not
        #     # guarantee the same order for each call (Python < 3.6),
        #     # meaning observations will have inconsistent ordering of values
        #     for s in self.game.get_sprites(key):
        #         position = self._rect_to_pos(s.rect)
        #         if hasattr(s, 'orientation'):
        #             orientation = [float(a) for a in s.orientation]
        #         else:
        #             orientation = [0.0, 0.0]
        #
        #         resources = [float(s.resources.get(r, 0)) for r in resource_types]
        #         color = s.color_str
        #         name = s.key
        #         state += [
        #             (s.id + f'.{color}.position', position),
        #             (s.id + '.orientation', orientation),
        #             (s.id + '.class', class_one_hot),
        #             (s.id + '.resources', resources),
        #         ]

        return KeyValueObservation(state)
