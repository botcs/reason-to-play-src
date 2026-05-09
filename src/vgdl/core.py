# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
import copy
import inspect
import logging
import random
from collections import defaultdict, UserDict, deque
from functools import partial
from typing import NewType, Optional, Union, Dict, List, Tuple, Generator
import pygame
import pygame.key
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_SPACE
from pygame.math import Vector2

from .tools import PrettyDict, PrettyClass, freeze_dict
from copy import deepcopy
import numpy as np
from src.vgdl.registration import registry

Color = NewType("Color", Tuple[int, int, int])


class SpriteRegistry:
    """
    A registry that knows of all types of sprites from a game description,
    and all the sprite instances during the life of a game.

    TODO: may need to split this into separate class and instance registries.

    Lingo:
        key - main identifier
        stypes - main identifier and any parent types
    """

    def __init__(self):
        # Sprite classes, a class is identified by its key
        self.classes: Dict[str, type] = {}
        self.class_args: Dict[str, List] = {}
        self.stypes: Dict[str, List] = {}
        self.sprite_keys: List[str] = []

        # Populated by parser
        self.singletons: List[str] = []

        # All sprite instances, each has a unique id
        self._sprite_by_id = {}
        # Sprites are grouped by their primary stype, but they can have more
        # More work to keep separate dicts but work is amortized
        self._live_sprites_by_key = defaultdict(list)
        self._dead_sprites_by_key = defaultdict(list)

    def reset(self):
        self._live_sprites_by_key.clear()
        self._dead_sprites_by_key.clear()
        self._sprite_by_id.clear()

    def register_singleton(self, key):
        self.singletons.append(key)

    def register_sprite_class(self, key, cls, args, stypes):
        assert key not in self.sprite_keys, f"Sprite key already registered: {key}"
        assert cls is not None
        self.classes[key] = cls
        self.class_args[key] = args
        self.stypes[key] = stypes
        self.sprite_keys.append(key)

    def get_sprite_defs(self):
        for key in self.sprite_keys:
            yield key, self.get_sprite_def(key)

    def get_sprite_def(self, key):
        try:
            return self.classes[key], self.class_args[key], self.stypes[key]
        except KeyError:
            raise KeyError("Unknown sprite type '%s', verify your domain file" % key)

    def generate_id_number(self, key):
        ids = [
            int(sprite.id.split(".")[-1]) for sprite in self._live_sprites_by_key[key]
        ] + [int(sprite.id.split(".")[-1]) for sprite in self._dead_sprites_by_key[key]]
        if len(ids) > 0:
            count = np.max(ids)
        else:
            count = 0
        # count = len(self._live_sprites_by_key[key]) + len(self._dead_sprites_by_key[key])
        return count + 1

    def generate_id(self, key):
        # Singletons always get the same id
        # if self.is_singleton(key):
        #     n = 0
        # else:
        n = self.generate_id_number(key)
        return "{}.{}".format(key, n)

    def is_singleton(self, key):
        return key in self.singletons

    def create_sprite(self, key, id=None, **kwargs):
        # TODO fix rng

        if self.is_singleton(key) and self._live_sprites_by_key[key]:
            return None
        sclass, args, stypes = self.get_sprite_def(key)
        id = id or self.generate_id(key)

        sprite = sclass(key=key, id=id, **{**args, **kwargs})
        sprite.stypes = stypes
        # print(key, stypes)

        self._live_sprites_by_key[key].append(sprite)
        self._sprite_by_id[id] = sprite
        return sprite

    def kill_sprite(self, sprite: "VGDLSprite"):
        """Kills a sprite while keeping track of it"""
        if sprite is not self._sprite_by_id[sprite.id]:
            logging.debug("Unknown sprite %s (id=%s)", sprite, id(sprite))
        sprite.alive = False
        if sprite in self._live_sprites_by_key[sprite.key]:
            self._live_sprites_by_key[sprite.key].remove(sprite)
            self._dead_sprites_by_key[sprite.key].append(sprite)

    def revive_sprite(self, sprite):
        sprite.alive = True
        if sprite is not self._sprite_by_id[sprite.id]:
            logging.debug("Unknown sprite %s (id=%s)", sprite, id(sprite))
        # if sprite not in self._dead_sprites_by_key[sprite.key]:
        #     import ipdb; ipdb.set_trace()
        self._dead_sprites_by_key[sprite.key].remove(sprite)
        self._live_sprites_by_key[sprite.key].append(sprite)

    def destroy_sprite(self, sprite):
        """
        Destroys sprite based on id, erase all traces.
        Potentially useful when replacing a sprite
        """
        if sprite.id in self._sprite_by_id:
            del self._sprite_by_id[sprite.id]
            self._live_sprites_by_key[sprite.key] = [
                s
                for s in self._live_sprites_by_key[sprite.key]
                if not s.id == sprite.id
            ]
            self._dead_sprites_by_key[sprite.key] = [
                s
                for s in self._dead_sprites_by_key[sprite.key]
                if not s.id == sprite.id
            ]
            return True
        return False

    # def groups(self, include_dead=False) -> Dict[str, List['VGDLSprite']]:
    def groups(
        self, include_dead=False
    ) -> Generator[Tuple[str, List["VGDLSprite"]], None, None]:
        if not include_dead:
            # TODO not sure if include_dead is worth making this a generator
            for k, v in self._live_sprites_by_key.items():
                yield k, v
            # return self._live_sprites_by_key.items()
        else:
            assert set(self._live_sprites_by_key).issuperset(self._dead_sprites_by_key)
            for key in self._live_sprites_by_key.keys():
                yield (
                    key,
                    self._live_sprites_by_key[key] + self._dead_sprites_by_key[key],
                )

    def group(self, key, include_dead=False) -> List["VGDLSprite"]:
        if not include_dead:
            return self._live_sprites_by_key[key]
        else:
            return self._live_sprites_by_key[key] + self._dead_sprites_by_key[key]

    def sprites(self, include_dead=False):
        assert include_dead is False
        # Order sprites by definition
        for key in self.sprite_keys:
            sprites = self._live_sprites_by_key[key]
            for sprite in sprites:
                yield sprite

    def with_stype(self, stype, include_dead=False):
        """
        It is possible for a sprite to have more than one stype. The main one is used as the key.
        This matches all sprites with a given stype, not just the stype as key.
        """
        if stype in self.sprite_keys:
            return self.group(stype, include_dead)
        else:
            sprites = []
            for key in self.sprite_keys:
                if stype in self.stypes[key]:
                    sprites += self._live_sprites_by_key[key]
            return sprites
            # return [s for _, sprites in self.groups(include_dead) for s in sprites if stype in s.stypes]

    def get_avatar(self):
        """
        Returns the game single avatar, fails if a different amount are present
        """
        res = []

        for _, ss in self.groups(include_dead=True):
            if ss and self.is_avatar(ss[0]):
                return ss[0]

    def defs_with_class(self, target):
        defs = []
        for stype, cls in self.classes.items():
            if any(
                target.__name__ == cls_name
                for cls_name in (parent.__name__ for parent in inspect.getmro(cls))
            ):
                # cls descends from target
                defs.append((stype, cls, self.class_args[stype]))
        return defs

    def issubclass(self, target, cls):
        return any(
            cls.__name__ == cls_name
            for cls_name in (parent.__name__ for parent in inspect.getmro(target))
        )

    def is_avatar(self, sprite):
        return self.is_avatar_cls(sprite.__class__)

    def is_avatar_cls(self, cls):
        return any(
            "Avatar" in cls_name
            for cls_name in (parent.__name__ for parent in inspect.getmro(cls))
        )

    # def should_save(self, key):
    #     is_immutable = self.issubclass(self.classes[key], Immutable)
    #     return not is_immutable

    def saveable_keys(self):
        return self.sprite_keys
        # return [key for key in self.sprite_keys if self.should_save(key)]

    def get_state(self) -> dict:
        def _sprite_state(sprite):
            return dict(id=sprite.id, state=sprite.get_game_state())

        sprite_states = {}
        # for sprite_type, sprites in self._live_sprites_by_key.items():
        for sprite_type in self.saveable_keys():
            sprites = (
                self._live_sprites_by_key[sprite_type]
                + self._dead_sprites_by_key[sprite_type]
            )
            # Do not save Immutables. Immutables are always alive, etc.
            sprite_states[sprite_type] = [
                _sprite_state(sprite) for sprite in sprites
            ]  # \
            # if not isinstance(sprite, Immutable)]
        # for sprite_type, sprites in self._dead_sprites_by_key.items():
        # for sprite_type in self.saveable_keys():
        #     sprites = self._dead_sprites_by_key[sprite_type]
        #     sprite_states[sprite_type] += [_sprite_state(sprite) for sprite in sprites]

        return sprite_states

    def set_state(self, state: dict, block_size):
        """
        - Overwrite the state of matching sprites (id-wise)
        - Remove sprites that are not in the new state
        - Add sprites that are in the new state

        Overwriting really is an unnecessary optimisation. All it gives us is
        last references to objects that do not get added, such as the avatar.
        """
        # It is possible less sprites are saved than we know of
        assert set(self.sprite_keys).issuperset(state.keys()), (
            "Known sprite keys should match"
        )

        # Is assumed to not hold any Immutables
        other_ids = {sprite["id"] for sprites in state.values() for sprite in sprites}
        # Do not consider Immutables, and expect that they were not saved.
        known_ids = {
            id
            for id, sprite in self._sprite_by_id.items()
            if not isinstance(sprite, Immutable)
        }
        deleted_ids = known_ids.difference(other_ids)
        added_ids = other_ids.difference(known_ids)

        for known_id in known_ids:
            self.destroy_sprite(self._sprite_by_id[known_id])

        for key, sprites in state.items():
            for sprite_state in sprites:
                self.reinstate_sprite(key, sprite_state, block_size)
        # TODO reinstate smarter state keeping
        # if len(deleted_ids) > 0:
        #     for key in self.sprite_keys:
        #         self._live_sprites_by_key[key] = [sprite for sprite in self._live_sprites_by_key[key] \
        #                                           if not sprite.id in deleted_ids]
        #         self._dead_sprites_by_key[key] = [sprite for sprite in self._dead_sprites_by_key[key] \
        #                                           if not sprite.id in deleted_ids]
        #     for deleted_id in deleted_ids:
        #         self._sprite_by_id.pop(deleted_id)

        # if len(added_ids) > 0:
        #     pass

        # for key, sprite_states in state.items():
        #     for sprite_state in sprite_states:
        #         id = sprite_state['id']
        #         if id in self._sprite_by_id:
        #             sprite = self._sprite_by_id[id]
        #             known_alive = sprite.alive
        #             if not known_alive and sprite not in self._dead_sprites_by_key[sprite.key]:
        #                 import ipdb; ipdb.set_trace()
        #             sprite.set_game_state(sprite_state['state'])
        #             if known_alive and not sprite.alive:
        #                 self.kill_sprite(sprite)
        #             elif not known_alive and sprite.alive:
        #                 self.revive_sprite(sprite)
        #         else:
        #             # Including pos here because I don't like allowing position-less sprites
        #             sprite = self.create_sprite(key, id, pos=sprite_state['state']['rect'].topleft)
        #             sprite.set_game_state(sprite_state['state'])

    def reinstate_sprite(self, key, sprite_state: "SpriteState", block_size):
        """
        Reinserts a sprite that may come from a saved state.
        """
        sprite = self.create_sprite(
            key,
            sprite_state["id"],
            pos=sprite_state["state"]["rect"].topleft,
            size=block_size,
        )
        sprite.set_game_state(sprite_state["state"])
        if not sprite.alive:
            self.kill_sprite(sprite)

    def assert_sanity(self):
        live = set(s.id for ss in self._live_sprites_by_key.values() for s in ss)
        dead = set([s.id for ss in self._dead_sprites_by_key.values() for s in ss])
        if len(live.intersection(dead)) > 0:
            print("not sane")
            import ipdb

            ipdb.set_trace()

    def copy(self):
        sprite_registry = SpriteRegistry()
        sprite_registry.classes = self.classes.copy()
        args_copies = []
        for arg in self.class_args.values():
            arg_copy = arg.copy()
            if "resource" in arg_copy:
                arg_copy["resource"] = arg["resource"].copy()
        sprite_registry.class_args = dict(zip(self.class_args.keys(), args_copies))
        sprite_registry.stypes = dict(
            zip(self.stypes.keys(), [el.copy() for el in self.stypes.values()])
        )
        sprite_registry.sprite_keys = self.sprite_keys.copy()
        sprite_registry.singletons = self.singletons.copy()
        sprite_registry._sprite_by_id = dict(
            zip(
                self._sprite_by_id.keys(),
                [el.copy() for el in self._sprite_by_id.values()],
            )
        )
        for sprite in sprite_registry._sprite_by_id.values():
            sprite_registry._live_sprites_by_key[sprite.key] = sprite


class SpriteState(PrettyDict, UserDict):
    # TODO be careful comparing SpriteStates, some attributes in _effect_data contain
    # timestamps that would cause equality to fail where we would want it to succeed
    # Either do not save in form of timestamp, or do time-sensitive equality check
    def norm_time_hash(self, time, notable_resources):
        """
        This relies on the HEAVY assumption that timestamp keys start with  't_'
        """
        overwrite = {}
        if "_effect_data" in self.data:
            effect_data = []
            # onceperstep events
            for k, v in self.data["_effect_data"].items():
                if k == "t_last_touched_ladder":
                    effect_data.append((k, v >= time - 1))
                elif k.startswith("t_"):
                    effect_data.append((k, v >= time))
                else:
                    effect_data.append((k, v))
                # elif isinstance(v, int):
                #     effect_data.append((k, v == time))
                # else:
                #     effect_data.append((k, v))
            # This should overwrite the original, absolute _effect_data
            overwrite["_effect_data"] = effect_data

        if "resources" in self.data:
            resources = {k: self.data["resources"].get(k, 0) for k in notable_resources}
            overwrite["resources"] = resources

        # TODO analyse unnecessary copy
        return freeze_dict({**self.data, **overwrite})


class GameState(UserDict):
    def __init__(self, game, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notable_resources = game.domain.notable_resources
        self.frozen = None
        self.hashed = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["frozen"]
        del state["hashed"]
        return state

    def __setstate__(self, state):
        self.data = state["data"]
        self.notable_resources = state.get("notable_resources", [])
        self.frozen = None
        self.hashed = None

    @property
    def avatar_state(self):
        return self.data["sprites"]["avatar"][0]

    def ended(self):
        return self.data["ended"]

    def get_reward(self):
        return self.data["last_reward"]

    def freeze(self):
        # Return cached frozen game state, since game states don't change
        if self.frozen is not None:
            return self.frozen

        # Take into account time, want to have time-insensitive equality
        time = self["time"]
        self.frozen = freeze_dict(
            self.data["sprites"],
            {
                SpriteState: partial(
                    SpriteState.norm_time_hash,
                    time=time,
                    notable_resources=self.notable_resources,
                )
            },
        )
        return self.frozen

    def hash(self):
        if self.hashed is None:
            self.hashed = hash(self.freeze())

        return self.hashed

    def __eq__(self, other):
        """Game state equality, should ignore time etc"""
        # return self.freeze() == other.freeze()
        if other is None:
            return False
        return self.hash() == other.hash()

    def __hash__(self):
        """Game state equality is based on sprite state"""
        return self.hash()

    def __lt__(self, other):
        # return self.data['time'] < other.data['time']
        return self.avatar_state["state"]["rect"] < other.avatar_state["state"]["rect"]

    def __repr__(self):
        """Assume single-avatar"""
        avatar_state = self.avatar_state["state"]
        return (
            "GameState(time={time}, score={score}, reward={last_reward}, ended={ended}, "
            "avatar=(pos={rect.topleft}, alive={alive}))"
        ).format(**self.data, **avatar_state)


class Action:
    """
    Actions are based on pygame keys, even though we do not actually
    use those keys. They're a handy leftover to vectorise actions
    and make thinking with them and programming game dynamics easier.
    """

    def __init__(self, *args):
        self.keys = tuple(sorted(args))

    def as_vector(self):
        """
        Directional keys are used to encode directions.
        Opposite directions cancel eachother out.
        """
        return Vector2(
            -1 * (K_LEFT in self.keys) + 1 * (K_RIGHT in self.keys),
            -1 * (K_UP in self.keys) + 1 * (K_DOWN in self.keys),
        )

    def __str__(self):
        _key_name = lambda k: (
            pygame.key.name(k) if pygame.key.name(k) != "unknown key" else str(k)
        )
        key_rep = ",".join(_key_name(k) for k in self.keys)
        if len(self.keys) <= 1:
            return key_rep or "noop"
        else:
            return "[{}]".format(key_rep)

    def __repr__(self):
        _key_name = lambda k: (
            pygame.key.name(k) if pygame.key.name(k) != "unknown key" else str(k)
        )
        key_rep = ",".join(_key_name(k) for k in self.keys)
        return "Action({})".format(key_rep or "noop")

    def __eq__(self, other):
        # if not hasattr(other, 'keys'):
        #     return False
        return isinstance(other, Action) and self.keys == other.keys

    def __hash__(self):
        return hash(self.keys)

    @classmethod
    def from_vectors(*v):
        pass


# Probably ACTION_UP is the cleaner way to code this?
class ACTION:
    NOOP = Action()
    UP = Action(pygame.K_UP)
    DOWN = Action(pygame.K_DOWN)
    RIGHT = Action(pygame.K_RIGHT)
    LEFT = Action(pygame.K_LEFT)
    SPACE = Action(pygame.K_SPACE)
    SPACE_RIGHT = Action(pygame.K_SPACE, pygame.K_RIGHT)
    SPACE_LEFT = Action(pygame.K_SPACE, pygame.K_LEFT)


class BasicGame:
    """
    Represents a game domain, with sprite types, transition dynamics (effects),
    and reward/termination conditions.
    """

    MAX_SPRITES = 10000

    def __init__(self, sprite_registry, title=None, block_size=1, seed=None, **kwargs):
        # TODO split the registries perhaps in domain/task?
        # For now, just keep an unused copy for the domain
        self.domain_registry = sprite_registry
        self.title = title
        # Evaluate seed at call time (not import time) so np.random.seed() is respected
        self.seed = seed if seed is not None else int(np.random.randint(1e6))
        # Completely 2D worlds really just need a block size of 1
        self.block_size = block_size

        for name, value in kwargs.items():
            print("WARNING: undefined parameter '%s' for game! " % (name))

        self.notable_resources: List[str] = []

        # z-level of sprite types (in case of overlap), populated by parser
        self.sprite_order = []
        # collision effects (ordered by execution order)
        self.collision_eff = []
        # for reading levels
        # TODO DEPRECATED defaults, backwards compatibility
        self.char_mapping = {}  # {'w': ['wall'], 'A': ['avatar']}
        # termination criteria
        self.terminations = []
        # resource properties, used to draw resource bar on the avatar sprite
        # TODO get rid of defaults
        # defaultdicts don't pickle well (it's the lambdas)
        # self.resources_limits = defaultdict(lambda: 1)
        self.resources_limits = {}

        # self.resources_colors = defaultdict(lambda: GOLD)
        self.resources_colors = {}

    def finish_setup(self):
        """
        Called when the parser is done populating the game
        """
        self.is_stochastic = any(e.is_stochastic for e in self.collision_eff)

        self.setup_resources()

        # Sprites with stype 'avatar' but not as main key won't work here
        if "avatar" in self.sprite_order:
            self.sprite_order.remove("avatar")
            self.sprite_order.append("avatar")

    def setup_resources(self):
        self.notable_resources.clear()

        for res_type, (sclass, args, _) in self.domain_registry.get_sprite_defs():
            if issubclass(sclass, Resource):
                # TODO use a more OO approach, alas need to instantiate Resource
                if "res_type" in args:
                    res_type = args["res_type"]
                if "color" in args:
                    self.resources_colors[res_type] = args["color"]
                if "limit" in args:
                    self.resources_limits[res_type] = args["limit"]
                self.notable_resources.append(res_type)

    def build_level(self, lstr):
        # TODO delegate this to a level parser
        lines = [l for l in lstr.split("\n") if len(l) > 0]
        lengths = [len(l) for l in lines]
        assert min(lengths) == max(lengths), "Inconsistent line lengths."

        level = BasicGameLevel(
            self,
            copy.deepcopy(self.domain_registry),
            lstr,
            width=lengths[0],
            height=len(lines),
            seed=self.seed,
        )

        # create sprites
        for row, l in enumerate(lines):
            for col, c in enumerate(l):
                key = self.char_mapping.get(c, None)
                if key is not None:
                    pos = (col * self.block_size, row * self.block_size)
                    level.create_sprites(key, pos)

        # TODO find a prettier way to drop this, should be after creating
        # sprites though
        level.init_state = level.get_game_state()

        return level

    def identity_dict(self):
        """
        Meant for __eq__ and __hash__, returns attributes that identify a
        BasicGame domain without level
        """
        import dill

        # TODO cache this?
        return dict(
            block_size=self.block_size,
            effects=tuple(dill.dumps(effect) for effect in self.collision_eff),
            terminations=tuple(dill.dumps(term) for term in self.terminations),
            # This summarises the domain. Careful, dill doesn't serialise class
            # definitions, so code changes won't be reflected.
            classes=dill.dumps(self.domain_registry.classes),
            class_args=dill.dumps(self.domain_registry.class_args),
        )

    def identity(self):
        return tuple(self.identity_dict().values())

    def __hash__(self):
        return hash(self.identity())

    def __eq__(self, other):
        return self.identity() == other.identity()


class BasicGameLevel:
    """
    Represents a game task and a game domain.

    Minor reliance on pygame for collision detection,
    hence we use pygame's integer rectangles.
    Beware, only integer-sized and positioned rectangles possible.

    This regroups all the components of a game's dynamics, after parsing.
    """

    def __init__(
        self,
        domain: BasicGame,
        sprite_registry,
        levelstring,
        width,
        height,
        seed=0,
        title=None,
    ):
        self.domain = domain
        self.sprite_registry = sprite_registry
        self.levelstring = levelstring
        self.width = width
        self.height = height
        self.block_size = domain.block_size
        self.screensize = (self.width * self.block_size, self.height * self.block_size)
        self.title = title

        # Random state
        self.seed = seed
        self.random_generator = random.Random(self.seed)

        # Can add sprites to this queue to update during this tick
        self.update_queue = deque()

        ### Below this is state keeping
        # used for erasing dead sprites
        self.kill_list = []
        self.create_list = []
        self.resource_changes = []
        # Accumulated reward
        self.score = 0
        self.last_reward = 0
        self.time = 0
        self.ended = False
        self.won = False
        self.is_stochastic = False
        self.last_state = None

    def set_seed(self, seed):
        self.seed = seed
        self.random_generator.seed(self.seed)

    def __repr__(self):
        if self.title is not None:
            return "{} `{}`".format(self.__class__.__name__, self.title)
        else:
            return "{}".format(self.__class__.__name__)

    def identity(self):
        """
        Meant for __eq__ and __hash__, returns attributes that identify a
        a level completely
        """
        # TODO careful this includes seed, re-visit this when we reconsider
        # stochastic games more carefully
        # Really I think seed should be part of game state or something
        # return dict(
        #     domain=self.domain.identity(),
        #     levelstring=self.levelstring,
        #     # seed=self.seed,
        # )
        return (
            self.domain.identity(),
            self.levelstring,
        )

    def __hash__(self):
        """
        Domain- and level-sensitive hash. Ignores GameState.
        """
        return hash(self.identity())

    def reset(self):
        """
        Resets the environment. If a level is known, revert to its initial state.
        """
        self.score = 0
        self.last_reward = 0
        self.time = 0
        self.ended = False
        self.won = False
        self.kill_list.clear()
        self.create_list.clear()
        self.resource_changes.clear()
        if self.init_state:
            self.set_game_state(self.init_state)
        self.last_state = None
        self.update_queue.clear()
        # self.random_generator.seed(self.seed)
        self.events_triggered = []

    def quick_reset(self):
        self.score = 0
        self.last_reward = 0
        self.time = 0
        self.ended = False
        self.won = False
        self.kill_list.clear()
        self.create_list.clear()
        self.resource_changes.clear()
        self.last_state = None
        self.update_queue.clear()
        self.events_triggered = []
        self.is_stochastic = False

    def round(self, number, decimal=4):
        bigger = int(number * 10**decimal)
        if str(bigger)[-1] != "0":
            lower = bigger
            higher = bigger
            while True:
                lower = lower - 1
                higher = higher + 1
                if str(lower)[-1] == "0":
                    return lower / (10**decimal)
                elif str(higher)[-1] == "0":
                    return higher / (10**decimal)
        else:
            return number

    def quick_build_and_set_state(self, hidden_state):
        timee = hidden_state["i_step"]
        state_info = hidden_state["obj_hidden_state"]
        self.width, self.height = hidden_state["shape"]
        # all_names = hidden_state['names']

        state_dict = {
            "last_reward": 0,
            "kill_list": [],
            "create_list": [],
            "resource_changes": [],
            "time": timee,
            "ended": False,
            "score": 0,
        }

        # update sprites with state_info here
        sprite_registry = SpriteRegistry()
        sprite_registry.classes = self.sprite_registry.classes
        sprite_registry.class_args = self.sprite_registry.class_args
        sprite_registry.stypes = self.sprite_registry.stypes
        sprite_registry.singletons = self.sprite_registry.singletons
        sprite_registry.sprite_keys = self.sprite_registry.sprite_keys

        # set wall objs
        for id, sprite in self.sprite_registry._sprite_by_id.items():
            if sprite.key == "wall":
                sprite_registry._sprite_by_id[id] = sprite
                sprite_registry._live_sprites_by_key[sprite.key].append(sprite)

        self.quick_reset()
        # keys = {'floor'}
        poss = []
        for obj in state_info:
            sclass = registry.request(obj["type"])
            args = obj["params"]
            stypes = obj["name"]
            id = obj["obj_id"]
            key = obj["name"]
            pos = (
                self.round(obj["pos"][0] * self.block_size, decimal=1),
                self.round(obj["pos"][1] * self.block_size, decimal=1),
            )
            poss.append(pos[1] / self.block_size)
            size = (self.block_size, self.block_size)
            img = obj["img"]
            rng = self.random_generator
            sprite = sclass(
                key=key,
                id=id,
                pos=pos,
                size=size,
                rng=rng,
                img=img,
                resources=obj["resources"].copy(),
                **args,
            )
            sprite.stypes = stypes
            sprite_registry._live_sprites_by_key[key].append(sprite)
            sprite_registry._sprite_by_id[id] = sprite
            self.is_stochastic = (
                self.domain.is_stochastic or sprite and sprite.is_stochastic
            )
        self.quick_set_game_state(state_dict, sprite_registry)

    def get_hidden_state(self, return_orientation=False):
        hidden_state = dict(
            i_step=self.time,
            obj_hidden_state=[],
            shape=(self.width, self.height),
            agent_orientation=dict(),
        )
        for key in self.sprite_registry.sprite_keys:
            for s in self.sprite_registry._live_sprites_by_key[key]:
                params = s.__dict__.copy()
                for k in [
                    "key",
                    "name",
                    "id",
                    "pos",
                    "size",
                    "rng",
                    "img",
                    "resources",
                ]:
                    if k in params.keys():
                        del params[k]
                obj = dict(
                    type=s.__class__.__name__,
                    params=params,
                    obj_id=s.id,
                    name=key,
                    resources=s.resources.copy(),
                    img=s.img,
                    pos=(s.rect.x // self.block_size, s.rect.y // self.block_size),
                )
                hidden_state["obj_hidden_state"].append(obj)
                if return_orientation and "Avatar" in s.__class__.__name__:
                    if "orientation" in params.keys():
                        hidden_state["agent_orientation"][s.id] = params["orientation"]
        return hidden_state

    # def __getstate__(self):
    #     """
    #     It is recommended to save a level and a game state separately.
    #     # TODO: decide on how this behaves.. right now I suggest
    #     to save game and state separately, but not doing so still works
    #     """
    #     d = self.__dict__.copy()
    #     d['gamestate'] = self.get_game_state()
    #     return d
    #
    # def __setstate__(self, state):
    #     gamestate = state.pop('gamestate')
    #     self.__dict__.update(state)
    #     self.set_game_state(gamestate)

    def create_sprite(self, key, pos, id=None) -> Optional["VGDLSprite"]:
        # assert self.num_sprites < self.domain.MAX_SPRITES, 'Sprite limit reached'

        sclass, args, stypes = self.sprite_registry.get_sprite_def(key)

        sprite = self.sprite_registry.create_sprite(
            key,
            pos=pos,
            id=id,
            size=(self.block_size, self.block_size),
            rng=self.random_generator,
        )
        self.is_stochastic = (
            self.domain.is_stochastic
            or sprite
            and sprite.is_stochastic
            or self.is_stochastic
        )
        return sprite

    def create_sprites(self, keys, pos) -> List["VGDLSprite"]:
        # Splitting it makes mypy happy
        filter_nones = lambda l: filter(lambda el: el, l)
        return list(filter_nones(self.create_sprite(key, pos) for key in keys))

    def kill_sprite(self, sprite: "VGDLSprite"):
        self.kill_list.append(sprite)
        # self.sprite_registry.kill_sprite(sprite)

    def add_sprite_creation(self, key, pos, id=None):
        self.create_list.append((key, pos, id))

    def destroy_sprite(self, sprite):
        self.kill_list.append(sprite)
        self.sprite_registry.destroy_sprite(sprite)

    def num_sprites(self, key):
        """Abstract groups are computed on demand only"""
        num = len(self.sprite_registry.with_stype(key))
        return num

    def get_sprites(self, key):
        return self.sprite_registry.with_stype(key)

    def get_avatars(self):
        """The currently alive avatar(s)"""
        res = []

        for _, ss in self.sprite_registry.groups(include_dead=True):
            if ss and self.is_avatar(ss[0]):
                res.extend(ss)

        return res

    def is_avatar(self, sprite):
        return self.is_avatar_cls(sprite.__class__)

    def is_avatar_cls(self, cls):
        return any(
            "Avatar" in cls_name
            for cls_name in (parent.__name__ for parent in inspect.getmro(cls))
        )

    def get_game_state(self, include_random_state=False) -> GameState:
        # Return cached state
        # if self.last_state is not None:
        #     return self.last_state

        state_dict = {
            "last_reward": self.last_reward,
            "sprites": self.sprite_registry.get_state(),
            "kill_list": self.kill_list,
            "create_list": self.create_list,
            "resource_changes": self.resource_changes,
            "time": self.time,
            "ended": self.ended,
            "score": self.score,
        }

        state = GameState(self, state_dict)
        self.last_state = state

        return state

    def set_game_state(self, state: GameState):
        """
        Rebuilds all sprites and resets game state
        TODO: Keep a sprite registry and even keep dead sprites around,
        just overwrite their state when setting game state.
        This has the advantage of keeping the Python objects intact.
        """
        # Careful, state is mutable but really shouldn't be
        self.last_state = None
        self.sprite_registry.set_state(state.get("sprites"), self.block_size)
        for k, v in state.items():
            if k in ["sprites"]:
                continue
            setattr(self, k, v)

    def quick_set_game_state(self, state, sprite_registry):
        self.last_state = None
        self.sprite_registry = sprite_registry
        for k, v in state.items():
            if k in ["sprites"]:
                continue
            setattr(self, k, v)

    def _move_event_handling(self):
        all_events_triggered = []
        all_events_triggered_keys = []
        self.lastcollisions: Dict[str, List["VGDLSprite"]] = {}
        ss = self.lastcollisions

        # apply stepbacks
        stepback_events = [
            eff
            for eff in self.domain.collision_eff
            if eff.name in ("stepBack", "stepBackIfHasLess")
        ]
        for effect in stepback_events:
            ss, events_triggered, events_triggered_keys = self.apply_effect(effect, ss)
            all_events_triggered += events_triggered
            all_events_triggered_keys += events_triggered_keys

        # apply movements
        move_events = [
            eff
            for eff in self.domain.collision_eff
            if eff.name in ["bounceForward", "reverseDirection", "turnAround"]
        ]
        for effect in move_events:
            ss, events_triggered, events_triggered_keys = self.apply_effect(effect, ss)
            all_events_triggered += events_triggered
            all_events_triggered_keys += events_triggered_keys

        # reapply stepbacks
        stepback_events = [
            eff
            for eff in self.domain.collision_eff
            if eff.name in ("stepBack", "stepBackIfHasLess")
        ]
        for effect in stepback_events:
            ss, events_triggered, events_triggered_keys = self.apply_effect(effect, ss)
            all_events_triggered += events_triggered
            all_events_triggered_keys += events_triggered_keys

        return ss, all_events_triggered, all_events_triggered_keys

    def apply_effect(self, effect, ss):
        events_triggered = []
        events_triggered_keys = []
        g1 = effect.actor_stype
        g2 = effect.actee_stype

        # build the current sprite lists (if not yet available)
        for g in [g1, g2]:
            if g not in ss:
                sprites = self.sprite_registry.with_stype(g)
                ss[g] = sprites

        # special case for EOS=end-of-screen
        if g2 == "EOS":
            ss1 = ss[g1]
            for index in range(len(ss1) - 1, -1, -1):
                s1 = ss1[index]
                if not self.contains_rect(s1.rect):
                    try:
                        event_key = (
                            effect.name,
                            s1.key,
                            "EOS",
                            (s1.rect.x, s1.rect.y),
                            (None, None),
                        )
                        self.add_score(effect.score)
                        try:
                            effect(s1, None, self)
                            events_triggered.append((effect.name, s1.id, "EOS"))
                            events_triggered_keys.append(event_key)
                        except:
                            pass
                        if not self.contains_rect(s1.rect) and s1.alive:
                            # if after the effect the sprite is out, kill it
                            self.kill_sprite(s1)
                    except Exception as e:
                        print(e)
                        import ipdb

                        ipdb.set_trace()
            return ss, events_triggered, events_triggered_keys

        # special case for any tile, as long as it's in the game window still
        if g2 == "ANY":
            assert False
            # for s1 in self.lastcollisions[g1]:
            #     if not self.contains_rect(s1.rect):
            #         continue
            #     try:
            #         event_key = (effect.name, sprite.key, None, (sprite.rect.x, sprite.rect.y), None)
            #         if event_key not in past_events_keys:
            #             self.add_score(effect.score)
            #             effect(s1, None, self)
            #             events_triggered.append((effect.name, sprite.id, None))
            #             events_triggered_keys.append(event_key)
            #         assert False, 'check when this happens??'
            #     except Exception as e:
            #         logging.critical("Exception during effect {}".format(effect))
            # continue
        sprites = ss[g1]
        others = ss[g2]
        if len(sprites) == 0 or len(others) == 0:
            return ss, events_triggered, events_triggered_keys

        if len(sprites) > len(others):
            sprites, others = others, sprites
            reverse = True
        else:
            reverse = False

        for sprite in sprites:
            # if sprite not in self.kill_list:
            for collision_i in sprite.rect.collidelistall(others):
                other = others[collision_i]
                # if other not in self.kill_list:
                if sprite is other:
                    continue
                elif sprite == other:
                    assert False, "Huh, interesting"
                if reverse:
                    if other not in self.kill_list:
                        reverse_event_key = (
                            effect.name,
                            other.key,
                            sprite.key,
                            (other.rect.x, other.rect.y),
                            (sprite.rect.x, sprite.rect.y),
                        )
                        self.add_score(effect.score)
                        effect(other, sprite, self)
                        events_triggered.append((effect.name, other.id, sprite.id))
                        events_triggered_keys.append(reverse_event_key)
                else:
                    if sprite not in self.kill_list:
                        event_key = (
                            effect.name,
                            sprite.key,
                            other.key,
                            (sprite.rect.x, sprite.rect.y),
                            (other.rect.x, other.rect.y),
                        )
                        self.add_score(effect.score)
                        effect(sprite, other, self)
                        events_triggered.append((effect.name, sprite.id, other.id))
                        events_triggered_keys.append(event_key)
        return ss, events_triggered, events_triggered_keys

    def _event_handling(self, ss):
        all_events_triggered = []
        all_events_triggered_keys = []
        non_move_effects = [
            eff
            for eff in self.domain.collision_eff
            if eff.name
            not in [
                "stepBack",
                "stepBackIfHasLess",
                "bounceForward",
                "reverseDirection",
                "turnAround",
            ]
        ]

        for effect in non_move_effects:
            ss, events_triggered, events_triggered_keys = self.apply_effect(effect, ss)
            all_events_triggered += events_triggered
            all_events_triggered_keys += events_triggered_keys
        # are we done or should we look for other interactions?
        return all_events_triggered, all_events_triggered_keys

    def add_score(self, score):
        self.score += score
        self.last_reward += score

    def get_possible_actions(self, include_noop=True) -> Dict[Tuple[int], Action]:
        """
        Assume actions don't change

        This used to return a Dict[str, Action],
        but I think it's a hassle to upkeep the strings.
        """
        try:
            # My version of issubclass because it can't be trusted
            avatar_cls = next(
                cls
                for cls in self.sprite_registry.classes.values()
                if self.is_avatar_cls(cls)
            )
        except StopIteration:
            print(
                [
                    parent.__name__
                    for parent in inspect.getmro(self.sprite_registry.classes["avatar"])
                ]
            )
            raise Exception("No avatar class registered")

        # Alternatively, use pygame names for keys instead of the key codes
        # pygame_keys = {k: v for k, v in vars(pygame).items() if k.startswith('K_')}
        # action_dict = avatar_cls.declare_possible_actions()

        actions = {}
        actions["UP"] = Action(K_UP)
        actions["DOWN"] = Action(K_DOWN)
        actions["LEFT"] = Action(K_LEFT)
        actions["RIGHT"] = Action(K_RIGHT)
        actions["NO_OP"] = Action()
        actions["SPACE"] = Action(K_SPACE)

        return {a.keys: a for a in actions.values() if include_noop or a != ACTION.NOOP}

    def tick(self, action: Union[Action, int]):
        """
        Actions are currently communicated to the rest of the program
        through game.keystate, which mimics pygame.key.get_pressed().
        Maybe one beautiful day we can step away from that.
        It's a leftover but it works and it focuses on designing
        games that are human playable. Key presses are easy to reason about,
        even if we do not actually use them.
        """
        # if isinstance(action, int):
        #     action = Action(action)
        # assert action in self.get_possible_actions().values(), \
        #     'Illegal action %s, expected one of %s' % (action, self.get_possible_actions())
        # if isinstance(action, int):
        #     action = Action(action)
        # This is required for game-updates to work properly
        self.time += 1

        # Careful, self.last_reward should not be used _during_ a `tick`
        self.last_reward = 0

        if self.ended:
            # logging.warning('Action performed while game ended')
            return

        # Update Keypresses
        # Agents are updated during the update routine in their ontology files,
        # this depends on BasicGame.active_keys
        self.active_keys = action.keys

        # # Clear last turn's kill list, used by the renderer to clear sprites
        # # Things can die during update and subsequent _eventHandling,
        # # so this clears previous tick's kill list
        # self.kill_list.clear()
        # self.resource_changes.clear()

        # Update Sprites
        # By default a newly created sprite won't get updated until next tick
        # Sometimes you want instantaneous results, so add sprite to the live
        # update queue
        self.update_queue.clear()
        for s in self.sprite_registry.sprites():
            s.just_pushed = None  # reset memory of being pushed
            self.update_queue.append(s)
        # print('h1', time.time() - t_init)
        # t_init = time.time()
        # While loop because it can keep growing, for loops are fickle
        while self.update_queue:
            s = self.update_queue.popleft()
            s.update(self)

        # Handle Collision Effects
        self.events_triggered = []
        ss, events_triggered, events_triggered_keys = self._move_event_handling()
        new_events_triggered, new_events_triggered_keys = self._event_handling(ss)
        events_triggered += new_events_triggered
        events_triggered_keys += new_events_triggered_keys
        self.events_triggered += events_triggered
        # Iterate Over Termination Criteria
        # kill sprites
        for sprite in self.kill_list:
            self.sprite_registry.kill_sprite(sprite)
        for sprite_info in self.create_list:
            self.create_sprite(*sprite_info)
        # update resources now
        for resource_change in self.resource_changes:
            sprite, resource, value = resource_change
            sprite.resources[resource] = max(
                0,
                min(
                    sprite.resources[resource] + value,
                    self.domain.resources_limits.get(resource, float("+inf")),
                ),
            )
        self._check_terminations()
        self.kill_list.clear()
        self.create_list.clear()
        self.resource_changes.clear()
        self.last_state = None

    def _check_terminations(self):
        self.lose = False
        for t in self.domain.terminations:
            self.ended, self.won = t.is_done(self)
            if t.__class__.__name__ in ("Timeout", "SpriteCounter", "MultiSpriteCounter"):
                # Any ended-but-not-won termination is a loss.  Previously
                # Timeout was special-cased to leave ``self.lose = False``,
                # which caused the agent loop to keep stepping past a
                # timed-out game and corrupted replay files (step records
                # say attempt=0 while the state sequence shows a hidden
                # mid-block reset).
                if self.ended and not self.won:
                    self.lose = True
            else:
                raise NotImplementedError
            if self.won is None:
                self.won = False
            if self.ended:
                # Terminations are allowed to specify a score
                self.add_score(t.score)
                break

    def colliding_sprites(
        self, sprite, stype=None
    ) -> Generator["VGDLSprite", None, None]:
        if stype is not None:
            sprites = self.sprite_registry.with_stype(stype)
        else:
            sprites = self.sprite_registry.sprites()

        for other in sprites:
            if sprite == other:
                continue
            if sprite.rect.colliderect(other.rect):
                yield other

    def sprites_at(self, topleft: Tuple, stype=None) -> Generator:
        if stype is not None:
            sprites = self.sprite_registry.with_stype(stype)
        else:
            sprites = self.sprite_registry.sprites()

        for other in sprites:
            if other.rect.topleft == topleft:
                yield other

    def contains_rect(self, rect: pygame.Rect):
        game_rect = pygame.Rect(
            (0, 0), self.screensize
        )  # (self.screensize[0] + self.block_size, self.screensize[1] + self.block_size))
        return game_rect.contains(rect)

    def contains_position(self, pos: Union[Tuple, Vector2]):
        """
        Checks if a standard rectangle at that position is in the game screen.
        So rectangles half falling off the screen are considered not to be contained.
        """
        rect = pygame.Rect(pos, (self.block_size, self.block_size))
        return self.contains_rect(rect)

    def is_freshly_killed(self, stype: str) -> bool:
        """
        Mind you, this is non-Markov and is not fully tested yet,
        as it relies on `self.kill_list`.
        """
        return bool(
            set(self.kill_list).intersection(
                self.sprite_registry.with_stype(stype, include_dead=True)
            )
        )


class VGDLSprite:
    """Base class for all sprite types."""

    COLOR_DISC = [20, 80, 140, 200]

    is_static = False
    only_active = False
    is_avatar = False
    is_stochastic = False
    color = None  # type: Optional[Color]
    cooldown = 0  # pause ticks in-between two moves
    speed = None  # type: Optional[int]
    mass = 1
    physicstype = None  # type: type
    shrinkfactor = 0.0
    just_pushed = False

    state_attributes = [
        "rect",
        "alive",
        "resources",
        "speed",
        "orientation",
        "_age",
        "lastmove",
    ]

    def __init__(
        self,
        key,
        id,
        pos,
        size=(1, 1),
        color=None,
        speed=None,
        cooldown=None,
        physicstype=None,
        random_generator=None,
        **kwargs,
    ):
        if not isinstance(size, tuple):
            size = (size, size)

        # Every sprite must have a key, an id, and a position
        self.key: str = key
        self.id: str = id
        self.rect = pygame.Rect(pos, size)
        self.lastrect = self.rect
        self.alive = True

        from .ontology import GridPhysics

        self.physicstype = physicstype or self.physicstype or GridPhysics
        self.physics = self.physicstype(size)
        self.speed = speed or self.speed
        self.cooldown = cooldown or self.cooldown
        self.img = None
        self.img_orient = None
        # TODO rng
        self.color = color or self.color

        # TODO re-evaluate whether this is useful
        # To be populated by events, should be cleared on reset
        self._effect_data = {}

        # how many timesteps ago was the last move?
        self.lastmove = 0

        # management of resources contained in the sprite
        self.resources = defaultdict(int)

        for name, value in kwargs.items():
            try:
                self.__dict__[name] = value
            except:
                print(
                    "WARNING: undefined parameter '%s' for sprite '%s'! "
                    % (name, self.__class__.__name__)
                )
        if self.img is not None:
            self.color_str = self.img.split("/")[1]

        try:
            if isinstance(self.stype, tuple):
                stop = 1
        except:
            pass

    def copy(self):
        sprite = VGDLSprite(self.key, self.id, (self.rect.x, self.rect.y))
        sprite_dict = self.__dict__.copy()
        for key, v in sprite_dict.items():
            if isinstance(v, dict) or isinstance(v, list):
                sprite_dict[key] = self.__dict__[key].copy()
        sprite.__dict__ = sprite_dict
        return sprite

    def get_game_state(self) -> SpriteState:
        state = {}
        for attr_name in self.state_attributes:
            if hasattr(self, attr_name):
                v = getattr(self, attr_name)
                if isinstance(v, dict) or isinstance(v, list):
                    state[attr_name] = deepcopy(v)
                else:
                    try:
                        state[attr_name] = v.copy()
                    except:
                        state[attr_name] = v
        # The alternative is to have each class define _just_ its state attrs,
        # flatten(c.state_attributes for c in inspect.getmro(self.__class__) \
        #   if c.hasattr('state_attributes'))
        state["_effect_data"] = self._effect_data.copy()
        return SpriteState(state)

    def set_game_state(self, state: SpriteState):
        # self._effect_data.clear()
        # self._effect_data.update(state.get('_effect_data'))
        self._effect_data = state["_effect_data"].copy()

        for k, v in state.items():
            if k in ["_effect_data"]:
                continue
            if hasattr(self, k):
                # Deep copy because v can definitely be altered (think resources)
                if isinstance(v, dict) or isinstance(v, list):
                    setattr(self, k, copy.deepcopy(v))
                else:
                    try:
                        setattr(self, k, v.copy())
                    except:
                        setattr(self, k, v)
            else:
                print(self)
                logging.warning("Unknown sprite state attribute `%s`", k)

    def update(self, game):
        """The main place where subclasses differ."""
        self.lastrect = self.rect
        # no need to redraw if nothing was updated

        self.lastmove += 1
        if not self.is_static and not self.only_active:
            self.physics.passive_movement(self)

    def _update_position(self, orientation, speed=None):
        # TODO use self.speed etc
        if isinstance(orientation, Action):
            import ipdb

            ipdb.set_trace()
        if speed is None:
            velocity = Vector2(orientation) * self.speed
        else:
            velocity = Vector2(orientation) * speed

        # if not(self.cooldown > self.lastmove or abs(orientation[0])+abs(orientation[1])==0):
        if self.lastmove >= self.cooldown:
            # TODO use self.velocity
            self.rect = self.rect.move(velocity)
            self.lastmove = 0

    @property
    def velocity(self) -> Vector2:
        """The velocity property is made up of the orientation and speed attributes"""
        if self.speed is None or self.speed == 0 or not hasattr(self, "orientation"):
            return Vector2(0, 0)
        else:
            return Vector2(self.orientation) * self.speed

    @velocity.setter
    def velocity(self, v):
        v = Vector2(v)
        self.speed = v.length()
        # Orientation is of unit length except when it isn't
        if self.speed == 0:
            self.orientation = Vector2(0, 0)
        else:
            self.orientation = v.normalize()

    @property
    def lastdirection(self) -> Vector2:
        return Vector2(self.rect.topleft) - Vector2(self.lastrect.topleft)

    def count_resource(self, resource):
        return self.resources.get(resource, 0)

    def has_resource(self, resource) -> bool:
        return bool(self.count_resource(resource))

    def __repr__(self):
        return "{} `{}` at ({}, {})".format(self.key, self.id, *self.rect.topleft)


class Avatar:
    """Abstract superclass of all avatars."""

    shrinkfactor = 0.15

    def __init__(self):
        raise NotImplementedError("Abstract base class Avatar")
        self.actions = Avatar.declare_possible_actions()


class Resource(VGDLSprite):
    """A special type of object that can be present in the game in two forms, either
    physically sitting around, or in the form of a counter inside another sprite."""

    value = 1
    limit = 2
    res_type = None

    state_attributes = VGDLSprite.state_attributes + ["limit"]

    @property
    def resource_type(self):
        if self.res_type is None:
            return self.key
        else:
            return self.res_type


class Immutable(VGDLSprite):
    """
    Class for sprites that we do not have to bother saving.
    It's a simple performance improvement but it seems to improve things a lot.
    """

    is_static = True

    def get_game_state(self):
        return SpriteState(dict(alive=self.alive))

    def set_game_state(self, state):
        self.alive = state["alive"]

    def _update_position(self):
        raise Exception("Tried to move Immutable")

    def update(self, game):
        return


class Termination(PrettyClass):
    def __init__(self, win, scoreChange=0):
        self.win = win
        self.score = scoreChange

    """ Base class for all termination criteria. """

    def is_done(self, game):
        """returns whether the game is over, with a win/lose flag"""
        from pygame.locals import K_ESCAPE, QUIT

        if K_ESCAPE in game.active_keys or pygame.event.peek(QUIT):
            return True, False
        else:
            return False, None


class Effect:
    """
    Effects are called during event handling, which is collision-based.
    An effect will only be called for sprites that match
    the actor and actee (acted-upon) stypes.
    """

    is_stochastic = False

    def __init__(self, actor_stype, actee_stype, scoreChange=0):
        self.actor_stype = actor_stype
        self.actee_stype = actee_stype
        self.score = scoreChange

    def __call__(self, sprite, partner, game):
        raise NotImplementedError


class FunctionalEffect(Effect):
    """
    DEPRECATED.

    Old-style effect, implemented with a function.
    The parser will use this when it finds effects that are functions.
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, scoreChange=kwargs.pop("scoreChange", 0))
        self.call_fn = fn
        self.fn_args = kwargs
        self.name = self.call_fn.__name__

    def __call__(self, sprite, partner, game):
        return self.call_fn(sprite, partner, game, **self.fn_args)

    @property
    def is_stochastic(self):
        # This is all old-style, assume nothing gets added to stochastic_effects
        from .ontology import stochastic_effects

        return self.call_fn in stochastic_effects


class Physics:
    def __init__(self, gridsize):
        self.gridsize = gridsize
        if gridsize[0] == 1:
            stop = 1


# TODO move this somewhere pretty
# This allows both copy and pickle to work with pygame stuff
import copyreg


def _pickle_vector(v):
    return Vector2, (v.x, v.y)


def _pickle_rect(r):
    return pygame.Rect, (*r.topleft, r.width, r.height)


copyreg.pickle(Vector2, _pickle_vector)
copyreg.pickle(pygame.Rect, _pickle_rect)


registry.register_class(Immutable)
