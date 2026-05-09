# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
import os
from collections import OrderedDict
import gym
from gym import spaces
import src.vgdl
import numpy as np
from .list_space import list_space
from src.vgdl.core import Action
from src.vgdl.registration import registry
import time

BG_NAME = "floor"
AVATAR_NAME = "avatar"
BG_COLOR = "LIGHTGRAY"
# AVATAR_COLORS = ['RED', 'LIGHTRED']
# COLORS = ['BLUE', 'GRAY', 'WHITE', 'BROWN', 'ORANGE', 'YELLOW', 'PINK', 'GOLD', 'LIGHTORANGE', 'LIGHTBLUE', 'LIGHTGREEN', 'DARKBLUE']


class VGDLEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

    def __init__(self, game_file=None, level_file=None, obs_type="image", **kwargs):
        # For rendering purposes only
        self.render_block_size = kwargs.pop("block_size")
        self.block_size = 100
        self.ontology_registry = registry
        # Variables
        self._obs_type = obs_type
        self.viewer = None
        self.game_args = kwargs
        self.notable_sprites = kwargs.get("notable_sprites", None)

        # Load game description and level description
        if game_file is not None:
            with open(game_file, "r") as myfile:
                game_desc = myfile.read()
            with open(level_file, "r") as myfile:
                level_desc = myfile.read()
            self.level_name = os.path.basename(level_file).split(".")[0]
            self.loadGame(game_desc, level_desc, block_size=self.block_size)

    def loadGame(self, game_desc, level_desc, **kwargs):

        self.game_desc = game_desc
        self.level_desc = level_desc
        level_split = [l for l in level_desc.split("\n") if len(l) > 0]
        self.level_height = len(level_split)
        self.level_width = len(level_split[0])

        self.game_args.update(kwargs)

        # Need to build a sample level to get the available actions and screensize....
        domain = src.vgdl.VGDLParser().parse_game(self.game_desc, **self.game_args)
        self.game = domain.build_level(self.level_desc)
        self.game.game_desc = game_desc

        self.score_last = self.game.score

        # Set action space and observation space
        self._action_set = OrderedDict(self.game.get_possible_actions())
        self.action_space = spaces.Discrete(len(self._action_set))

        self.screen_width, self.screen_height = self.game.screensize

        if self._obs_type == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.screen_height, self.screen_width, 3)
            )
        elif self._obs_type == "objects":
            from .state import NotableSpritesObserver

            self.observer = NotableSpritesObserver(self.game, self.notable_sprites)
            self.observation_space = list_space(
                spaces.Box(low=-100, high=100, shape=(1,))
            )
            # shape=self.observer.observation_shape) )
        elif self._obs_type == "features":
            from .state import AvatarOrientedObserver

            self.observer = AvatarOrientedObserver(self.game)
            self.observation_space = spaces.Box(
                low=0, high=100, shape=self.observer.observation_shape
            )
        # elif isinstance(self._obs_type, type) and issubclass(self._obs_type, StateObserver):
        else:
            try:
                self.observer = self._obs_type(self.game)
                self.observation_space = spaces.Box(
                    low=0, high=100, shape=self.observer.observation_shape
                )
            except:
                raise Exception("Unknown obs_type `{}`".format(self._obs_type))

        # For rendering purposes, will be initialised by first `render` call
        self.renderer = None
        self.resources_max_info = dict(
            zip(
                self.game.domain.notable_resources,
                [
                    self.game.domain.resources_limits.get(r_name, np.inf)
                    for r_name in self.game.domain.notable_resources
                ],
            )
        )

    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def _action_keys(self):
        return list(self._action_set.values())

    def get_action_meanings(self):
        # In the spirit of the Atari environment, describe actions with strings
        return list(self._action_set.keys())

    def _get_obs(self, with_img=False):
        img = self.render(mode="rgb_array") if with_img else None
        # try:
        info = {
            "state": self.observer.get_observation(),
            "events_triggered": self.relabel_obj_ids_events(self.game.events_triggered),
        }
        # except:
        #     observation = self.observer.get_observation()
        #     info = {'state': self.src.vgdl2gvgai_format(observation),
        #             'events_triggered': self.relabel_obj_ids_events(self.game.events_triggered)}
        return img, info

    def relabel_obj_ids_events(self, events_triggered):
        for i in range(len(events_triggered)):
            obj_id1 = events_triggered[i][1]
            obj_id2 = events_triggered[i][2]
            if isinstance(obj_id1, tuple):
                obj_id1 = obj_id1[1]
            if isinstance(obj_id2, tuple):
                obj_id2 = obj_id2[1]
            try:
                name1, id1 = obj_id1.split(".")
            except:
                stop = 1
            if obj_id2 == "EOS":
                name2, id2 = "EOS", "EOS"
            else:
                name2, id2 = obj_id2.split(".")
            relabeled_event = (
                events_triggered[i][0],
                (name1, obj_id1),
                (name2, obj_id2),
            )
            events_triggered[i] = relabeled_event
        return events_triggered

    def vgdl2gvgai_format(self, obs):
        symb_obs = [
            [[] for _ in range(self.level_height)] for _ in range(self.level_width)
        ]
        for i, key, value in zip(range(len(obs)), obs.keys(), obs.values()):
            if "position" in key:
                name, ind, color = key.split(".")[:3]
                obj_id = name + "." + ind
                # try:
                resources_dict = dict(
                    zip(self.game.domain.notable_resources, obs[obj_id + ".resources"])
                )
                symb_obs[int(value[0])][int(value[1])].append(
                    dict(
                        name=name,
                        color=color,
                        obj_id=obj_id,
                        pos=value,
                        resources=resources_dict,
                        resources_max=self.resources_max_info,
                    )
                )
                # except:
                #     print(name, color, value, key)
                #     assert False
        return symb_obs

    def step(
        self,
        a,
        verbose=True,
        no_obs=False,
        with_img=False,
        return_state_pre_effect=False,
    ):
        # if not self.mode_initialised:
        #     raise Exception('Please call `render` at least once for initialisation')
        if a > len(self._action_keys) - 1 or a < 0:
            action = Action()
        else:
            action = self._action_keys[a]
        # ss = self.game.tick_obj_mvt(action)
        # if return_state_pre_effect:
        #     state_pre, info_pre = self._get_obs(with_img=False)
        # else:
        #     state_pre, info_pre = None, None
        # self.game.tick_effects_and_wrap(ss)
        self.game.tick(action)
        if no_obs:
            state = None
            info = {}
        else:
            state, info = self._get_obs(with_img=with_img)
        reward = self.game.last_reward
        terminal = self.game.ended
        info["won"] = self.game.won
        info["lose"] = self.game.lose
        info["reward"] = self.game.last_reward
        return state, reward, terminal, False, info

    def get_win_and_lose(self, hidden_state):
        self.set_state(hidden_state)
        self.game._check_terminations()
        return self.game.won, self.game.lose

    def simulate(self, hidden_states, action, n_simulations, with_img=False):
        states, rewards, isOvers, infos = [], [], [], []
        t_init = time.time()
        # print('time get', time.time() - t_init)
        time_step, time_reset = 0, 0
        for i in range(n_simulations):
            t_init = time.time()
            self.set_state(hidden_states[i])
            time_reset += time.time() - t_init
            t_init = time.time()
            try:
                state, reward, isOver, truncated, info = self.step(
                    action, with_img=with_img
                )
                info["reward"] = reward
            except:
                state, reward, isOver, truncated, info = [None] * 5
            # print('    ', time.time() - t_init)
            states.append(state.copy() if state is not None else state)
            rewards.append(reward)
            isOvers.append(isOver)
            infos.append(info)
            time_step += time.time() - t_init
        # print('time_reset', time_reset)
        # print('time step', time_step)
        return states, rewards, isOvers, infos

    def reset(self, with_img=False):
        # TODO improve the reset with the new domain split
        self.game.reset()
        # self.game = self.game.doma
        # in.build_level(self.level_desc)
        self.score_last = self.game.score
        state, info = self._get_obs(with_img=with_img)
        return state, info

    def render(self, mode="human", close=False):
        headless = mode != "human"

        if self.renderer is None:
            from src.vgdl.render import PygameRenderer

            self.renderer = PygameRenderer(self.game, self.render_block_size)
            self.renderer.init_screen(headless)

        self.renderer.draw_all()
        self.renderer.update_display()

        if close:
            self.renderer.close()
        if mode == "rgb_array":
            img = self.renderer.get_image()
            return img
        elif mode == "human":
            return True

    def close(self):
        self.renderer.close()

    def set_state(self, hidden_state, return_obs=False, with_img=False):
        self.level_width, self.level_height = hidden_state["shape"]
        self.game.quick_build_and_set_state(hidden_state)
        if return_obs:
            obs = self._get_obs(with_img=with_img)
            return obs

    def get_state(self, return_orientation=False):
        return self.game.get_hidden_state(return_orientation=return_orientation)


class Padlist(gym.ObservationWrapper):
    def __init__(self, env=None, max_objs=200):
        self.max_objects = max_objs
        super(Padlist, self).__init__(env)
        env_shape = self.observation_space.shape
        env_shape[0] = self.max_objects
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=env_shape)

    def _observation(self, obs):
        return Padlist.process(obs, self.max_objects)

    @staticmethod
    def process(input_list, to_len):
        max_len = to_len
        item_len = len(input_list)
        if item_len < max_len:
            padded = np.pad(
                np.array(input_list, dtype=np.float32),
                ((0, max_len - item_len), (0, 0)),
                mode="constant",
            )
            return padded
        else:
            return np.array(input_list, dtype=np.float32)[:max_len]
