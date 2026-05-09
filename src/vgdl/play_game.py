# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
import gym
from gym.envs.registration import register
import pygame
import pickle
import os

games_path = "./games/"


def register_game(game, level=0, fast=False):
    game_id = f"{game}-v0"
    game_path = games_path + "/"
    register(
        id=game_id,
        entry_point="vgdl.interfaces.gym:VGDLEnv",
        kwargs={
            "game_file": os.path.join(game_path, game + ".txt"),
            "level_file": os.path.join(game_path, game + f"_lvl{level}.txt"),
            "obs_type": "objects",
            "block_size": 1 if fast else 55,
        },
    )

    return game_id


def play_game(game_name="sokoban", save_to=None):
    game_id = register_game(game_name)
    env = gym.make(game_id)
    # env.render()
    obs_0, info = env.reset()
    env.render()

    trajectory = dict(
        states=[], actions=[], rewards=[], done=[], obs_symb=[info["obs_symb"]], won=[]
    )
    score = 0
    step = 0
    while True:
        run, action_id = get_action_from_key(env)
        if not run:
            break
        if action_id is not None:
            state, reward, isOver, truncated, info = env.step(action_id)
            for k, data in zip(
                trajectory.keys(),
                [state, action_id, reward, isOver, info["obs_symb"], info["won"]],
            ):
                trajectory[k].append(data)
            score += reward
            print(
                "Action "
                + str(action_id)
                + " played at game tick "
                + str(step + 1)
                + ", reward="
                + str(reward)
                + ", new score="
                + str(score)
            )
            env.render()
            step += 1
            if isOver:
                if info["won"]:
                    print("You won!")
                else:
                    print("You lost!")
                break
    # save trajectory
    if save_to is not None:
        with open(save_to, "wb") as f:
            pickle.dump(trajectory, f)


def get_action_from_key(env):
    run, action_id = True, None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                action_id = 1  # action_mapping['ACTION_DOWN']
            elif event.key == pygame.K_UP:
                action_id = 0  # action_mapping['ACTION_UP']
            elif event.key == pygame.K_LEFT:
                action_id = 2  # action_mapping['ACTION_LEFT']
            elif event.key == pygame.K_RIGHT:
                action_id = 3  # action_mapping['ACTION_RIGHT']
            elif event.key == pygame.K_SPACE:
                action_id = None  # action_mapping['ACTION_USE']
            else:
                action_id = None
    return run, action_id


if __name__ == "__main__":
    play_game()
