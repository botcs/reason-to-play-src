# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python


def list_envs():
    import gym

    env_names = list(gym.envs.registry.env_specs.keys())
    return env_names


def main():
    envs = list_envs()
    for env in envs:
        print(env)


if __name__ == "__main__":
    main()
