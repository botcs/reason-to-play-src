"""Train DDQN with explicit local game and output directories."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--trial_num", default=1, type=int, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--lr", default=0.00025, type=float, required=False)
    parser.add_argument("--gamma", default=0.999, type=float, required=False)
    parser.add_argument("--eps_start", default=1, type=float, required=False)
    parser.add_argument("--eps_end", default=0.01, type=float, required=False)
    parser.add_argument("--eps_decay", default=10000, type=int, required=False)
    parser.add_argument("--target_update", default=100, type=int, required=False)
    parser.add_argument("--steps_per_level", default=100000, type=int, required=False)
    parser.add_argument("--max_mem", default=50000, type=int, required=False)
    parser.add_argument("--model_name", default="DQN", type=str, required=False)
    parser.add_argument("--model_weight_path", type=str, required=False)
    parser.add_argument("--test_mode", default=0, type=int, required=False)
    parser.add_argument("--pretrain", default=0, type=int, required=False)
    parser.add_argument("--cuda", default=1, type=int, required=False)
    parser.add_argument("--doubleq", default=1, type=int, required=False)
    parser.add_argument(
        "--level_switch", default="sequential", type=str, required=False
    )
    parser.add_argument("--timeout", default=2000, type=int, required=False)
    parser.add_argument("--game_name", default="aliens", type=str, required=False)
    parser.add_argument("--random_seed", default=7, type=int, required=False)
    parser.add_argument(
        "--use_grid_representation",
        default=1,
        type=int,
        required=False,
        help="Use one-hot grid representation (1) or pixel representation (0)",
    )
    parser.add_argument(
        "--frame_stack_size",
        default=4,
        type=int,
        required=False,
        help="Number of frames to stack for temporal information",
    )
    parser.add_argument(
        "--clip_rewards",
        default=0,
        type=int,
        required=False,
        help="Clip rewards to [-1, 1] range (1) or use original rewards (0)",
    )
    parser.add_argument(
        "--wandb_project",
        default="ddqn-vgdl",
        type=str,
        required=False,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", required=False, help="Disable wandb logging"
    )
    parser.add_argument(
        "--save_video",
        default=0,
        type=int,
        required=False,
        help="Save gameplay videos (1) or not (0)",
    )
    parser.add_argument(
        "--grad_clip",
        default=10.0,
        type=float,
        required=False,
        help="Gradient clipping threshold (0 to disable)",
    )
    parser.add_argument(
        "--augment",
        default="random",
        type=str,
        required=False,
        help='Data augmentation: None, "rotate90", "rotate180", "rotate270", "hflip", "vflip", or "random"',
    )

    parser.add_argument(
        "--game-dir",
        type=Path,
        default=Path(__file__).parent / "environment/training/all_games",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("ddqn-output"))
    parser.add_argument("--threads", type=int, default=32)
    return parser


def main():
    config = build_parser().parse_args()
    if config.threads < 1:
        raise ValueError("--threads must be positive")
    import torch
    from agents.ddqn.player import Player

    torch.set_num_threads(config.threads)
    print(config)

    config.file_names = str(config.game_dir.resolve())

    print("Game: {}".format(config.game_name))
    print("Random seed: {}".format(config.random_seed))

    # Initialize wandb
    if not config.no_wandb:
        import wandb

        tags = [
            config.game_name,
            config.level_switch,
            f"seed{config.random_seed}",
            f"lr{config.lr}",
            f"gamma{config.gamma}",
            f"batch{config.batch_size}",
            f"mem{config.max_mem}",
            f"target{config.target_update}",
            f"decay{config.eps_decay}",
            f"frames{config.frame_stack_size}",
        ]
        if config.augment:
            tags.append(f"aug{config.augment}")
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            tags=tags,
        )

    game_player = Player(config)
    game_player.train_model()

    if not config.no_wandb:
        wandb.finish()

    print("Done training!")


if __name__ == "__main__":
    main()
