#!/usr/bin/env python3
"""
Extract model (GridDQN) activations aligned with behavioral states.

Input: Canonical behavior/human JSON recordings and local model checkpoints.
Output: NPZ files with per-frame activations and original behavioral timestamps.
The included extraction environment and model work in source and wheel installs.
Checkpoint downloads are opt-in.

Usage:
    python -m agents.ddqn.extract_features \
        --subject sub-01 \
        --run 1 \
        --model-id ddqn-trial1-sequential \
        --checkpoint-dir ./checkpoints \
        --behavior-dir ./dataset/behavior/human \
        --output-dir ./features/ddqn
"""

from __future__ import annotations

import argparse
import logging
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from human.behavior import load_behavioral_data, play_states

DEFAULT_RC_RL_DIR = Path(__file__).parent / "environment" / "extraction"
RC_RL_EXTRACTION_REVISION = "8f5f8a3c6facba2962319ea4892396057061370c"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "DDQN extraction requires PyTorch; install a suitable PyTorch build "
            "or reason-to-play[ddqn]."
        ) from exc
    return torch


def resolve_baseline_directory(rc_rl_dir=None):
    """Resolve the installed extraction environment, independently of the CWD."""
    return Path(rc_rl_dir or DEFAULT_RC_RL_DIR).resolve()


def baseline_source_provenance(rc_rl_dir):
    """Verify the included environment against its pinned source manifest."""
    rc_rl_dir = Path(rc_rl_dir).resolve()
    manifest = rc_rl_dir / "PROVENANCE.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text())
        files = record.get("files", [])
        if not files:
            raise ValueError(f"Empty baseline source manifest: {manifest}")
        recorded = set()
        for entry in files:
            relative = entry["path"]
            path = (rc_rl_dir / relative).resolve()
            if not path.is_relative_to(rc_rl_dir) or relative in recorded:
                raise ValueError(f"Invalid baseline source path: {relative}")
            recorded.add(relative)
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Baseline source checksum mismatch: {relative}")
        unrecorded_code = [
            str(path.relative_to(rc_rl_dir))
            for path in rc_rl_dir.rglob("*.py")
            if str(path.relative_to(rc_rl_dir)) not in recorded
        ]
        if unrecorded_code:
            raise ValueError(f"Unrecorded Python files in baseline: {unrecorded_code}")
        return {
            "revision": record["commit"],
            "dirty": False,
            "distribution": "curated-vendor",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
    raise ValueError(f"Missing DDQN environment source manifest: {manifest}")


def load_baseline(rc_rl_dir=None):
    """Load the included extraction engine and checkpoint-compatible model."""
    directory = resolve_baseline_directory(rc_rl_dir)
    if directory != DEFAULT_RC_RL_DIR.resolve():
        raise ValueError("Use the included DDQN extraction environment")
    provenance = baseline_source_provenance(directory)
    sources_path = Path(__file__).with_name("sources.json")
    sources = json.loads(sources_path.read_text())
    model_record = next(item for item in sources["files"] if item["path"] == "model.py")
    if (
        hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest()
        != model_record["sha256"]
    ):
        raise ValueError("DDQN model source checksum mismatch")
    provenance["manifest_sha256"] = hashlib.sha256(
        (
            provenance["manifest_sha256"]
            + hashlib.sha256(sources_path.read_bytes()).hexdigest()
        ).encode()
    ).hexdigest()
    from agents.ddqn.environment.extraction.VGDLEnv import VGDLEnv
    from agents.ddqn.model import GridDQN

    return VGDLEnv, GridDQN, provenance


# TODO: Make this configurable via YAML or auto-discover from wandb sweep
WANDB_CHECKPOINTS = {
    "bait": {
        "run_id": "555zfxvb",
        "pattern": "model_weights/vgfmri4_bait_trial1_sequential_{level}.pt",
    },
    "avoidgeorge": {
        "run_id": "n7zigwsk",
        "pattern": "model_weights/vgfmri4_avoidgeorge_trial1_sequential_{level}.pt",
    },
    "chase": {
        "run_id": "d4a2bgga",
        "pattern": "model_weights/vgfmri4_chase_trial1_sequential_{level}.pt",
    },
    "helper": {
        "run_id": "iyj2pekc",
        "pattern": "model_weights/vgfmri4_helper_trial1_sequential_{level}.pt",
    },
    "lemmings": {
        "run_id": "y320497g",
        "pattern": "model_weights/vgfmri4_lemmings_trial1_sequential_{level}.pt",
    },
    # 'plaqueAttack': {
    #     'run_id': 'eb2msw31',
    #     'pattern': 'model_weights/vgfmri3_plaqueAttack_trial1_sequential_{level}.pt'
    # },
    "zelda": {
        "run_id": "ucd6wqxk",
        "pattern": "model_weights/vgfmri4_zelda_trial1_sequential_{level}.pt",
    },
}
WANDB_PROJECT = "dpag-rl/ddqn-vgdl"


class ModelFeatureExtractor:
    """Extract activations from all layers of a GridDQN model."""

    def __init__(
        self,
        checkpoint_path: Path,
        game_name: str,
        level: int,
        frame_stack_size: int = 8,
        device: str = "cpu",
        rc_rl_dir: Path = DEFAULT_RC_RL_DIR,
    ):
        """
        Initialize model feature extractor.

        Args:
            checkpoint_path: Path to model checkpoint file
            game_name: Game name (e.g., 'vgfmri4_bait')
            level: Level number
            frame_stack_size: Number of frames to stack
            device: Device to run on ('cpu' or 'cuda')
        """
        self.checkpoint_path = checkpoint_path
        self.game_name = game_name
        self.level = level
        self.frame_stack_size = frame_stack_size
        rc_rl_dir = resolve_baseline_directory(rc_rl_dir)
        torch = _torch()
        self.device = torch.device(device)

        VGDLEnv, self.model_class, self.source_provenance = load_baseline(rc_rl_dir)
        self.checkpoint_sha256 = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        game_folder = str(Path(rc_rl_dir) / "all_games")
        self.env = VGDLEnv(game_name, game_folder)
        self.env.set_level(level)

        game = self.env.current_env._game
        sprite_types = list(game.sprite_groups.keys())
        self.sprite_type_mapping = {st: i for i, st in enumerate(sorted(sprite_types))}
        self.num_object_types = len(sprite_types)
        self.game_height = game.height
        self.game_width = game.width
        self.block_size = game.block_size

        self.n_actions = 6
        self.input_channels = frame_stack_size * self.num_object_types

        logging.info(f"  Object types: {self.num_object_types}")
        logging.info(f"  Input channels: {self.input_channels}")
        logging.info(f"  Grid size: {self.game_height}x{self.game_width}")

        self.model = self._load_model()
        self.model.eval()

        self.layer_activations = {}
        self.hook_handles = []
        self._register_hooks()

    def _load_model(self):
        """Load GridDQN model from checkpoint."""
        torch = _torch()
        model = self.model_class(self.input_channels, self.n_actions).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def _create_hook(self, layer_name: str):
        """Create forward hook to capture layer activation."""

        def hook(module, input, output):
            activation = output.detach().cpu()

            if len(activation.shape) == 4:
                activation = activation.view(activation.shape[0], -1)

            activation = activation.squeeze(0)

            if layer_name not in self.layer_activations:
                self.layer_activations[layer_name] = []
            self.layer_activations[layer_name].append(activation)

        return hook

    def _register_hooks(self):
        """Register forward hooks on all layers."""
        torch = _torch()
        conv_idx = 0
        fc_idx = 0

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_idx += 1
                layer_name = f"conv{conv_idx}"
                handle = module.register_forward_hook(self._create_hook(layer_name))
                self.hook_handles.append(handle)
                logging.info(f"  Registered hook: {layer_name}")

            elif isinstance(module, torch.nn.Linear):
                fc_idx += 1
                if fc_idx == 2:
                    layer_name = "q_values"
                else:
                    layer_name = f"fc{fc_idx}"
                handle = module.register_forward_hook(self._create_hook(layer_name))
                self.hook_handles.append(handle)
                logging.info(f"  Registered hook: {layer_name}")

    def create_grid_from_state(self, state: dict) -> torch.Tensor:
        """
        Create one-hot grid representation from decompressed state.

        Args:
            state: Original-style sprite state from the canonical recording reader

        Returns:
            torch.Tensor of shape (num_object_types, height, width)
        """
        torch = _torch()
        grid = torch.zeros(
            (self.num_object_types, self.game_height, self.game_width),
            dtype=torch.float32,
            device=self.device,
        )

        objects = state.get("objects", {})
        for sprite_type, sprite_dict in objects.items():
            if sprite_type in self.sprite_type_mapping:
                type_idx = self.sprite_type_mapping[sprite_type]
                for pos_str, sprite_attrs in sprite_dict.items():
                    x = sprite_attrs.get("x", 0)
                    y = sprite_attrs.get("y", 0)

                    grid_x = int(x / self.block_size)
                    grid_y = int(y / self.block_size)

                    if 0 <= grid_x < self.game_width and 0 <= grid_y < self.game_height:
                        grid[type_idx, grid_y, grid_x] = 1.0

        return grid

    def extract_features_from_play(self, play_doc: dict) -> dict:
        """
        Extract model activations for a single play episode.

        Args:
            play_doc: Document from plays collection

        Returns:
            Dict with activations and metadata
        """
        torch = _torch()
        self.layer_activations = {}

        play_id = play_doc["_id"]
        subj_id = play_doc["subj_id"]
        game_name = play_doc["game_name"]
        level_id = play_doc["level_id"]
        run_id = play_doc.get("run_id")
        human_win = play_doc.get("win")
        human_score = play_doc.get("score")

        states = play_states(play_doc)
        state_timestamps = [state.get("ts", 0) for state in states]

        frame_buffer = []

        for state in states:
            current_frame = self.create_grid_from_state(state)

            frame_buffer.append(current_frame)
            if len(frame_buffer) > self.frame_stack_size:
                frame_buffer.pop(0)

            while len(frame_buffer) < self.frame_stack_size:
                frame_buffer.append(current_frame.clone())

            stacked = torch.stack(frame_buffer, dim=0)
            stacked_state = stacked.view(-1, stacked.shape[-2], stacked.shape[-1])
            stacked_state = stacked_state.unsqueeze(0)

            with torch.no_grad():
                _ = self.model(stacked_state)

        stacked_activations = {}
        for layer_name, activation_list in self.layer_activations.items():
            stacked_activations[layer_name] = torch.stack(activation_list, dim=0)

        result = {
            "activations": stacked_activations,
            "behavioral": {
                "win": human_win,
                "score": human_score,
                "timestamps": state_timestamps,
                "num_states": len(states),
            },
            "metadata": {
                "play_id": str(play_id),
                "subj_id": int(subj_id),
                "run_id": int(run_id) if run_id is not None else None,
                "game_name": game_name,
                "level_id": int(level_id),
                "model_type": "GridDQN",
                "checkpoint_sha256": self.checkpoint_sha256,
                "source_revision": self.source_provenance["revision"],
                "source_dirty": self.source_provenance["dirty"],
                "source_distribution": self.source_provenance["distribution"],
                "source_manifest_sha256": self.source_provenance["manifest_sha256"],
                "shapes": {k: tuple(v.shape) for k, v in stacked_activations.items()},
            },
        }

        return result

    def cleanup(self):
        """Remove activation hooks."""
        for handle in self.hook_handles:
            handle.remove()


def download_checkpoint(
    game_name: str,
    level: int,
    cache_dir: Path,
    *,
    allow_download=False,
    checkpoint_map=None,
) -> Path:
    """
    Download checkpoint from wandb if not cached.

    Args:
        game_name: Game name (e.g., 'vgfmri4_bait')
        level: Level number (0-8)
        cache_dir: Directory to cache checkpoints

    Returns:
        Path to local checkpoint file
    """
    if checkpoint_map is not None:
        try:
            local = Path(checkpoint_map[game_name][str(level)]).expanduser().resolve()
        except KeyError as exc:
            raise ValueError(
                f"Checkpoint map has no {game_name}, level {level}"
            ) from exc
        if not local.is_file():
            raise FileNotFoundError(local)
        return local
    if game_name not in WANDB_CHECKPOINTS:
        raise ValueError(
            f"Game {game_name} not found in WANDB_CHECKPOINTS. Available: {list(WANDB_CHECKPOINTS.keys())}"
        )

    game_config = WANDB_CHECKPOINTS[game_name]
    run_id = game_config["run_id"]
    pattern = game_config["pattern"]
    checkpoint_filename = pattern.format(level=level)

    cache_path = cache_dir / f"{game_name}_level{level}.pt"

    if cache_path.exists():
        logging.info(f"  Using cached checkpoint: {cache_path.name}")
        return cache_path

    if not allow_download:
        raise FileNotFoundError(
            f"Missing checkpoint {cache_path}. Supply --checkpoint-map or explicitly "
            "enable the historical W&B mapping with --allow-checkpoint-download."
        )
    import wandb

    logging.info(f"  Downloading checkpoint from wandb: {checkpoint_filename}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")

    file = run.file(checkpoint_filename)

    temp_dir = cache_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    file.download(root=str(temp_dir), replace=True)

    downloaded_path = temp_dir / checkpoint_filename
    downloaded_path.rename(cache_path)

    model_weights_dir = temp_dir / "model_weights"
    if model_weights_dir.exists() and not any(model_weights_dir.iterdir()):
        model_weights_dir.rmdir()
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    logging.info(f"  Downloaded: {cache_path.name}")
    return cache_path


def save_model_features_for_level(
    plays_data: list,
    output_dir: str,
    model_id: str,
    subject: str,
    game_name: str,
    level: int,
):
    """
    Save model features for all rollouts of a level to a single file.

    Args:
        plays_data: List of dicts with activations, behavioral, metadata
        output_dir: Base output directory
        model_id: Model identifier (e.g., 'ddqn-trial1-sequential')
        subject: Subject ID (e.g., 'sub-01')
        game_name: Game name
        level: Level number
    """
    subj_str = subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"
    model_dir = Path(output_dir) / f"model-{model_id}"
    subj_dir = model_dir / subj_str
    from data.game_ids import canonical_game_id

    game_dir = subj_dir / canonical_game_id(game_name)
    game_dir.mkdir(parents=True, exist_ok=True)

    output_file = game_dir / f"level-{level:02d}.npz"

    save_dict = {}
    play_ids = []

    for idx, data in enumerate(plays_data):
        play_ids.append(data["metadata"]["play_id"])

        for layer_name, tensor in data["activations"].items():
            save_dict[f"play_{idx}_model_{layer_name}"] = tensor.cpu().numpy()

        save_dict[f"play_{idx}_behavioral_win"] = data["behavioral"]["win"]
        save_dict[f"play_{idx}_behavioral_score"] = data["behavioral"]["score"]
        save_dict[f"play_{idx}_behavioral_timestamps"] = np.array(
            data["behavioral"]["timestamps"]
        )
        save_dict[f"play_{idx}_behavioral_num_states"] = data["behavioral"][
            "num_states"
        ]

        save_dict[f"play_{idx}_metadata_play_id"] = data["metadata"]["play_id"]
        save_dict[f"play_{idx}_metadata_subj_id"] = data["metadata"]["subj_id"]
        save_dict[f"play_{idx}_metadata_run_id"] = (
            data["metadata"]["run_id"] if data["metadata"]["run_id"] is not None else -1
        )
        save_dict[f"play_{idx}_metadata_game_name"] = data["metadata"]["game_name"]
        save_dict[f"play_{idx}_metadata_level_id"] = data["metadata"]["level_id"]
        save_dict[f"play_{idx}_metadata_model_type"] = data["metadata"]["model_type"]
        for key in (
            "checkpoint_sha256",
            "source_revision",
            "source_dirty",
            "source_distribution",
            "source_manifest_sha256",
        ):
            if key in data["metadata"]:
                save_dict[f"play_{idx}_metadata_{key}"] = data["metadata"][key]

    save_dict["play_ids"] = np.array(play_ids, dtype="U24")
    save_dict["num_plays"] = len(plays_data)

    np.savez_compressed(output_file, **save_dict)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logging.info(f"Saved: {file_size_mb:.1f}MB -> {output_file.name}")

    return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Extract model features aligned with behavioral states"
    )
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier (e.g., ddqn-trial1-sequential)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to cache model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--behavior-dir",
        "--local-dir",
        dest="behavior_dir",
        required=True,
        help="Canonical human recordings root or downloaded dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default="./workdir/extract_model_features_to_npz",
        help="Output directory (default: ./workdir/extract_model_features_to_npz)",
    )

    parser.add_argument(
        "--checkpoint-map",
        type=Path,
        help="JSON: {game: {level: absolute checkpoint path}}; overrides historical W&B mapping",
    )
    parser.add_argument(
        "--allow-checkpoint-download",
        action="store_true",
        help="Download missing checkpoints from the historical trial1-sequential W&B runs",
    )
    parser.add_argument(
        "--skip-unsupported-games",
        action="store_true",
        help="Explicitly omit games absent from the checkpoint mapping",
    )
    args = parser.parse_args()
    checkpoint_map = (
        json.loads(args.checkpoint_map.read_text()) if args.checkpoint_map else None
    )

    input_dir = args.behavior_dir
    output_dir = args.output_dir
    checkpoint_dir = Path(args.checkpoint_dir)

    logging.info(f"Processing {args.subject} run-{args.run:02d}")

    plays, run_doc = load_behavioral_data(args.subject, args.run, input_dir)
    logging.info(f"  Found {len(plays)} plays")

    plays_by_level = defaultdict(list)
    extractors = {}

    for play in plays:
        game_name_full = play["game_name"]
        level = play["level_id"]

        assert game_name_full.startswith("vgfmri3_") or game_name_full.startswith(
            "vgfmri4_"
        ), f"Game name must start with vgfmri3_ or vgfmri4_, got: {game_name_full}"

        game_name = game_name_full.split("_", 1)[1]

        mapping = checkpoint_map if checkpoint_map is not None else WANDB_CHECKPOINTS
        if game_name not in mapping:
            if not args.skip_unsupported_games:
                raise ValueError(
                    f"No checkpoint mapping for {game_name_full}; provide --checkpoint-map"
                )
            logging.warning(
                f"  Explicitly skipping {game_name_full}: absent from checkpoint map"
            )
            continue

        key = (game_name_full, level)

        if key not in extractors:
            logging.info(f"  Loading model for {game_name} level {level}")
            checkpoint_path = download_checkpoint(
                game_name,
                level,
                checkpoint_dir,
                allow_download=args.allow_checkpoint_download,
                checkpoint_map=checkpoint_map,
            )

            extractor = ModelFeatureExtractor(
                checkpoint_path=checkpoint_path,
                game_name=game_name_full,
                level=level,
            )
            extractors[key] = extractor

        result = extractors[key].extract_features_from_play(play)
        plays_by_level[key].append(result)

    for extractor in extractors.values():
        extractor.cleanup()

    if not plays_by_level:
        raise ValueError("No plays were extracted; refusing to report success")
    saved_files = []
    for (game_name_full, level), level_plays in plays_by_level.items():
        logging.info(
            f"  Saving {game_name_full} level {level}: {len(level_plays)} plays"
        )
        output_file = save_model_features_for_level(
            level_plays, output_dir, args.model_id, args.subject, game_name_full, level
        )
        saved_files.append(output_file)

    logging.info(
        f"Completed {args.subject} run-{args.run:02d}: {len(saved_files)} files saved"
    )
    logging.info(f"All done! Data saved to {output_dir}")
