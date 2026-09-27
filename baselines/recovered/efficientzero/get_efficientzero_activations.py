from __future__ import annotations

import argparse
import ast
import inspect
import math
import os
import sys
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from omegaconf import OmegaConf

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None  # type: ignore

try:
    from bson import BSON, decode_file_iter
except ImportError:  # pragma: no cover - optional dependency
    BSON = None  # type: ignore
    decode_file_iter = None  # type: ignore

try:
    from vgdl import colors  # type: ignore
    from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    colors = None  # type: ignore
    createRLInputGameFromStrings = None  # type: ignore

# Ensure dependencies in the parent workspace remain importable after moving this script.
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent.parent
_WORKSPACE_ROOT = _REPO_ROOT.parent

_DEFAULT_EZ_ROOT = Path(
    os.environ.get("EZ_ROOT", str(_WORKSPACE_ROOT / "EfficientZeroV2"))
).expanduser()
if _DEFAULT_EZ_ROOT.exists() and str(_DEFAULT_EZ_ROOT) not in sys.path:
    sys.path.append(str(_DEFAULT_EZ_ROOT))

_DEFAULT_RC_RL_ROOT = Path(
    os.environ.get("RC_RL_ROOT", str(_WORKSPACE_ROOT / "RC_RL"))
).expanduser()
if _DEFAULT_RC_RL_ROOT.exists() and str(_DEFAULT_RC_RL_ROOT) not in sys.path:
    sys.path.append(str(_DEFAULT_RC_RL_ROOT))

from ez.agents.models import EfficientZero  # noqa: E402  (depends on sys.path edit above)
from ez.agents.models.base_model import (  # noqa: E402
    DynamicsNetwork,
    ProjectionHeadNetwork,
    ProjectionNetwork,
    RepresentationNetwork,
    SupportLSTMNetwork,
    SupportNetwork,
    ValuePolicyNetwork,
)
from ez.utils.format import DiscreteSupport, symexp


class HiddenStateExtractor:
    """Callable wrapper that exposes hidden-state activations from EfficientZero."""

    def __init__(self, model: EfficientZero, config: OmegaConf, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()

    def __call__(
        self,
        observations: Union[Tensor, Sequence],
        *,
        as_numpy: bool = True,
    ) -> Union[Tensor, Any]:
        prepared = _prepare_observations(observations, self.config, self.device)
        with torch.no_grad():
            hidden = self.model.do_representation(prepared)
        hidden = hidden.detach().cpu()
        return hidden.numpy() if as_numpy else hidden


def load_hidden_state_extractor(
    model_path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    config_override: Optional[Union[Mapping[str, Any], OmegaConf]] = None,
) -> HiddenStateExtractor:
    """Load an EfficientZero checkpoint and return a hidden-state extractor."""
    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {resolved_model_path}")

    run_dir = _locate_run_directory(resolved_model_path)
    config = _load_config(run_dir, override=config_override)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = _build_model_from_config(config, target_device)
    load_kwargs = {"map_location": target_device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    state_dict = torch.load(str(resolved_model_path), **load_kwargs)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if isinstance(state_dict, Mapping):
        state_dict = _normalize_state_dict_keys(state_dict)

    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    return HiddenStateExtractor(model, config, target_device)


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strip known training-time prefixes (DDP/torch.compile) from checkpoint keys."""
    prefixes = ("_orig_mod.", "module.")
    needs_normalisation = any(
        isinstance(key, str) and any(key.startswith(prefix) for prefix in prefixes)
        for key in state_dict.keys()
    )
    if not needs_normalisation:
        return state_dict

    sanitized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if isinstance(new_key, str):
            for prefix in prefixes:
                while new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
        sanitized[new_key] = value
    return sanitized


def _locate_run_directory(checkpoint_path: Path) -> Path:
    """Infer the EfficientZero run directory from a checkpoint path."""
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "models" else checkpoint_path.parent
    if not run_dir.exists():
        raise FileNotFoundError(f"Unable to infer run directory from {checkpoint_path}")
    return run_dir


def _load_config(run_dir: Path, override: Optional[Union[Mapping[str, Any], OmegaConf]]) -> OmegaConf:
    """Recover the Hydra config stored in logs/Train.log, or apply a manual override."""
    if override is not None:
        return OmegaConf.create(override)

    log_path = run_dir / "logs" / "Train.log"
    if not log_path.exists():
        raise FileNotFoundError(
            f"Cannot recover the training config. Expected a log file at {log_path}."
        )

    config_line = None
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            if "config:" in line:
                config_line = line.partition("config:")[2].strip()
                break

    if not config_line:
        raise RuntimeError(f"No serialized config found in {log_path}")

    try:
        config_dict = ast.literal_eval(config_line)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"Failed to parse config from {log_path}: {exc}") from exc

    return OmegaConf.create(config_dict)


def _build_model_from_config(config: OmegaConf, device: torch.device) -> EfficientZero:
    """Reproduce the EfficientZero architecture corresponding to the stored config."""
    obs_shape = list(config.env.obs_shape)
    n_stack = int(config.env.n_stack)
    input_shape = obs_shape.copy()
    input_shape[0] *= n_stack

    num_blocks = int(config.model.num_blocks)
    num_channels = int(config.model.num_channels)
    reduced_channels = int(config.model.reduced_channels)
    fc_layers = list(config.model.fc_layers)
    down_sample = bool(config.model.down_sample)
    state_norm = bool(config.model.state_norm)
    value_prefix = bool(config.model.value_prefix)
    init_zero = bool(config.model.init_zero)
    v_num = int(config.train.v_num)
    action_embedding = bool(config.model.get("action_embedding", False))
    action_embedding_dim = int(config.model.get("action_embedding_dim", 32))
    value_policy_detach = bool(config.train.get("value_policy_detach", False))
    action_space_size = int(config.env.action_space_size)

    if down_sample:
        height = math.ceil(obs_shape[1] / 16)
        width = math.ceil(obs_shape[2] / 16)
    else:
        height, width = obs_shape[1], obs_shape[2]

    state_shape = (num_channels, height, width)
    state_dim = state_shape[0] * state_shape[1] * state_shape[2]
    flatten_size = reduced_channels * height * width

    representation_model = RepresentationNetwork(tuple(input_shape), num_blocks, num_channels, down_sample)
    dynamics_model = DynamicsNetwork(
        num_blocks,
        num_channels,
        action_space_size,
        action_embedding=action_embedding,
        action_embedding_dim=action_embedding_dim,
    )

    value_policy_model = ValuePolicyNetwork(
        num_blocks,
        num_channels,
        reduced_channels,
        flatten_size,
        fc_layers,
        int(config.model.value_support.size),
        action_space_size,
        init_zero,
        value_policy_detach=value_policy_detach,
        v_num=v_num,
    )

    reward_support_size = int(config.model.reward_support.size)
    if value_prefix:
        reward_prediction_model = SupportLSTMNetwork(
            0,
            num_channels,
            reduced_channels,
            flatten_size,
            fc_layers,
            reward_support_size,
            int(config.model.lstm_hidden_size),
            init_zero,
        )
    else:
        reward_prediction_model = SupportNetwork(
            num_blocks,
            num_channels,
            reduced_channels,
            flatten_size,
            fc_layers,
            reward_support_size,
            init_zero,
        )

    projection_layers = list(config.model.projection_layers)
    projection_head_layers = list(config.model.prjection_head_layers)
    projection_model = ProjectionNetwork(state_dim, projection_layers[0], projection_layers[1])
    projection_head_model = ProjectionHeadNetwork(
        projection_layers[1],
        projection_head_layers[0],
        projection_head_layers[1],
    )

    model = EfficientZero(
        representation_model,
        dynamics_model,
        reward_prediction_model,
        value_policy_model,
        projection_model,
        projection_head_model,
        config,
        state_norm=state_norm,
        value_prefix=value_prefix,
    )

    model.to(device)
    return model


def _prepare_observations(
    observations: Union[Tensor, Sequence],
    config: OmegaConf,
    device: torch.device,
) -> Tensor:
    """Convert user-provided observations into a [B, C, H, W] tensor."""
    if isinstance(observations, Tensor):
        obs_tensor = observations.detach().clone()
    else:
        obs_tensor = torch.as_tensor(observations)

    if obs_tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        obs_tensor = obs_tensor.float()
        if obs_tensor.max().item() > 1.0:
            obs_tensor = obs_tensor / 255.0
    else:
        obs_tensor = obs_tensor.float()

    expected_channels = int(config.env.obs_shape[0]) * int(config.env.n_stack)
    obs_tensor = _ensure_channels_first(obs_tensor, expected_channels)

    if obs_tensor.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} stacked channels but received {obs_tensor.shape[1]}. "
            "Ensure you provide all frames used during training."
        )

    return obs_tensor.to(device, non_blocking=True)


def _infer_frame_channels(config: OmegaConf) -> int:
    """Heuristically recover the per-frame channel count from the training config."""
    obs_channels = int(config.env.obs_shape[0])
    n_stack = max(1, int(config.env.n_stack))
    if obs_channels >= n_stack and obs_channels % n_stack == 0:
        per_frame = obs_channels // n_stack
        if per_frame > 0:
            return per_frame
    if bool(config.env.get("gray_scale", False)):
        return 1
    if obs_channels in (1, 3):
        return obs_channels
    return 3 if obs_channels > 1 else 1


def _ensure_channels_first(obs: Tensor, expected_channels: int) -> Tensor:
    """Normalize various observation layouts to [B, C, H, W] tensors."""
    if obs.ndim == 2:
        obs = obs.unsqueeze(0).unsqueeze(0)
    elif obs.ndim == 3:
        if obs.shape[0] == expected_channels and obs.shape[1] != expected_channels:
            obs = obs.unsqueeze(0)
        elif obs.shape[-1] == expected_channels:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        elif obs.shape[0] == expected_channels:
            obs = obs.unsqueeze(0)
        else:
            raise ValueError(f"Cannot infer channel dimension from observation with shape {tuple(obs.shape)}.")
    elif obs.ndim == 4:
        if obs.shape[1] == expected_channels:
            pass
        elif obs.shape[-1] == expected_channels:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.shape[0] == expected_channels and obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Cannot infer channels for tensor with shape {tuple(obs.shape)}.")
    elif obs.ndim == 5:
        obs = obs.permute(0, 1, 4, 2, 3)
        batch, stack, channel, height, width = obs.shape
        obs = obs.reshape(batch, stack * channel, height, width)
    else:
        raise ValueError(f"Unsupported observation rank: {obs.ndim}")

    if obs.ndim != 4:
        raise RuntimeError(f"Observation preprocessing failed; received tensor with shape {tuple(obs.shape)}")

    return obs


_PYGAME_INITIALIZED = False


def _require_numpy() -> None:
    if np is None:  # type: ignore[truthy-bool]
        raise ImportError(
            "numpy is required to decode VGDL z-states. "
            "Install it with `pip install numpy`."
        )


def _require_cv2() -> None:
    if cv2 is None:  # type: ignore[truthy-bool]
        raise ImportError(
            "opencv-python is required to render VGDL frames. "
            "Install it with `pip install opencv-python`."
        )


def _require_pygame() -> None:
    if pygame is None:  # type: ignore[truthy-bool]
        raise ImportError(
            "pygame is required to initialize VGDL games. "
            "Install it with `pip install pygame`."
        )


def _require_bson() -> None:
    if BSON is None or decode_file_iter is None:  # type: ignore[truthy-bool]
        raise ImportError(
            "pymongo is required to read the Mongo dump. "
            "Install it with `pip install pymongo`."
        )


def _require_vgdl() -> None:
    global createRLInputGameFromStrings, colors
    if createRLInputGameFromStrings is not None:  # type: ignore[truthy-bool]
        return
    try:  # pragma: no cover - optional dependency
        from vgdl import colors as vgdl_colors  # type: ignore
        from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings as create_fn  # type: ignore
    except ImportError as exc:  # pragma: no cover - developer feedback
        raise ImportError(
            "RC_RL/vgdl is required to recreate VGDL states. "
            "Set RC_RL_ROOT or place the RC_RL repo next to this project."
        ) from exc
    colors = vgdl_colors  # type: ignore[assignment]
    createRLInputGameFromStrings = create_fn  # type: ignore[assignment]


def _ensure_pygame_initialized() -> None:
    global _PYGAME_INITIALIZED
    if not _PYGAME_INITIALIZED:
        _require_pygame()
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()  # type: ignore[call-arg]
        _PYGAME_INITIALIZED = True


def _binary_to_bytes(payload: Any) -> bytes:
    if payload is None:
        raise ValueError("Expected binary payload but received None")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    try:
        return bytes(payload)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Unable to coerce payload of type {type(payload).__name__!r} to bytes") from exc


def _rect_to_bounds(rect: Any) -> Tuple[int, int, int, int]:
    """Return (top, left, height, width) for pygame.Rect instances or dict-like snapshots."""
    if hasattr(rect, "top"):
        return int(rect.top), int(rect.left), int(rect.height), int(rect.width)

    if isinstance(rect, Mapping):
        keys = set(rect.keys())

        def _as_pair(value: Any, label: str) -> Tuple[int, int]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
                return int(value[0]), int(value[1])
            raise TypeError(f"Rect field '{label}' expected a length-2 sequence; received {value!r}")

        if {"top", "left", "height", "width"}.issubset(keys):
            return int(rect["top"]), int(rect["left"]), int(rect["height"]), int(rect["width"])

        if {"y", "x", "height", "width"}.issubset(keys):
            return int(rect["y"]), int(rect["x"]), int(rect["height"]), int(rect["width"])

        if {"y", "x", "h", "w"}.issubset(keys):
            return int(rect["y"]), int(rect["x"]), int(rect["h"]), int(rect["w"])

        if "pos" in keys and "size" in keys:
            left, top = _as_pair(rect["pos"], "pos")
            width, height = _as_pair(rect["size"], "size")
            return top, left, height, width

        if "topleft" in keys and "size" in keys:
            left, top = _as_pair(rect["topleft"], "topleft")
            width, height = _as_pair(rect["size"], "size")
            return top, left, height, width

        if "center" in keys and "size" in keys:
            cx, cy = _as_pair(rect["center"], "center")
            width, height = _as_pair(rect["size"], "size")
            left = cx - width // 2
            top = cy - height // 2
            return top, left, height, width

        if {"left", "top", "right", "bottom"}.issubset(keys):
            left = int(rect["left"])
            top = int(rect["top"])
            width = int(rect["right"]) - left
            height = int(rect["bottom"]) - top
            return top, left, height, width

        raise KeyError(f"Unrecognized rect mapping keys: {sorted(keys)}")

    raise TypeError(f"Unsupported rect type: {type(rect).__name__}")


def _render_state_frame(
    game,
    state: Mapping[str, Any],
    *,
    resize: Optional[int],
    num_channels: int = 1,
) -> "np.ndarray":
    _require_numpy()
    _require_cv2()

    game.setFullState(state)  # type: ignore[attr-defined]
    background = np.array(colors.LIGHTGRAY if colors is not None else (211, 211, 211), dtype=np.uint8)  # type: ignore[arg-type]
    width, height = game.screensize  # (w, h)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = background

    active_kill_list = set(getattr(game, "kill_list", []))
    for sprite_class in getattr(game, "sprite_order", []):
        group = getattr(game, "sprite_groups", {}).get(sprite_class)
        if not group:
            continue
        for sprite in group:
            if sprite in active_kill_list:
                continue
            rect = sprite.rect
            top, left, height_px, width_px = _rect_to_bounds(rect)
            canvas[top : top + height_px, left : left + width_px, :] = np.array(sprite.color, dtype=np.uint8)

    if resize and (canvas.shape[0] != resize or canvas.shape[1] != resize):
        canvas = cv2.resize(canvas, (resize, resize), interpolation=cv2.INTER_AREA)

    if num_channels == 1:
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.uint8)
    if num_channels == 3:
        return canvas.astype(np.uint8)
    raise ValueError(f"Unsupported channel count requested: {num_channels}")


def _stack_frame_sequence(frames: "np.ndarray", n_stack: int) -> "np.ndarray":
    if n_stack <= 1:
        return frames

    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, C, H, W); received {frames.shape}")

    pad_count = n_stack - 1
    padding = np.repeat(frames[:1], pad_count, axis=0)
    padded = np.concatenate([padding, frames], axis=0)
    stacked = []
    for idx in range(pad_count, padded.shape[0]):
        window = padded[idx - pad_count : idx + 1]
        stacked.append(window.reshape(-1, frames.shape[2], frames.shape[3]))
    return np.stack(stacked, axis=0)


class VGDLZStateLoader:
    """Utility for reading VGDL z-states from the ds004323 Mongo dump."""

    def __init__(self, dataset_root: Union[str, Path], *, rc_rl_root: Optional[Union[str, Path]] = None) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.dump_root = self._resolve_dump_root()
        self.rc_rl_root = Path(rc_rl_root).expanduser() if rc_rl_root else _DEFAULT_RC_RL_ROOT
        if self.rc_rl_root and self.rc_rl_root.exists() and str(self.rc_rl_root) not in sys.path:
            sys.path.append(str(self.rc_rl_root))

    def _resolve_dump_root(self) -> Path:
        direct_candidate = self.dataset_root
        if direct_candidate.is_dir() and (direct_candidate / "plays.bson").exists():
            return direct_candidate
        nested_candidate = self.dataset_root / "behavior" / "dump" / "vgfmri"
        if nested_candidate.is_dir() and (nested_candidate / "plays.bson").exists():
            return nested_candidate
        raise FileNotFoundError(
            "Could not locate a Mongo dump beneath the provided dataset root. "
            "Ensure you have extracted behavior/dump.tar.gz into behavior/dump/."
        )

    def _iter_collection(self, collection: str) -> Iterable[Mapping[str, Any]]:
        _require_bson()
        collection_path = self.dump_root / f"{collection}.bson"
        if not collection_path.exists():
            raise FileNotFoundError(f"Expected {collection_path} to exist. Did you extract the Mongo dump?")
        with collection_path.open("rb") as handle:
            yield from decode_file_iter(handle)  # type: ignore[arg-type]

    def iter_plays(
        self,
        *,
        subj_id: Optional[Union[str, int]] = None,
        run_id: Optional[int] = None,
        play_id: Optional[int] = None,
        game_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Mapping[str, Any]]:
        count = 0
        for doc in self._iter_collection("plays"):
            if subj_id is not None and str(doc.get("subj_id")) != str(subj_id):
                continue
            if run_id is not None and int(doc.get("run_id", -1)) != int(run_id):
                continue
            if play_id is not None and int(doc.get("play_id", -1)) != int(play_id):
                continue
            if game_name is not None and doc.get("game_name") != game_name:
                continue
            yield doc
            count += 1
            if limit is not None and count >= limit:
                return

    def get_play(
        self,
        *,
        subj_id: Optional[Union[str, int]] = None,
        run_id: Optional[int] = None,
        play_id: Optional[int] = None,
        game_name: Optional[str] = None,
    ) -> Mapping[str, Any]:
        try:
            return next(self.iter_plays(subj_id=subj_id, run_id=run_id, play_id=play_id, game_name=game_name))
        except StopIteration as exc:
            filters = {k: v for k, v in {
                "subj_id": subj_id,
                "run_id": run_id,
                "play_id": play_id,
                "game_name": game_name,
            }.items() if v is not None}
            raise LookupError(f"No play matched filters {filters}") from exc

    def decode_zstates(self, play_doc: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        raw = _binary_to_bytes(play_doc.get("zstates"))
        decompressed = zlib.decompress(raw)
        payload = BSON(decompressed).decode()  # type: ignore[call-arg]
        states = payload.get("states", [])
        if not isinstance(states, Sequence) or isinstance(states, (bytes, str)):
            raise TypeError("Decoded zstates payload does not contain a sequence of states")
        return states

    def render_frames(
        self,
        play_doc: Mapping[str, Any],
        *,
        resize: Optional[int] = 84,
        num_channels: int = 1,
    ) -> "np.ndarray":
        _require_numpy()
        _require_cv2()
        _require_vgdl()
        _ensure_pygame_initialized()

        game_str = play_doc.get("game_str")
        level_str = play_doc.get("level_str")
        if not game_str or not level_str:
            raise ValueError("play document is missing 'game_str' or 'level_str'")

        env = createRLInputGameFromStrings(game_str, level_str)  # type: ignore[call-arg]
        env.visualize = False  # type: ignore[attr-defined]
        frames = []
        for state in self.decode_zstates(play_doc):
            frame = _render_state_frame(
                env._game,
                state,
                resize=resize,
                num_channels=num_channels,
            )  # type: ignore[attr-defined]
            frames.append(frame)
        if not frames:
            raise ValueError("No decoded frames were produced from zstates")
        stacked = np.stack(frames, axis=0)
        if num_channels == 1:
            return stacked[:, np.newaxis, :, :]
        if num_channels == 3:
            return stacked.transpose(0, 3, 1, 2)
        raise ValueError(f"Unsupported channel count requested: {num_channels}")

    def load_play_observations(
        self,
        *,
        subj_id: Optional[Union[str, int]] = None,
        run_id: Optional[int] = None,
        play_id: Optional[int] = None,
        game_name: Optional[str] = None,
        resize: Optional[int] = 84,
        n_stack: int = 1,
        num_channels: int = 1,
        as_tensor: bool = True,
    ) -> Tuple[Any, Mapping[str, Any]]:
        play_doc = self.get_play(subj_id=subj_id, run_id=run_id, play_id=play_id, game_name=game_name)
        frames = self.render_frames(play_doc, resize=resize, num_channels=num_channels)
        stacked = _stack_frame_sequence(frames, n_stack)
        if as_tensor:
            return torch.from_numpy(stacked), play_doc
        return stacked, play_doc


def _load_config_override(path_or_literal: Optional[str]) -> Optional[Union[Mapping[str, Any], OmegaConf]]:
    if not path_or_literal:
        return None
    candidate = Path(path_or_literal)
    if candidate.exists():
        return OmegaConf.load(str(candidate))
    try:
        parsed = ast.literal_eval(path_or_literal)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"Could not interpret config override '{path_or_literal}'. "
            "Provide a path to YAML/JSON or a Python literal."
        )
    return OmegaConf.create(parsed)


class EfficientZeroActivationEvaluator:
    """Evaluate EfficientZero plays and capture per-subnetwork activations."""

    ACTION_NAME_TO_INDEX = {
        "noop": 0,
        "left": 1,
        "right": 2,
        "up": 3,
        "down": 4,
        "spacebar": 5,
    }

    def __init__(
        self,
        model_path: Path,
        dataset_root: Path,
        *,
        device: Optional[str] = None,
        config_override: Optional[OmegaConf] = None,
    ) -> None:
        extractor = load_hidden_state_extractor(
            model_path,
            device=device,
            config_override=config_override,
        )
        self.model = extractor.model
        self.config = extractor.config
        self.device = extractor.device
        self.loader = VGDLZStateLoader(dataset_root)
        self.frame_channels = _infer_frame_channels(self.config)
        self.obs_resize = int(self.config.env.obs_shape[1]) if len(self.config.env.obs_shape) > 1 else None
        self.n_stack = int(self.config.env.n_stack)
        self.action_space_size = int(self.config.env.action_space_size)
        self.model.eval()

    def iter_matching_plays(
        self,
        *,
        subj_id: Optional[str],
        run_id: Optional[int],
        play_id: Optional[int],
        game_name: Optional[str],
        limit: Optional[int],
    ) -> Iterable[Dict[str, Any]]:
        yield from self.loader.iter_plays(
            subj_id=subj_id,
            run_id=run_id,
            play_id=play_id,
            game_name=game_name,
            limit=limit,
        )

    def evaluate_play(self, play_doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        observations = self._observations_from_play(play_doc)
        prepared = _prepare_observations(observations, self.config, self.device)
        num_states = prepared.shape[0]

        reward_hidden = self._initial_reward_hidden()
        initial_states: List[torch.Tensor] = []
        initial_values: List[torch.Tensor] = []
        initial_policies: List[torch.Tensor] = []
        projections: List[torch.Tensor] = []
        projection_heads: List[torch.Tensor] = []
        policy_entropies: List[float] = []
        value_scalars: List[float] = []
        predicted_actions: List[int] = []

        recurrent_states: List[torch.Tensor] = []
        recurrent_values: List[torch.Tensor] = []
        recurrent_policies: List[torch.Tensor] = []
        reward_predictions: List[torch.Tensor] = []

        human_actions = self._human_action_indices(play_doc)
        human_win = bool(play_doc.get("win"))
        human_score = play_doc.get("score")

        with torch.no_grad():
            for idx in range(num_states):
                obs = prepared[idx : idx + 1]
                state, values, policy = self.model.initial_inference(obs, training=True)
                initial_states.append(state.detach().cpu())
                initial_values.append(values.detach().cpu())
                initial_policies.append(policy.detach().cpu())

                proj = self.model.projection_model(state)
                projections.append(proj.detach().cpu())
                projection_heads.append(self.model.projection_head_model(proj).detach().cpu())

                policy_probs = torch.softmax(policy, dim=-1)
                policy_entropy = -(policy_probs * policy_probs.clamp_min(1e-12).log()).sum(dim=-1)
                policy_entropies.append(float(policy_entropy.squeeze(0).cpu().item()))
                predicted_actions.append(int(policy_probs.argmax(dim=-1).item()))

                scalar_value = self._decode_values(values)
                value_scalars.append(float(scalar_value.item()))

                if idx < len(human_actions):
                    action_value = torch.tensor([[human_actions[idx]]], device=self.device, dtype=torch.float32)
                    next_state, reward_pred, next_values, next_policy, reward_hidden = self.model.recurrent_inference(
                        state,
                        action_value,
                        reward_hidden,
                        training=True,
                    )
                    recurrent_states.append(next_state.detach().cpu())
                    reward_predictions.append(reward_pred.detach().cpu())
                    recurrent_values.append(next_values.detach().cpu())
                    recurrent_policies.append(next_policy.detach().cpu())

        activations = self._build_activation_payload(
            initial_states,
            initial_values,
            initial_policies,
            projections,
            projection_heads,
            recurrent_states,
            recurrent_values,
            recurrent_policies,
            reward_predictions,
        )

        agreement_rate = self._compute_agreement(predicted_actions, human_actions)
        result = {
            "play_id": str(play_doc.get("_id", play_doc.get("play_id"))),
            "subj_id": play_doc.get("subj_id"),
            "run_id": play_doc.get("run_id"),
            "play_index": play_doc.get("play_id"),
            "game_name": play_doc.get("game_name"),
            "num_states": num_states,
            "num_human_actions": len(human_actions),
            "agreement_rate": agreement_rate,
            "mean_value": float(np.mean(value_scalars)) if value_scalars else None,
            "std_value": float(np.std(value_scalars)) if value_scalars else None,
            "mean_policy_entropy": float(np.mean(policy_entropies)) if policy_entropies else None,
            "human_win": human_win,
            "human_score": human_score,
        }
        return result, activations

    def _observations_from_play(self, play_doc: Dict[str, Any]) -> torch.Tensor:
        frames = self.loader.render_frames(
            play_doc,
            resize=self.obs_resize,
            num_channels=self.frame_channels,
        )
        stacked = _stack_frame_sequence(frames, self.n_stack)
        return torch.from_numpy(stacked)

    def _initial_reward_hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not bool(self.config.model.value_prefix):
            return None
        hidden_size = int(self.config.model.lstm_hidden_size)
        h = torch.zeros(1, 1, hidden_size, device=self.device)
        c = torch.zeros(1, 1, hidden_size, device=self.device)
        return h, c

    def _human_action_indices(self, play_doc: Dict[str, Any]) -> List[int]:
        raw_actions = play_doc.get("actions") or []
        converted = []
        for entry in raw_actions:
            if isinstance(entry, (list, tuple)) and entry:
                name = entry[0]
            else:
                name = entry
            converted.append(self.ACTION_NAME_TO_INDEX.get(str(name).lower(), 0))
        return converted

    def _decode_values(self, values: torch.Tensor) -> torch.Tensor:
        if str(self.config.model.value_support.type) == "symlog":
            decoded = symexp(values)
        else:
            decoded = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support)
        return decoded.mean(dim=0).squeeze()

    def _build_activation_payload(
        self,
        initial_states: List[torch.Tensor],
        initial_values: List[torch.Tensor],
        initial_policies: List[torch.Tensor],
        projections: List[torch.Tensor],
        projection_heads: List[torch.Tensor],
        recurrent_states: List[torch.Tensor],
        recurrent_values: List[torch.Tensor],
        recurrent_policies: List[torch.Tensor],
        reward_predictions: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        activations: Dict[str, torch.Tensor] = {}
        if initial_states:
            activations["representation_states"] = torch.cat(initial_states, dim=0)
        if initial_values:
            stacked_values = torch.cat(initial_values, dim=1)
            activations["initial_value_logits"] = stacked_values.permute(1, 0, 2)
        if initial_policies:
            activations["initial_policy_logits"] = torch.cat(initial_policies, dim=0)
        if projections:
            activations["projection_outputs"] = torch.cat(projections, dim=0)
        if projection_heads:
            activations["projection_head_outputs"] = torch.cat(projection_heads, dim=0)
        if recurrent_states:
            activations["dynamics_states"] = torch.cat(recurrent_states, dim=0)
        if recurrent_values:
            stacked = torch.cat(recurrent_values, dim=1)
            activations["recurrent_value_logits"] = stacked.permute(1, 0, 2)
        if recurrent_policies:
            activations["recurrent_policy_logits"] = torch.cat(recurrent_policies, dim=0)
        if reward_predictions:
            activations["reward_predictions"] = torch.cat(reward_predictions, dim=0)
        return activations

    @staticmethod
    def _compute_agreement(model_actions: List[int], human_actions: List[int]) -> Optional[float]:
        overlap = min(len(model_actions), len(human_actions))
        if overlap == 0:
            return None
        matches = sum(1 for idx in range(overlap) if model_actions[idx] == human_actions[idx])
        return matches / overlap


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientZero on VGDL playthroughs and collect subnetwork activations.",
    )
    parser.add_argument("--model", required=True, help="Path to the EfficientZero checkpoint (.pt).")
    parser.add_argument("--dataset-root", required=True, help="Local directory containing the Mongo dump.")
    parser.add_argument("--game-name", help="Filter plays by VGDL game name.")
    parser.add_argument("--subj-id", help="Optional subject identifier.")
    parser.add_argument("--run-id", type=int, help="Optional run index.")
    parser.add_argument("--play-id", type=int, help="Specific play identifier.")
    parser.add_argument("--limit", type=int, help="Maximum number of plays to process.")
    parser.add_argument("--device", help="Torch device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--config-override",
        help="Optional path to a config override (YAML/JSON) or literal dict expression.",
    )
    parser.add_argument("--save-activations", action="store_true", help="Persist captured activations to disk.")
    parser.add_argument("--activations-dir", default="efficientzero_activations", help="Directory for saved activations.")
    parser.add_argument("--output-csv", help="Optional CSV filepath for aggregated metrics.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    config_override = _load_config_override(args.config_override)
    evaluator = EfficientZeroActivationEvaluator(
        Path(args.model),
        Path(args.dataset_root),
        device=args.device,
        config_override=config_override,
    )

    if args.save_activations:
        Path(args.activations_dir).mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    plays = evaluator.iter_matching_plays(
        subj_id=args.subj_id,
        run_id=args.run_id,
        play_id=args.play_id,
        game_name=args.game_name,
        limit=args.limit,
    )

    for idx, play_doc in enumerate(plays, start=1):
        meta = (
            f"subj={play_doc.get('subj_id')}, run={play_doc.get('run_id')}, "
            f"play={play_doc.get('play_id')}, game={play_doc.get('game_name')}"
        )
        print(f"\nProcessing play {idx}: {meta}")
        result, activations = evaluator.evaluate_play(play_doc)
        results.append(result)
        agreement = result["agreement_rate"]
        agreement_str = f"{agreement:.2%}" if agreement is not None else "N/A"
        if result["mean_value"] is not None:
            print(
                f"  States: {result['num_states']}, Human win: {result['human_win']}, "
                f"Agreement: {agreement_str}, Mean value: {result['mean_value']:.3f}"
            )
        else:
            print(
                f"  States: {result['num_states']}, Human win: {result['human_win']}, "
                f"Agreement: {agreement_str}"
            )

        if args.save_activations and activations:
            play_id = result["play_id"]
            activation_path = Path(args.activations_dir) / f"{play_id}_activations.pt"
            payload = {
                "activations": activations,
                "metadata": {
                    "play_id": play_id,
                    "subj_id": result["subj_id"],
                    "run_id": result["run_id"],
                    "play_index": result["play_index"],
                    "game_name": result["game_name"],
                    "shapes": {k: tuple(v.shape) for k, v in activations.items()},
                },
            }
            torch.save(payload, activation_path)
            print(f"  Saved activations to {activation_path}")

    if not results:
        print("No plays matched the requested filters.")
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total plays processed: {len(results)}")

    agreement_values = [r["agreement_rate"] for r in results if r["agreement_rate"] is not None]
    if agreement_values:
        print(f"Average action agreement: {np.mean(agreement_values):.2%}")
    mean_values = [r["mean_value"] for r in results if r["mean_value"] is not None]
    if mean_values:
        print(f"Average predicted value: {np.mean(mean_values):.3f}")
    human_wins = sum(1 for r in results if r["human_win"])
    print(f"Human wins: {human_wins}/{len(results)} ({100 * human_wins / len(results):.1f}%)")

    if args.output_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
