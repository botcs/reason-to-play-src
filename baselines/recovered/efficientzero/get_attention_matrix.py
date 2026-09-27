from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import sys
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None  # type: ignore

try:
    from bson import BSON, decode_file_iter, ObjectId
except ImportError:  # pragma: no cover - optional dependency
    BSON = None  # type: ignore
    decode_file_iter = None  # type: ignore
    ObjectId = None  # type: ignore

try:
    from omegaconf import OmegaConf
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "omegaconf is required to load EfficientZero configs. "
        "Install it with `pip install omegaconf` inside your environment."
    ) from exc


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

try:  # pragma: no cover - optional dependency
    from vgdl import colors  # type: ignore
    from vgdl.rlenvironmentnonstatic import createRLInputGameFromStrings  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    colors = None  # type: ignore
    createRLInputGameFromStrings = None  # type: ignore

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


ACTION_NAME_TO_INDEX = {
    "noop": 0,
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "spacebar": 5,
}

HEAD_OUTPUT_REGISTRY = OrderedDict(
    [
        ("representation", "representation_states"),
        ("projection", "projection_outputs"),
        ("projection_head", "projection_head_outputs"),
        ("value_initial", "initial_value_logits"),
        ("policy_initial", "initial_policy_logits"),
        ("dynamics", "dynamics_states"),
        ("value_recurrent", "recurrent_value_logits"),
        ("policy_recurrent", "recurrent_policy_logits"),
    ("reward", "reward_predictions"),
]
)

class ActivationTraceRecorder:
    """Hook-based recorder that captures per-layer activations across timesteps."""

    def __init__(
        self,
        *,
        flatten: bool = True,
        allowed_layers: Optional[Sequence[str]] = None,
    ) -> None:
        self.flatten = flatten
        self.allowed_layers = set(allowed_layers) if allowed_layers else None
        self._requested_layers = list(allowed_layers) if allowed_layers else None
        self._available_layers: set[str] = set()
        self.traces: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
        self._hooks: list[Any] = []
        self._step: Optional[int] = None
        self._phase: Optional[str] = None

    def attach(self, model: EfficientZero) -> None:
        """Register hooks on all parameterized leaf modules in each EfficientZero subnetwork."""
        self._register_prefixed(model.representation_model, "representation")
        self._register_prefixed(model.dynamics_model, "dynamics")
        self._register_prefixed(model.reward_prediction_model, "reward")
        self._register_prefixed(model.value_policy_model, "value_policy")
        self._register_prefixed(model.projection_model, "projection")
        self._register_prefixed(model.projection_head_model, "projection_head")
        if self._requested_layers:
            missing = [name for name in self._requested_layers if name not in self._available_layers]
            if missing:
                print(
                    f"[WARN] Requested trace layers not found in model: {missing}",
                    file=sys.stderr,
                )

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def set_context(self, step: int, phase: str) -> None:
        self._step = int(step)
        self._phase = str(phase)

    def clear_context(self) -> None:
        self._step = None
        self._phase = None

    def _register_prefixed(self, module: nn.Module, prefix: str) -> None:
        for name, submodule in module.named_modules():
            if name == "":
                continue
            if not self._should_trace(submodule):
                continue
            hook_name = f"{prefix}.{name}"
            self._available_layers.add(hook_name)
            if self.allowed_layers is not None and hook_name not in self.allowed_layers:
                continue
            hook = submodule.register_forward_hook(self._hook_factory(hook_name))
            self._hooks.append(hook)

    def _should_trace(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return False
        return any(p.requires_grad for p in module.parameters())

    def _hook_factory(self, name: str):
        def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
            if self._step is None or self._phase is None:
                return
            tensors = self._gather_tensors(output)
            if not tensors:
                return
            flattened = [self._prepare_tensor(t) for t in tensors]
            shapes = [tuple(t.shape) for t in tensors]
            payload = flattened[0] if len(flattened) == 1 else tuple(flattened)
            record = {
                "timestep": self._step,
                "phase": self._phase,
                "shapes": shapes,
                "activation": payload,
            }
            self.traces.setdefault(name, []).append(record)

        return hook

    def _gather_tensors(self, value: Any) -> list[Tensor]:
        tensors: list[Tensor] = []
        if isinstance(value, Tensor):
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensors.extend(self._gather_tensors(item))
        elif isinstance(value, Mapping):
            for item in value.values():
                tensors.extend(self._gather_tensors(item))
        return tensors

    def _prepare_tensor(self, tensor: Tensor) -> Tensor:
        prepared = tensor.detach().cpu()
        return prepared.flatten() if self.flatten else prepared


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


def _normalize_play_key(play_key: Optional[Union[str, Any]]) -> Optional[Any]:
    if play_key is None:
        return None
    if ObjectId is None:  # type: ignore[truthy-bool]
        raise ImportError(
            "ObjectId is required to resolve play keys. "
            "Install pymongo to enable play-key lookups."
        )
    if isinstance(play_key, str):
        return ObjectId(play_key)
    return play_key


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


def _initial_reward_hidden(config: OmegaConf, device: torch.device) -> Optional[Tuple[Tensor, Tensor]]:
    if not bool(config.model.value_prefix):
        return None
    hidden_size = int(config.model.lstm_hidden_size)
    h = torch.zeros(1, 1, hidden_size, device=device)
    c = torch.zeros(1, 1, hidden_size, device=device)
    return h, c


def _human_action_indices(play_doc: Mapping[str, Any]) -> list[int]:
    raw_actions = play_doc.get("actions") or []
    converted: list[int] = []
    for entry in raw_actions:
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and entry:
            name = entry[0]
        else:
            name = entry
        converted.append(ACTION_NAME_TO_INDEX.get(str(name).lower(), 0))
    return converted


def _filter_traces_to_max_step(
    traces: Optional[Mapping[str, list[Mapping[str, Any]]]],
    max_step: int,
) -> Optional[OrderedDict[str, list[Mapping[str, Any]]]]:
    """Return a copy of traces truncated to timesteps < max_step."""
    if traces is None:
        return None

    filtered: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for layer, records in traces.items():
        kept = [rec for rec in records if rec.get("timestep", -1) < max_step]
        if kept:
            filtered[layer] = kept
    return filtered


def _build_head_activation_payload(
    *,
    initial_states: list[Tensor],
    initial_values: list[Tensor],
    initial_policies: list[Tensor],
    projections: list[Tensor],
    projection_heads: list[Tensor],
    recurrent_states: list[Tensor],
    recurrent_values: list[Tensor],
    recurrent_policies: list[Tensor],
    reward_predictions: list[Tensor],
) -> OrderedDict[str, Tensor]:
    payload: OrderedDict[str, Tensor] = OrderedDict()
    if initial_states:
        payload["representation"] = torch.cat(initial_states, dim=0)
    if projections:
        payload["projection"] = torch.cat(projections, dim=0)
    if projection_heads:
        payload["projection_head"] = torch.cat(projection_heads, dim=0)
    if initial_values:
        stacked_values = torch.cat(initial_values, dim=1)
        payload["value_initial"] = stacked_values.permute(1, 0, 2)
    if initial_policies:
        payload["policy_initial"] = torch.cat(initial_policies, dim=0)
    if recurrent_states:
        payload["dynamics"] = torch.cat(recurrent_states, dim=0)
    if recurrent_values:
        stacked_recurrent = torch.cat(recurrent_values, dim=1)
        payload["value_recurrent"] = stacked_recurrent.permute(1, 0, 2)
    if recurrent_policies:
        payload["policy_recurrent"] = torch.cat(recurrent_policies, dim=0)
    if reward_predictions:
        payload["reward"] = torch.cat(reward_predictions, dim=0)
    return payload


def _collect_head_activations(
    extractor: HiddenStateExtractor,
    observations: Tensor,
    play_doc: Mapping[str, Any],
    trace_recorder: Optional[ActivationTraceRecorder] = None,
    human_actions: Optional[Sequence[int]] = None,
) -> OrderedDict[str, Tensor]:
    prepared = _prepare_observations(observations, extractor.config, extractor.device)
    num_states = prepared.shape[0]

    reward_hidden = _initial_reward_hidden(extractor.config, extractor.device)
    initial_states: list[Tensor] = []
    initial_values: list[Tensor] = []
    initial_policies: list[Tensor] = []
    projections: list[Tensor] = []
    projection_heads: list[Tensor] = []
    recurrent_states: list[Tensor] = []
    recurrent_values: list[Tensor] = []
    recurrent_policies: list[Tensor] = []
    reward_predictions: list[Tensor] = []

    human_actions = list(human_actions) if human_actions is not None else _human_action_indices(play_doc)

    with torch.no_grad():
        for idx in range(num_states):
            if trace_recorder:
                trace_recorder.set_context(idx, "initial")
            obs = prepared[idx : idx + 1]
            state, values, policy = extractor.model.initial_inference(obs, training=True)
            initial_states.append(state.detach().cpu())
            initial_values.append(values.detach().cpu())
            initial_policies.append(policy.detach().cpu())

            proj = extractor.model.projection_model(state)
            projections.append(proj.detach().cpu())
            projection_heads.append(extractor.model.projection_head_model(proj).detach().cpu())

            if idx < len(human_actions):
                if trace_recorder:
                    trace_recorder.set_context(idx, "recurrent")
                action_value = torch.tensor(
                    [[human_actions[idx]]],
                    device=extractor.device,
                    dtype=torch.float32,
                )
                next_state, reward_pred, next_values, next_policy, reward_hidden = extractor.model.recurrent_inference(
                    state,
                    action_value,
                    reward_hidden,
                    training=True,
                )
                recurrent_states.append(next_state.detach().cpu())
                reward_predictions.append(reward_pred.detach().cpu())
                recurrent_values.append(next_values.detach().cpu())
                recurrent_policies.append(next_policy.detach().cpu())
            if trace_recorder:
                trace_recorder.clear_context()

    return _build_head_activation_payload(
        initial_states=initial_states,
        initial_values=initial_values,
        initial_policies=initial_policies,
        projections=projections,
        projection_heads=projection_heads,
        recurrent_states=recurrent_states,
        recurrent_values=recurrent_values,
        recurrent_policies=recurrent_policies,
        reward_predictions=reward_predictions,
    )


def _resolve_requested_heads(
    requested: Optional[Sequence[str]],
    available: Mapping[str, Tensor],
) -> Tuple[list[str], list[str]]:
    ordered_available = [head for head in HEAD_OUTPUT_REGISTRY.keys() if head in available]
    if not ordered_available:
        return [], []

    if not requested:
        default_head = "representation" if "representation" in available else ordered_available[0]
        return [default_head], []

    normalized = [head.lower() for head in requested]
    if "all" in normalized:
        skipped = [head for head in HEAD_OUTPUT_REGISTRY.keys() if head not in available]
        return ordered_available, skipped

    unknown = [head for head in normalized if head not in HEAD_OUTPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unrecognized heads requested: {unknown}. Valid options: {list(HEAD_OUTPUT_REGISTRY.keys()) + ['all']}")

    missing = [head for head in normalized if head not in available]
    if missing:
        raise RuntimeError(
            "Requested heads are unavailable for this play (likely due to missing human actions): "
            f"{missing}"
        )

    selected = [head for head in HEAD_OUTPUT_REGISTRY.keys() if head in normalized]
    return selected, []


class VGDLZStateLoader:
    '''Utility for reading VGDL z-states from the ds004323 Mongo dump.'''

    def __init__(self, dataset_root: Union[str, Path], *, rc_rl_root: Optional[Union[str, Path]] = None) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split_plays_dir: Optional[Path] = None
        self.dump_root = self._resolve_dump_root()
        self.rc_rl_root = Path(rc_rl_root).expanduser() if rc_rl_root else _DEFAULT_RC_RL_ROOT
        if self.rc_rl_root and self.rc_rl_root.exists() and str(self.rc_rl_root) not in sys.path:
            sys.path.append(str(self.rc_rl_root))

    def _resolve_dump_root(self) -> Path:
        direct_candidate = self.dataset_root
        if direct_candidate.is_dir() and (direct_candidate / "plays.bson").exists():
            return direct_candidate
        split_candidate = self.dataset_root / "plays"
        if split_candidate.is_dir() and list(split_candidate.rglob("*.bson")):
            self.split_plays_dir = split_candidate
            return direct_candidate
        nested_candidate = self.dataset_root / "behavior" / "dump" / "vgfmri"
        if nested_candidate.is_dir() and (nested_candidate / "plays.bson").exists():
            return nested_candidate
        raise FileNotFoundError(
            "Could not locate a Mongo dump beneath the provided dataset root. "
            "Ensure you have extracted behavior/dump.tar.gz into behavior/dump/, "
            "or provide a dataset root with plays.bson or a plays/ directory of split BSON files."
        )

    def _iter_collection(self, collection: str) -> Iterator[Mapping[str, Any]]:
        _require_bson()
        if collection == "plays" and self.split_plays_dir is not None:
            play_paths = sorted(self.split_plays_dir.rglob("*.bson"))
            if not play_paths:
                raise FileNotFoundError(f"No BSON play files found under {self.split_plays_dir}")
            for path in play_paths:
                with path.open("rb") as handle:
                    yield from decode_file_iter(handle)  # type: ignore[arg-type]
            return
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
        play_key: Optional[Union[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Mapping[str, Any]]:
        count = 0
        normalized_key = _normalize_play_key(play_key)
        for doc in self._iter_collection("plays"):
            if normalized_key is not None and doc.get("_id") != normalized_key:
                continue
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
        play_key: Optional[Union[str, Any]] = None,
    ) -> Mapping[str, Any]:
        try:
            return next(
                self.iter_plays(
                    subj_id=subj_id,
                    run_id=run_id,
                    play_id=play_id,
                    game_name=game_name,
                    play_key=play_key,
                )
            )
        except StopIteration as exc:
            filters = {k: v for k, v in {
                "subj_id": subj_id,
                "run_id": run_id,
                "play_id": play_id,
                "game_name": game_name,
                "play_key": play_key,
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
        play_key: Optional[Union[str, Any]] = None,
        resize: Optional[int] = 84,
        n_stack: int = 1,
        num_channels: int = 1,
        as_tensor: bool = True,
    ) -> Tuple[Any, Mapping[str, Any]]:
        play_doc = self.get_play(
            subj_id=subj_id,
            run_id=run_id,
            play_id=play_id,
            game_name=game_name,
            play_key=play_key,
        )
        frames = self.render_frames(play_doc, resize=resize, num_channels=num_channels)
        stacked = _stack_frame_sequence(frames, n_stack)
        if as_tensor:
            return torch.from_numpy(stacked), play_doc
        return stacked, play_doc


def prepare_model_inputs_from_play(
    extractor: HiddenStateExtractor,
    dataset_root: Union[str, Path],
    *,
    subj_id: Union[str, int],
    run_id: Optional[int] = None,
    play_id: Optional[int] = None,
    game_name: Optional[str] = None,
    play_key: Optional[Union[str, Any]] = None,
    as_tensor: bool = True,
) -> Tuple[Any, Mapping[str, Any]]:
    '''Load VGDL z-states for a single play and format them for an EfficientZero model.'''
    obs_shape = list(extractor.config.env.obs_shape)
    resize = int(obs_shape[1]) if len(obs_shape) > 1 else None
    n_stack = int(extractor.config.env.n_stack)
    frame_channels = _infer_frame_channels(extractor.config)

    loader = VGDLZStateLoader(dataset_root)
    observations, play_doc = loader.load_play_observations(
        subj_id=subj_id,
        run_id=run_id,
        play_id=play_id,
        game_name=game_name,
        play_key=play_key,
        resize=resize,
        n_stack=n_stack,
        num_channels=frame_channels,
        as_tensor=as_tensor,
    )
    return observations, play_doc


def extract_hidden_states_from_play(
    model_path: Union[str, Path],
    dataset_root: Union[str, Path],
    *,
    subj_id: Union[str, int],
    run_id: Optional[int] = None,
    play_id: Optional[int] = None,
    game_name: Optional[str] = None,
    play_key: Optional[Union[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    config_override: Optional[Union[Mapping[str, Any], OmegaConf]] = None,
    as_numpy: bool = True,
) -> Tuple[Any, Mapping[str, Any]]:
    '''Convenience wrapper that loads a checkpoint and returns hidden states for a play.'''
    extractor = load_hidden_state_extractor(
        model_path,
        device=device,
        config_override=config_override,
    )
    observations, play_doc = prepare_model_inputs_from_play(
        extractor,
        dataset_root,
        subj_id=subj_id,
        run_id=run_id,
        play_id=play_id,
        game_name=game_name,
        play_key=play_key,
        as_tensor=True,
    )
    hidden = extractor(observations, as_numpy=as_numpy)
    return hidden, play_doc


def compute_hidden_state_similarity(
    hidden_states: Union[Tensor, 'np.ndarray', Sequence[Any]],
    *,
    flatten: bool = True,
    normalize: bool = False,
    as_numpy: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Union[Tensor, 'np.ndarray']:
    '''Convert a sequence of hidden activations into a Gram (similarity) matrix.'''
    if isinstance(hidden_states, Tensor):
        hidden = hidden_states.detach().cpu()
    elif np is not None and isinstance(hidden_states, np.ndarray):  # type: ignore[truthy-bool]
        hidden = torch.from_numpy(hidden_states)
    else:
        hidden = torch.as_tensor(hidden_states)

    if hidden.ndim < 2:
        raise ValueError(
            f'Expected at least 2 dimensions for hidden states; received shape {tuple(hidden.shape)}'
        )

    hidden = hidden.to(dtype=dtype)
    if flatten and hidden.ndim > 2:
        hidden = hidden.reshape(hidden.shape[0], -1)
    elif hidden.ndim != 2:
        raise ValueError(
            'Hidden states must be 2-D when flatten=False. '
            f'Received tensor with shape {tuple(hidden.shape)}.'
        )

    if normalize:
        norms = hidden.norm(dim=1, keepdim=True).clamp_min(1e-12)
        hidden = hidden / norms

    similarity = hidden @ hidden.T
    return similarity.numpy() if as_numpy else similarity


def compute_similarity_matrix_from_play(
    model_path: Union[str, Path],
    dataset_root: Union[str, Path],
    *,
    subj_id: Union[str, int],
    run_id: Optional[int] = None,
    play_id: Optional[int] = None,
    game_name: Optional[str] = None,
    play_key: Optional[Union[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    config_override: Optional[Union[Mapping[str, Any], OmegaConf]] = None,
    heads: Optional[Sequence[str]] = None,
    flatten: bool = True,
    normalize: bool = False,
    as_numpy: bool = False,
    trace: bool = False,
    trace_flatten: bool = True,
    trace_human_actions_only: bool = False,
    trace_layers: Optional[Sequence[str]] = None,
) -> Tuple[
    OrderedDict[str, Union[Tensor, 'np.ndarray']],
    Mapping[str, Any],
    list[str],
    Optional[OrderedDict[str, list[Mapping[str, Any]]]],
]:
    '''Load a playthrough, run the model, and return similarity matrices per EfficientZero head.'''
    extractor = load_hidden_state_extractor(
        model_path,
        device=device,
        config_override=config_override,
    )
    trace_recorder: Optional[ActivationTraceRecorder] = None
    if trace:
        trace_recorder = ActivationTraceRecorder(
            flatten=trace_flatten,
            allowed_layers=trace_layers,
        )
        trace_recorder.attach(extractor.model)
    observations, play_doc = prepare_model_inputs_from_play(
        extractor,
        dataset_root,
        subj_id=subj_id,
        run_id=run_id,
        play_id=play_id,
        game_name=game_name,
        play_key=play_key,
        as_tensor=True,
    )
    human_actions = _human_action_indices(play_doc)
    activations = _collect_head_activations(
        extractor,
        observations,
        play_doc,
        trace_recorder=trace_recorder,
        human_actions=human_actions,
    )
    selected_heads, skipped_heads = _resolve_requested_heads(heads, activations)
    similarities: OrderedDict[str, Union[Tensor, 'np.ndarray']] = OrderedDict()
    for head in selected_heads:
        head_tensor = activations[head]
        similarities[head] = compute_hidden_state_similarity(
            head_tensor,
            flatten=flatten,
            normalize=normalize,
            as_numpy=as_numpy,
        )
    traces = trace_recorder.traces if trace_recorder else None
    if trace_recorder:
        trace_recorder.close()
    if trace_human_actions_only and traces is not None:
        traces = _filter_traces_to_max_step(traces, len(human_actions))
    return similarities, play_doc, skipped_heads, traces


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute a hidden-state similarity matrix for a VGDL playthrough.',
    )
    parser.add_argument('--model', required=True, help='Path to the EfficientZero checkpoint (.pt).')
    parser.add_argument('--dataset-root', required=True, help='Root directory containing the Mongo dump.')
    parser.add_argument('--subj-id', required=True, help='Subject identifier for the playthrough.')
    parser.add_argument('--run-id', type=int, help='Run index for the requested playthrough.')
    parser.add_argument('--play-id', type=int, help='Play index for the requested playthrough.')
    parser.add_argument(
        '--play-key',
        help='Mongo ObjectId (hex string) for an exact playthrough. '
             'When provided, this is used to select the play document.',
    )
    parser.add_argument('--game-name', help='VGDL game name filter.')
    parser.add_argument(
        '--heads',
        nargs='+',
        choices=list(HEAD_OUTPUT_REGISTRY.keys()) + ['all'],
        help=(
            "EfficientZero heads to process (default: representation). "
            "Pass 'all' to compute attention matrices for every available head."
        ),
    )
    parser.add_argument('--device', help="Device to run the model on, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument(
        '--config-override',
        help='Optional path to a config override (YAML/JSON) or literal dict expression.',
    )
    parser.add_argument('--output', help='Optional file path to save the similarity matrix.')
    parser.add_argument(
        '--output-format',
        choices=('auto', 'pt', 'npy'),
        default='auto',
        help='Serialization format when saving (default: infer from extension).',
    )
    parser.add_argument(
        '--trace-output',
        help='Optional path to save per-layer activation traces across timesteps (.pt).',
    )
    parser.add_argument(
        '--trace-no-flatten',
        dest='trace_flatten',
        action='store_false',
        help='Keep full activation tensors instead of flattening to 1-D feature vectors.',
    )
    parser.set_defaults(trace_flatten=True)
    parser.add_argument(
        '--trace-layers-json',
        help='Optional JSON file with a list of layer names to trace (exact match).',
    )
    parser.add_argument(
        '--trace-human-actions-only',
        action='store_true',
        help='Drop trace records beyond the number of available human actions.',
    )
    parser.add_argument(
        '--no-flatten',
        dest='flatten',
        action='store_false',
        help='Skip flattening spatial hidden dimensions before similarity.',
    )
    parser.set_defaults(flatten=True)
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='L2-normalize hidden vectors before computing similarities.',
    )
    parser.add_argument(
        '--as-numpy',
        action='store_true',
        help='Return the similarity matrix as a numpy array instead of a torch tensor.',
    )
    return parser.parse_args(argv)


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
            'Provide a path to YAML/JSON or a Python literal.'
        )
    return OmegaConf.create(parsed)


def _load_trace_layers(path: Optional[str]) -> Optional[list[str]]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Trace layers JSON not found: {candidate}")
    with candidate.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        if "layers" in data:
            data = data["layers"]
        elif "trace_layers" in data:
            data = data["trace_layers"]
        else:
            raise ValueError("Trace layers JSON must be a list or contain a 'layers' key.")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise ValueError("Trace layers JSON must be a list of layer names.")
    return [str(item) for item in data]


def _resolve_output_format(output_path: Path, explicit: str) -> str:
    if explicit != 'auto':
        return explicit
    suffix = output_path.suffix.lower()
    if suffix == '.npy':
        return 'npy'
    return 'pt'


def _save_similarity_matrix(output_path: Path, similarity: Union[Tensor, 'np.ndarray'], format_hint: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format_hint == 'npy':
        _require_numpy()
        np.save(output_path, similarity)  # type: ignore[arg-type]
    else:
        tensor = similarity if isinstance(similarity, Tensor) else torch.from_numpy(similarity)
        torch.save(tensor, output_path)


def _save_head_similarity_matrices(
    similarities: Mapping[str, Union[Tensor, 'np.ndarray']],
    output_path: Path,
    explicit_format: str,
) -> None:
    if not similarities:
        return

    head_items = list(similarities.items())
    if len(head_items) == 1:
        head, matrix = head_items[0]
        format_hint = _resolve_output_format(output_path, explicit_format)
        _save_similarity_matrix(output_path, matrix, format_hint)
        print(f"Saved {head} similarity matrix to {output_path}")
        return

    base_path = output_path
    if base_path.suffix:
        for head, matrix in head_items:
            head_path = base_path.with_name(f"{base_path.stem}_{head}{base_path.suffix}")
            format_hint = _resolve_output_format(head_path, explicit_format)
            _save_similarity_matrix(head_path, matrix, format_hint)
            print(f"Saved {head} similarity matrix to {head_path}")
        return

    target_dir = base_path
    target_dir.mkdir(parents=True, exist_ok=True)
    format_hint = explicit_format if explicit_format != 'auto' else 'pt'
    for head, matrix in head_items:
        head_path = target_dir / f"{head}.{format_hint}"
        _save_similarity_matrix(head_path, matrix, format_hint)
        print(f"Saved {head} similarity matrix to {head_path}")


def _save_activation_traces(
    traces: Optional[Mapping[str, list[Mapping[str, Any]]]],
    output_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    if not traces:
        print("No activation traces were recorded; skipping trace save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata),
        "traces": traces,
    }
    try:
        torch.save(payload, output_path)
    except Exception as exc:  # file write failures (quota/full) should not kill the batch
        print(f"[WARN] Failed to save activation traces to {output_path}: {exc}", file=sys.stderr)
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return
    print(f"Saved activation traces to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli_args(argv)
    config_override = _load_config_override(args.config_override)
    trace_layers = _load_trace_layers(args.trace_layers_json)
    try:
        similarities, play_doc, skipped_heads, traces = compute_similarity_matrix_from_play(
            model_path=args.model,
            dataset_root=args.dataset_root,
            subj_id=args.subj_id,
            run_id=args.run_id,
            play_id=args.play_id,
            game_name=args.game_name,
            play_key=args.play_key,
            device=args.device,
            config_override=config_override,
            heads=args.heads,
            flatten=args.flatten,
            normalize=args.normalize,
            as_numpy=args.as_numpy,
            trace=bool(args.trace_output),
            trace_flatten=args.trace_flatten,
            trace_human_actions_only=args.trace_human_actions_only,
            trace_layers=trace_layers,
        )
    except LookupError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
        return

    if not similarities:
        print("No head activations were produced for this play.")
        if skipped_heads:
            print(f"Heads without activations: {', '.join(skipped_heads)}")
        return

    subj = play_doc.get('subj_id')
    run = play_doc.get('run_id')
    play_idx = play_doc.get('play_id')
    game = play_doc.get('game_name')
    play_key = play_doc.get('_id')
    print(
        f'Play metadata - subj_id: {subj}, run_id: {run}, play_id: {play_idx}, '
        f'game_name: {game}, play_key: {play_key}'
    )

    for head, matrix in similarities.items():
        print(f"Head '{head}': similarity matrix shape {tuple(matrix.shape)}")

    if skipped_heads:
        print(f"Heads without activations: {', '.join(skipped_heads)}")

    if args.output:
        output_path = Path(args.output)
        _save_head_similarity_matrices(similarities, output_path, args.output_format)
    if args.trace_output:
        trace_metadata = {
            "subj_id": play_doc.get('subj_id'),
            "run_id": play_doc.get('run_id'),
            "play_id": play_doc.get('play_id'),
            "game_name": play_doc.get('game_name'),
            "play_key": play_doc.get('_id'),
            "trace_flatten": args.trace_flatten,
            "trace_human_actions_only": args.trace_human_actions_only,
            "trace_layers": trace_layers,
        }
        _save_activation_traces(traces, Path(args.trace_output), trace_metadata)


if __name__ == '__main__':
    main()
