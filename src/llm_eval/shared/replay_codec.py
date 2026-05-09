# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Delta-encode / expand the ``states`` array in .replay.json.gz files.

Encoding rule (per-sprite-type):
  - First state: full sprites dict.
  - Subsequent states: only include sprite-type keys whose value differs
    from the previous frame.  If ALL types are unchanged the ``sprites``
    key is omitted entirely.

A top-level ``"delta_encoded": true`` flag marks encoded files.
``expand_delta_states`` is a no-op on files without that flag, so old
files pass through safely.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path


def delta_encode_states(data: dict) -> None:
    """Delta-encode ``data["states"]`` in place.  Adds ``delta_encoded`` flag."""
    if data.get("delta_encoded"):
        raise ValueError("replay data is already delta-encoded")

    states = data.get("states")
    if not states or len(states) < 2:
        return

    prev_by_type: dict[str, list] = states[0]["sprites"]

    for i in range(1, len(states)):
        cur_sprites: dict[str, list] = states[i]["sprites"]
        delta: dict[str, list] = {}

        all_keys = set(cur_sprites) | set(prev_by_type)
        for key in all_keys:
            cur_val = cur_sprites.get(key)
            prev_val = prev_by_type.get(key)
            if cur_val != prev_val:
                delta[key] = cur_val

        prev_by_type = cur_sprites

        if delta:
            states[i]["sprites"] = delta
        else:
            del states[i]["sprites"]

    data["delta_encoded"] = True


def expand_delta_states(data: dict) -> None:
    """Expand delta-encoded ``data["states"]`` in place.  No-op if not encoded."""
    if not data.get("delta_encoded"):
        return

    states = data.get("states")
    if not states or len(states) < 2:
        del data["delta_encoded"]
        return

    prev_sprites: dict[str, list] = states[0]["sprites"]

    for i in range(1, len(states)):
        if "sprites" not in states[i]:
            states[i]["sprites"] = copy.deepcopy(prev_sprites)
        else:
            merged = copy.deepcopy(prev_sprites)
            merged.update(states[i]["sprites"])
            # None sentinel means "type removed in this frame"
            states[i]["sprites"] = {k: v for k, v in merged.items() if v is not None}

        prev_sprites = states[i]["sprites"]

    del data["delta_encoded"]


def load_replay(path: str | Path) -> dict:
    """Load a ``.replay.json.gz`` file, transparently expanding delta-encoded states."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    expand_delta_states(data)
    return data


def save_replay(data: dict, path: str | Path) -> None:
    """Delta-encode states and write a ``.replay.json.gz`` file.

    Does not mutate the input dict.
    """
    data = copy.deepcopy(data)
    delta_encode_states(data)
    json_bytes = json.dumps(data).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(json_bytes)
