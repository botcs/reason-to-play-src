#!/usr/bin/env python3
"""Run the preserved EfficientZero implementation with explicit source paths.

This wrapper sets the historical two-repository environment without modifying
upstream source or substituting the LLM VGDL engine. Use a dedicated EfficientZero
Python environment and pass upstream arguments after ``--``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {
    "train": ("efficientzero", "ez/train.py"),
    "features": ("extraction", "get_efficientzero_activations.py"),
    "traces": ("extraction", "get_attention_matrix.py"),
}


def command(
    mode: str, checkouts: Path, args: list[str], rc_rl_dir: Path | None = None
) -> tuple[list[str], dict, Path]:
    sources = json.loads((ROOT / "sources.json").read_text())["sources"]
    source_name, entry = ENTRYPOINTS[mode]
    required = {"efficientzero"} if mode == "train" else set()
    for name in sorted(required):
        path = checkouts / name
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != sources[name]["revision"]:
            raise ValueError(
                f"{name}: expected {sources[name]['revision']}, found {actual}"
            )
    engine = (rc_rl_dir or ROOT / "vendor" / "rc_rl" / "ez").resolve()
    provenance = engine / "PROVENANCE.json"
    if provenance.is_file():
        record = json.loads(provenance.read_text())
        if record["commit"] != sources["rc-rl-ez"]["revision"]:
            raise ValueError(
                "Vendored RC_RL revision does not match the EZ source lock"
            )
        for item in record["files"]:
            if sha256(engine / item["path"]) != item["sha256"]:
                raise ValueError(f"RC_RL source checksum mismatch: {item['path']}")
    else:
        actual = subprocess.check_output(
            ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != sources["rc-rl-ez"]["revision"]:
            raise ValueError(
                "External RC_RL revision does not match the EZ source lock"
            )
        if subprocess.call(["git", "-C", str(engine), "diff", "--quiet", "HEAD", "--"]):
            raise ValueError("External RC_RL checkout has tracked changes")
    source_dir = (
        ROOT / "extraction" / "efficientzero"
        if source_name == "extraction"
        else checkouts / source_name
    )
    script = source_dir / entry
    if not script.is_file():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env["EZ_ROOT"] = str(
        checkouts / "efficientzero"
        if mode == "train"
        else ROOT / "vendor" / "efficientzero"
    )
    # Historical wrappers use both names; point both at the EZ-specific engine.
    env["RC_RL_ROOT"] = env["RC_RL_PATH"] = str(engine)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            env["RC_RL_ROOT"],
            env["EZ_ROOT"],
            str(ROOT.parent / "src"),
            str(ROOT.parent),  # The translated-game parser lives in src.vgdl.
            env.get("PYTHONPATH", ""),
        ]
    )
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env.setdefault("MPLBACKEND", "Agg")
    # The child uses the extraction directory as cwd. Resolve dataset roots
    # against the caller's cwd before crossing that boundary.
    args = list(args)
    for index, value in enumerate(args):
        if value == "--dataset-root" and index + 1 < len(args):
            args[index + 1] = str(Path(args[index + 1]).expanduser().resolve())
        elif value.startswith("--dataset-root="):
            args[index] = "--dataset-root=" + str(
                Path(value.split("=", 1)[1]).expanduser().resolve()
            )
    return [sys.executable, str(script), *args], env, source_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_trace_provenance(
    argv: list[str], cwd: Path, args: list[str], engine: Path
) -> None:
    options = argparse.ArgumentParser(add_help=False)
    for name in (
        "model",
        "trace-output",
        "config-override",
        "trace-layers-json",
        "play-key",
    ):
        options.add_argument("--" + name)
    selected, _ = options.parse_known_args(args)
    if not selected.trace_output:
        return

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else cwd / path

    output = resolve(selected.trace_output)
    if not output.is_file():
        raise FileNotFoundError(f"Extractor did not create requested traces: {output}")
    checkpoint = resolve(selected.model)
    run_dir = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "models"
        else checkpoint.parent
    )
    configuration = (
        resolve(selected.config_override)
        if selected.config_override
        else run_dir / "logs" / "Train.log"
    )
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration_sha256": sha256(configuration)
        if configuration.is_file()
        else None,
        "configuration_override_literal": selected.config_override
        if selected.config_override and not configuration.is_file()
        else None,
        "extractor_sha256": sha256(Path(argv[1])),
        "trace_layers_sha256": sha256(resolve(selected.trace_layers_json))
        if selected.trace_layers_json
        else None,
        "trace_sha256": sha256(output),
        "source_lock": json.loads((ROOT / "sources.json").read_text())["sources"][
            "rc-rl-ez"
        ],
        "engine_source": json.loads((engine / "PROVENANCE.json").read_text())
        if (engine / "PROVENANCE.json").is_file()
        else {
            "revision": subprocess.check_output(
                ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
            ).strip()
        },
        "model_source": json.loads(
            (ROOT / "vendor/efficientzero/PROVENANCE.json").read_text()
        ),
        "extractor_source": json.loads(
            (ROOT / "extraction/efficientzero/PROVENANCE.json").read_text()
        ),
        "requested_play_key": selected.play_key,
    }
    destination = output.with_suffix(output.suffix + ".provenance.json")
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved extraction provenance to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=ENTRYPOINTS)
    parser.add_argument("--checkouts", type=Path, default=ROOT / "checkouts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rc-rl-dir",
        type=Path,
        help="Override the bundled, checksum-verified RC_RL EZ engine",
    )
    known, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    argv, env, cwd = command(
        known.mode, known.checkouts.resolve(), forwarded, known.rc_rl_dir
    )
    if known.dry_run:
        print(
            json.dumps(
                {
                    "argv": argv,
                    "cwd": str(cwd),
                    "environment": {
                        key: env[key] for key in ("EZ_ROOT", "RC_RL_ROOT", "RC_RL_PATH")
                    },
                },
                indent=2,
            )
        )
        return
    status = subprocess.call(argv, env=env, cwd=cwd)
    if status == 0 and known.mode == "traces" and "--help" not in forwarded:
        record_trace_provenance(argv, cwd, forwarded, Path(env["RC_RL_ROOT"]))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
