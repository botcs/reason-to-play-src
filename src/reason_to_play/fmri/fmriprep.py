#!/usr/bin/env python3
"""Run the per-subject fMRIPrep 24.1.0 configuration with a local SIF."""

import argparse
import hashlib
import json
import logging
import re
import shlex
import subprocess
from pathlib import Path

import tomllib

PROVENANCE = Path(__file__).resolve().parents[1] / "resources" / "fmriprep"
LOGGER = logging.getLogger(__name__)


def replay_config_text(config):
    """Drop prior invocation bookkeeping while retaining preprocessing settings."""
    stale = {
        "bids_database_dir",
        "run_uuid",
        "layout",
        "bids_description_hash",
        "log_dir",
    }
    section = None
    lines = []
    for line in config.read_text().splitlines():
        if line.startswith("["):
            section = line.strip("[]")
        if section == "execution" and line.split("=", 1)[0].strip() in stale:
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def preprocessing_arguments(cfg):
    """Override 24.1.0 argparse defaults, which otherwise overwrite the TOML."""
    workflow = cfg["workflow"]
    arguments = []
    for flag, key in (
        ("--bold2anat-dof", "bold2anat_dof"),
        ("--bold2anat-init", "bold2anat_init"),
        ("--dummy-scans", "dummy_scans"),
        ("--slice-time-ref", "slice_time_ref"),
        ("--skull-strip-t1w", "skull_strip_t1w"),
        ("--skull-strip-template", "skull_strip_template"),
        ("--fd-spike-threshold", "regressors_fd_th"),
        ("--dvars-spike-threshold", "regressors_dvars_th"),
        ("--me-t2s-fit-method", "me_t2s_fit_method"),
        ("--level", "level"),
    ):
        arguments.extend([flag, str(workflow[key])])
    for flag, key, enabled in (
        ("--force-bbr", "use_bbr", True),
        ("--force-no-bbr", "use_bbr", False),
        ("--fs-no-reconall", "run_reconall", False),
        ("--no-submm-recon", "hires", False),
        ("--return-all-components", "regressors_all_comps", True),
        ("--skull-strip-fixed-seed", "skull_strip_fixed_seed", True),
        ("--medial-surface-nan", "medial_surface_nan", True),
    ):
        if workflow[key] is enabled:
            arguments.append(flag)
    arguments.extend(["--output-spaces", *cfg["execution"]["output_spaces"].split()])
    arguments.extend(["--random-seed", str(cfg["seeds"]["master"])])
    for flag, key in (("--nprocs", "nprocs"), ("--omp-nthreads", "omp_nthreads")):
        arguments.extend([flag, str(cfg["nipype"][key])])
    arguments.extend(["--mem-mb", str(int(cfg["nipype"]["memory_gb"] * 1000))])
    for flag, section, key in (
        ("--write-graph", "execution", "write_graph"),
        ("--resource-monitor", "nipype", "resource_monitor"),
        ("--stop-on-first-crash", "nipype", "stop_on_first_crash"),
    ):
        if cfg[section][key]:
            arguments.append(flag)
    return arguments


def command(args):
    manifest = json.loads((PROVENANCE / "manifest.json").read_text())
    config = PROVENANCE / manifest["selected_configs"][args.subject]
    expected = next(
        record["sha256"]
        for record in manifest["configs"]
        if record["path"] == manifest["selected_configs"][args.subject]
    )
    if hashlib.sha256(config.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Configuration checksum does not match manifest: {config}")
    cfg = tomllib.loads(config.read_text())
    if cfg["environment"]["version"] != "24.1.0":
        raise ValueError("Only fMRIPrep 24.1.0 configurations are supported")
    mounts = [
        (args.bids_dir, "/data", "ro"),
        (args.output_dir, "/out", "rw"),
        (args.work_dir, "/work", "rw"),
        (args.fs_license, "/license/license.txt", "ro"),
        (args.templateflow_dir, "/home/fmriprep/.cache/templateflow", "rw"),
        (args.work_dir / "reproduction-config.toml", "/config/fmriprep.toml", "ro"),
    ]
    result = [args.runtime, "run", "--cleanenv"]
    for source, target, mode in mounts:
        source = source.resolve()
        if ":" in str(source) or "," in str(source):
            raise ValueError(
                f"Container bind paths cannot contain ':' or ',': {source}"
            )
        result.extend(["--bind", f"{source}:{target}:{mode}"])
    result.extend(
        [
            str(args.image.resolve()),
            "/data",
            "/out",
            "participant",
            "--config-file",
            "/config/fmriprep.toml",
            "--participant-label",
            args.subject.removeprefix("sub-"),
            "--fs-license-file",
            "/license/license.txt",
            "--work-dir",
            "/work",
        ]
    )
    result.extend(preprocessing_arguments(cfg))
    return result, config


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    for name in (
        "bids-dir",
        "output-dir",
        "work-dir",
        "fs-license",
        "templateflow-dir",
        "image",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--runtime", choices=["apptainer", "singularity"], default="apptainer"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"sub-(0[1-9]|[12][0-9]|3[0-2])", args.subject):
        parser.error("--subject must be sub-01 through sub-32")
    argv, config = command(args)
    LOGGER.info("Configuration: %s", config)
    LOGGER.info("Command: %s", shlex.join(argv))
    if args.dry_run:
        return
    for required in (
        args.image,
        args.fs_license,
        args.bids_dir / "dataset_description.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (args.bids_dir / args.subject).is_dir():
        raise FileNotFoundError(args.bids_dir / args.subject)
    description = json.loads((args.bids_dir / "dataset_description.json").read_text())
    if description["DatasetDOI"] != "doi:10.18112/openneuro.ds004323.v1.0.0":
        raise ValueError(
            "Input BIDS data must identify OpenNeuro ds004323 version 1.0.0"
        )
    for directory in (args.output_dir, args.work_dir, args.templateflow_dir):
        directory.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(
        [args.runtime, "run", "--cleanenv", str(args.image.resolve()), "--version"],
        text=True,
    ).strip()
    if not re.search(r"\b24\.1\.0\b", version):
        raise ValueError(f"Expected fMRIPrep 24.1.0, got {version!r}")
    with args.image.open("rb") as image_file:
        image_sha256 = hashlib.file_digest(image_file, "sha256").hexdigest()
    record = {
        "argv": argv,
        "version": version,
        "image_sha256": image_sha256,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "replay_config_sha256": hashlib.sha256(
            replay_config_text(config).encode()
        ).hexdigest(),
        "raw_dataset_version": "ds004323/1.0.0",
        "status": "started",
    }
    record_path = args.output_dir / f"{args.subject}_reproduction-command.json"
    if record_path.exists():
        raise FileExistsError(f"Refusing to overwrite run provenance: {record_path}")
    (args.work_dir / "reproduction-config.toml").write_text(replay_config_text(config))
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        subprocess.run(argv, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    record["status"] = "completed"
    record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
