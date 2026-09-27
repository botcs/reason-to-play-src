#!/usr/bin/env python3
"""
Split MongoDB BSON dump files by subject/run for efficient AWS access.

Stage: prepare_behavioral_data
Input: OpenNeuro behavioral BSON dumps
Output: Split BSON files organized by subject/run

Usage:
    python prepare_behavioral_data.py \\
        --input-dir ~/Downloads/ds004323-download/behavior/dump/heroku_7lzprs54 \\
        --output-dir ./workdir/prepare_behavioral_data
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import bson

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def split_plays_bson(bson_path: Path, output_dir: Path):
    """
    Split plays.bson into separate files per subject/run.

    Args:
        bson_path: Path to plays.bson
        output_dir: Output directory
    """
    logging.info(f"Reading {bson_path.name}...")

    plays_by_run = defaultdict(list)

    total_processed = 0
    with open(bson_path, "rb") as f:
        for doc in bson.decode_file_iter(f):
            key = (int(doc["subj_id"]), doc["run_id"])
            plays_by_run[key].append(doc)

            total_processed += 1
            if total_processed % 100 == 0:
                logging.info(
                    f"  Processed {total_processed} plays, {len(plays_by_run)} unique (subject, run) pairs"
                )

    logging.info(f"  Total plays: {total_processed}")
    logging.info(f"  Unique (subject, run) pairs: {len(plays_by_run)}")

    logging.info("  Writing BSON files per subject/run...")
    for i, ((subj_id, run_id), plays) in enumerate(plays_by_run.items(), 1):
        subj_str = f"sub-{subj_id:02d}"
        run_str = f"run-{run_id:02d}"

        if i % 10 == 0 or i == len(plays_by_run):
            logging.info(
                f"    Writing {i}/{len(plays_by_run)}: {subj_str}/{run_str} ({len(plays)} plays)"
            )

        bson_data = b"".join([bson.encode(play) for play in plays])

        subject_dir = output_dir / "plays" / subj_str
        subject_dir.mkdir(parents=True, exist_ok=True)

        output_file = subject_dir / f"{run_str}.bson"
        with open(output_file, "wb") as f:
            f.write(bson_data)

        size_mb = len(bson_data) / 1024 / 1024
        logging.info(f"      Wrote {size_mb:.2f} MB")


def copy_runs_bson(bson_path: Path, output_dir: Path):
    """
    Copy runs.bson as-is (small enough to not need splitting).

    Args:
        bson_path: Path to runs.bson
        output_dir: Output directory
    """
    import shutil

    logging.info(f"Copying {bson_path.name}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "runs.bson"
    shutil.copy(bson_path, output_file)

    size_mb = output_file.stat().st_size / 1024 / 1024
    logging.info(f"  Copied {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split BSON dump files by subject/run (prepare_behavioral_data stage)"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing BSON files (plays.bson, runs.bson)",
    )
    parser.add_argument(
        "--output-dir",
        default="./workdir/prepare_behavioral_data",
        help="Output directory (default: ./workdir/prepare_behavioral_data)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    logging.info("Preparing behavioral data")
    logging.info(f"  Input: {input_dir}")
    logging.info(f"  Output: {output_dir}")

    plays_bson = input_dir / "plays.bson"
    if not plays_bson.exists():
        raise ValueError(f"plays.bson not found: {plays_bson}")

    runs_bson = input_dir / "runs.bson"
    if not runs_bson.exists():
        raise ValueError(f"runs.bson not found: {runs_bson}")

    split_plays_bson(plays_bson, output_dir)
    copy_runs_bson(runs_bson, output_dir)

    logging.info("Split complete!")
    logging.info(f"All done! Data saved to {output_dir}")
