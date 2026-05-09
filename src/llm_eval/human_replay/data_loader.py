# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Load human behavioral data from BSON files."""

import zlib
from pathlib import Path

import bson


def decompress_zstates(zstates_binary: bytes) -> list[dict]:
    """
    Decompress zstates from BSON plays collection.

    Args:
        zstates_binary: Binary data from plays.zstates field

    Returns:
        List of state dicts, one per timestep
    """
    decompressed = zlib.decompress(zstates_binary)
    data = bson.decode(decompressed)
    return data["states"]


def read_bson_file(bson_path: Path) -> list[dict]:
    """
    Read BSON file and return list of documents.

    Args:
        bson_path: Path to .bson file

    Returns:
        List of documents
    """
    documents = []
    with open(bson_path, "rb") as f:
        while True:
            doc_iter = bson.decode_file_iter(f)
            doc = next(doc_iter, None)
            if doc is None:
                break
            documents.append(doc)
    return documents


class HumanPlayLoader:
    """Load human gameplay data from behavioral research BSON files."""

    def __init__(self, data_dir: str):
        """
        Initialize the loader.

        Args:
            data_dir: Path to prepare_behavioral_data directory
                      (e.g., ./workdir/prepare_behavioral_data)
        """
        self.data_dir = Path(data_dir)
        self.plays_dir = self.data_dir / "plays"

        if not self.plays_dir.exists():
            raise FileNotFoundError(f"Plays directory not found: {self.plays_dir}")

    def list_subjects(self) -> list[str]:
        """
        List all available subjects.

        Returns:
            List of subject IDs like ['sub-01', 'sub-02', ...]
        """
        subjects = []
        for subdir in sorted(self.plays_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("sub-"):
                subjects.append(subdir.name)
        return subjects

    def list_runs(self, subject: str) -> list[int]:
        """
        List all runs for a subject.

        Args:
            subject: Subject ID (e.g., 'sub-01')

        Returns:
            List of run numbers like [0, 1, 2, ...]
        """
        subj_dir = self.plays_dir / subject
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

        runs = []
        for bson_file in sorted(subj_dir.glob("run-*.bson")):
            # Extract run number from filename like "run-00.bson"
            run_str = bson_file.stem.replace("run-", "")
            runs.append(int(run_str))
        return runs

    def list_plays(self, subject: str, run: int) -> list[dict]:
        """
        List all plays in a run with metadata.

        Args:
            subject: Subject ID (e.g., 'sub-01')
            run: Run number

        Returns:
            List of dicts with keys: game_name, level_id, win, score, play_idx
        """
        bson_path = self.plays_dir / subject / f"run-{run:02d}.bson"
        if not bson_path.exists():
            raise FileNotFoundError(f"Run file not found: {bson_path}")

        documents = read_bson_file(bson_path)
        plays = []
        for idx, doc in enumerate(documents):
            plays.append(
                {
                    "play_idx": idx,
                    "game_name": doc["game_name"],
                    "level_id": doc["level_id"],
                    "win": doc.get("win"),
                    "score": doc.get("score"),
                }
            )
        return plays

    def load_play(
        self, subject: str, run: int, play_idx: int
    ) -> tuple[dict, list[dict]]:
        """
        Load a specific play and its decompressed states.

        Args:
            subject: Subject ID (e.g., 'sub-01')
            run: Run number
            play_idx: Index of play within the run

        Returns:
            Tuple of (play_doc, decompressed_states)
            - play_doc: Full play document with metadata
            - decompressed_states: List of state dicts, one per timestep
        """
        bson_path = self.plays_dir / subject / f"run-{run:02d}.bson"
        if not bson_path.exists():
            raise FileNotFoundError(f"Run file not found: {bson_path}")

        documents = read_bson_file(bson_path)

        if play_idx < 0 or play_idx >= len(documents):
            raise IndexError(
                f"Play index {play_idx} out of range [0, {len(documents) - 1}]"
            )

        play_doc = documents[play_idx]
        zstates_binary = play_doc["zstates"]
        states = decompress_zstates(zstates_binary)

        return play_doc, states

    def get_num_plays(self, subject: str, run: int) -> int:
        """
        Get the number of plays in a run.

        Args:
            subject: Subject ID
            run: Run number

        Returns:
            Number of plays in the run
        """
        bson_path = self.plays_dir / subject / f"run-{run:02d}.bson"
        if not bson_path.exists():
            raise FileNotFoundError(f"Run file not found: {bson_path}")

        return len(read_bson_file(bson_path))
