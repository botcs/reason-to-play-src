# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Load human JSON recordings for replay and extraction."""

from reason_to_play.data.behavior import behavior_root, iter_plays, play_states
from reason_to_play.data.replay_behavior import read_record, replay_paths


class HumanPlayLoader:
    """Read measured human plays from JSON recordings."""

    def __init__(self, data_dir: str):
        """
        Initialize the loader.

        Args:
            data_dir: Canonical behavior/human directory or dataset root.
        """
        self.data_dir = behavior_root(data_dir)
        if not replay_paths(self.data_dir):
            raise FileNotFoundError(
                f"No human JSON recordings found in {self.data_dir}"
            )
        self.per_game = True
        self.canonical = True
        self._cached_run = None
        self._cached_documents = None

    def _load_run(self, subject: str, run: int) -> list[dict]:
        key = (subject, run)
        if self._cached_run != key:
            documents = list(iter_plays(self.data_dir, subject=subject, run=run))
            self._cached_run = key
            self._cached_documents = documents
        return self._cached_documents

    def list_subjects(self) -> list[str]:
        """
        List all available subjects.

        Returns:
            List of subject IDs like ['sub-01', 'sub-02', ...]
        """
        return sorted(
            {
                read_record(path, expand=False)["subject"]
                for path in replay_paths(self.data_dir)
            }
        )

    def list_runs(self, subject: str) -> list[int]:
        """
        List all runs for a subject.

        Args:
            subject: Subject ID (e.g., 'sub-01')

        Returns:
            List of run numbers like [0, 1, 2, ...]
        """
        return sorted(
            {
                int(play["run_id"])
                for path in replay_paths(self.data_dir, subject)
                for play in read_record(path, expand=False)["plays"]
            }
        )

    def list_plays(self, subject: str, run: int) -> list[dict]:
        """
        List all plays in a run with metadata.

        Args:
            subject: Subject ID (e.g., 'sub-01')
            run: Run number

        Returns:
            List of dicts with keys: game_name, level_id, win, score, play_idx
        """
        documents = self._load_run(subject, run)
        plays = []
        for doc in documents:
            plays.append(
                {
                    "play_idx": doc["_canonical"]["source_document_index"],
                    "game_name": doc["game_name"],
                    "level_id": doc["level_id"],
                    "win": doc.get("win"),
                    "outcome": doc.get("_canonical", {}).get("outcome"),
                    "score": doc.get("score"),
                    "source_play_id": str(doc["_id"]),
                    "source_recording": doc["_canonical"]["source_recording"],
                }
            )
        return plays

    def load_play(
        self, subject: str, run: int, play_idx: int
    ) -> tuple[dict, list[dict]]:
        """
        Load a specific play and its measured states.

        Args:
            subject: Subject ID (e.g., 'sub-01')
            run: Run number
            play_idx: Index of play within the run

        Returns:
            Tuple of (play_doc, states)
            - play_doc: Full play document with metadata
            - states: List of state dicts, one per timestep
        """
        documents = self._load_run(subject, run)

        matching = [
            doc
            for doc in documents
            if doc["_canonical"]["source_document_index"] == play_idx
        ]
        if len(matching) != 1:
            raise IndexError(
                f"Original play ordinal {play_idx} not found uniquely in {subject} run {run}"
            )
        return matching[0], play_states(matching[0])

    def get_num_plays(self, subject: str, run: int) -> int:
        """
        Get the number of plays in a run.

        Args:
            subject: Subject ID
            run: Run number

        Returns:
            Number of plays in the run
        """
        return len(self._load_run(subject, run))
