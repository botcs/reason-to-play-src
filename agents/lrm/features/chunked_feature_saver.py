# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Per-level feature saving with chunking for memory efficiency.

Saves LLM hidden states for all plays of a level into a single .pt file
using torch.save with bf16 tensors for 2x storage savings vs fp32.

Output format:
    out/extract_model_features/
      <prompt_config>/           # e.g., "first_person_all" or "observer_non_wait"
        model-<model_id>/
          <subject>/
            <game_name>/
              level_<level:02d>.pt                 # One PT per level, aggregating all plays
              level_<level:02d>_prompts.jsonl.gz   # Corresponding prompts (gzip JSONL)
"""

import gzip
import json
from pathlib import Path

import torch


class LevelFeatureSaver:
    """Save features for all plays of a level to a single .pt file."""

    def __init__(
        self,
        output_dir: Path | str,
        model_id: str,
        subject: str,
        game_name: str,
        level_id: int,
        prompt_config: str = "",
        chunk_size: int = 100,
        expect_sublayers: bool = False,
    ):
        """
        Initialize level feature saver.

        Args:
            output_dir: Base output directory (e.g., 'out/extract_model_features')
            model_id: Model identifier (sanitized)
            subject: Subject ID (e.g., 'sub-01')
            game_name: Game name (e.g., 'vgfmri4_bait')
            level_id: Level number
            prompt_config: Prompt configuration string (e.g., 'first_person_all')
            chunk_size: Number of steps per chunk for memory efficiency
            expect_sublayers: If True, every append_step() must supply non-None
                sublayer_attn and sublayer_mlp.  If False, those arguments
                must be None.  Enforced end-to-end so a wrapper silently
                dropping the sublayer flag cannot produce a valid-looking
                but incomplete .pt file.
        """
        self.expect_sublayers = expect_sublayers
        self.base_output_dir = Path(output_dir)
        if prompt_config:
            self.output_dir = (
                self.base_output_dir
                / prompt_config
                / f"model-{model_id}"
                / subject
                / game_name
            )
        else:
            self.output_dir = (
                self.base_output_dir / f"model-{model_id}" / subject / game_name
            )
        self.level_id = level_id
        self.output_file = self.output_dir / f"level_{level_id:02d}.pt"
        self.chunk_size = chunk_size

        # Working directory for chunks
        self.work_dir = self.output_dir / f".level_{level_id:02d}_work"
        self.metadata_path = self.work_dir / "metadata.json"

        # Track plays data
        self.plays_data: list[dict] = []
        self.current_play_idx: int = -1

        # Current chunk buffer
        self.current_chunk: list[dict] = []
        self.current_chunk_idx: int = 0

        # In-memory storage for all chunks (avoids disk I/O)
        self.in_memory_chunks: list[dict] = []

        # Prompt records for reproducibility (saved as .jsonl.gz)
        self.prompt_records: list[dict] = []
        self.prompts_file = self.output_dir / f"level_{level_id:02d}_prompts.jsonl.gz"

        # Load existing metadata if resuming
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing metadata if available."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            self.plays_data = self.metadata["plays_data"]
            self.current_chunk_idx = self.metadata["current_chunk_idx"]
        else:
            self.metadata = {
                "level_id": self.level_id,
                "completed_chunks": [],
                "plays_data": [],
                "current_chunk_idx": 0,
                "finalized": False,
            }

    def _save_metadata(self) -> None:
        """Save metadata to JSON."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metadata["plays_data"] = [
            {k: v for k, v in p.items() if k != "steps"} for p in self.plays_data
        ]
        self.metadata["current_chunk_idx"] = self.current_chunk_idx
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def is_finalized(self) -> bool:
        """Check if this level has already been finalized."""
        return self.metadata["finalized"]

    def get_completed_plays(self) -> list[str]:
        """Get list of play_ids that have been completed."""
        return [p["play_id"] for p in self.plays_data if p["completed"]]

    def start_play(
        self,
        play_idx: int,
        play_id: str,
        win: bool,
        score: int,
        run_id: int,
    ) -> None:
        """
        Start collecting features for a new play.

        Args:
            play_idx: Index of this play within the level (0, 1, 2, ...)
            play_id: Unique play identifier (e.g., 'sub-01_5_0')
            win: Whether this play was a win
            score: Final score for this play
            run_id: Run number containing this play
        """
        self.current_play_idx = play_idx
        self.plays_data.append(
            {
                "play_idx": play_idx,
                "play_id": play_id,
                "win": win,
                "score": score,
                "run_id": run_id,
                "steps": [],
                "completed": False,
            }
        )
        self._save_metadata()

    def append_step(
        self,
        step_num: int,
        hidden_states: torch.Tensor,
        realworld_ts: float,
        prompt_record: dict,
        sublayer_attn: torch.Tensor | None = None,
        sublayer_mlp: torch.Tensor | None = None,
    ) -> None:
        """
        Add step features to current play.

        Args:
            step_num: Step number in the play
            hidden_states: Hidden states (num_layers+1, hidden_dim), bf16 tensor
            realworld_ts: Wall-clock Unix timestamp from the original fMRI experiment
            prompt_record: Full prompt record dict (with messages) for reproducibility
            sublayer_attn: Pre-residual attention outputs (num_layers, hidden_dim), optional
            sublayer_mlp: Pre-residual MLP outputs (num_layers, hidden_dim), optional
        """
        if self.current_play_idx < 0 or not self.plays_data:
            raise RuntimeError("Must call start_play() before append_step()")

        if self.expect_sublayers:
            if sublayer_attn is None or sublayer_mlp is None:
                raise ValueError(
                    "expect_sublayers=True but sublayer_attn/sublayer_mlp are None. "
                    "Check the extraction wrapper is actually returning sublayer features."
                )
        else:
            if sublayer_attn is not None or sublayer_mlp is not None:
                raise ValueError(
                    "expect_sublayers=False but sublayer_attn/sublayer_mlp were passed. "
                    "Enable expect_sublayers at construction time to save them."
                )

        step_data = {
            "play_idx": self.current_play_idx,
            "step_num": step_num,
            "hidden_states": hidden_states,
            "realworld_ts": realworld_ts,
        }
        if self.expect_sublayers:
            step_data["sublayer_attn"] = sublayer_attn
            step_data["sublayer_mlp"] = sublayer_mlp

        self.current_chunk.append(step_data)

        self.prompt_records.append(prompt_record)

        # Save chunk if full
        if len(self.current_chunk) >= self.chunk_size:
            self._save_current_chunk()

    def end_play(self) -> None:
        """Mark current play as completed."""
        if self.plays_data and self.current_play_idx >= 0:
            self.plays_data[-1]["completed"] = True
            self._save_metadata()

    def _save_current_chunk(self) -> None:
        """Save current chunk to in-memory storage."""
        if not self.current_chunk:
            return

        # Stack features into tensors
        play_indices = torch.tensor(
            [s["play_idx"] for s in self.current_chunk], dtype=torch.int32
        )
        step_nums = torch.tensor(
            [s["step_num"] for s in self.current_chunk], dtype=torch.int32
        )
        hidden_states = torch.stack([s["hidden_states"] for s in self.current_chunk])
        realworld_ts = torch.tensor(
            [s["realworld_ts"] for s in self.current_chunk], dtype=torch.float64
        )

        chunk_data = {
            "play_indices": play_indices,
            "step_nums": step_nums,
            "hidden_states": hidden_states,
            "realworld_ts": realworld_ts,
        }

        if self.expect_sublayers:
            chunk_data["sublayer_attn"] = torch.stack(
                [s["sublayer_attn"] for s in self.current_chunk]
            )
            chunk_data["sublayer_mlp"] = torch.stack(
                [s["sublayer_mlp"] for s in self.current_chunk]
            )

        # Store in memory instead of writing to disk
        self.in_memory_chunks.append(chunk_data)

        # Update metadata (for tracking purposes)
        self.metadata["completed_chunks"].append(self.current_chunk_idx)
        self.current_chunk_idx += 1

        # Reset buffer
        self.current_chunk = []

    def flush(self) -> None:
        """Save any remaining data in current chunk."""
        if self.current_chunk:
            self._save_current_chunk()

    def finalize(
        self,
        model_type: str,
        prompt_variant: str,
        subject: str | None = None,
        extraction_meta: dict | None = None,
    ) -> Path:
        """
        Save all plays to single .pt file with bf16 tensors.

        Args:
            model_type: Model class name
            prompt_variant: Prompt variant used
            subject: Subject ID (optional, for metadata)
            extraction_meta: Provenance metadata (git hash, command, etc.) to embed in .pt file

        Returns:
            Path to final .pt file
        """
        self.flush()

        # If already finalized, return existing path
        if self.metadata["finalized"]:
            return self.output_file

        # Process all in-memory chunks and organize by play
        plays_hidden_states: dict[int, list] = {}
        plays_step_nums: dict[int, list] = {}
        plays_realworld_ts: dict[int, list] = {}
        plays_sublayer_attn: dict[int, list] = {}
        plays_sublayer_mlp: dict[int, list] = {}

        for chunk_data in self.in_memory_chunks:
            play_indices = chunk_data["play_indices"]
            step_nums = chunk_data["step_nums"]
            hidden_states = chunk_data["hidden_states"]
            realworld_ts = chunk_data["realworld_ts"]
            if self.expect_sublayers:
                sublayer_attn = chunk_data["sublayer_attn"]
                sublayer_mlp = chunk_data["sublayer_mlp"]

            for i, play_idx in enumerate(play_indices):
                play_idx = int(play_idx)
                if play_idx not in plays_hidden_states:
                    plays_hidden_states[play_idx] = []
                    plays_step_nums[play_idx] = []
                    plays_realworld_ts[play_idx] = []
                    plays_sublayer_attn[play_idx] = []
                    plays_sublayer_mlp[play_idx] = []

                plays_hidden_states[play_idx].append(hidden_states[i])
                plays_step_nums[play_idx].append(step_nums[i])
                plays_realworld_ts[play_idx].append(realworld_ts[i])
                if self.expect_sublayers:
                    plays_sublayer_attn[play_idx].append(sublayer_attn[i])
                    plays_sublayer_mlp[play_idx].append(sublayer_mlp[i])

        if not plays_hidden_states:
            raise ValueError("No features to finalize")

        # Build save dict
        save_dict = {}
        play_ids = []

        for play_data in self.plays_data:
            play_idx = play_data["play_idx"]
            if play_idx not in plays_hidden_states:
                continue

            play_ids.append(play_data["play_id"])

            # Stack hidden states: (num_steps, num_layers+1, hidden_dim)
            all_hidden_states = torch.stack(plays_hidden_states[play_idx])
            num_steps, num_layers, hidden_dim = all_hidden_states.shape

            # Save per-layer activations as bf16
            for layer_idx in range(num_layers):
                save_dict[f"play_{play_idx}_model_layer_{layer_idx}"] = (
                    all_hidden_states[:, layer_idx, :]
                )

            # Save sublayer (pre-residual) activations
            if self.expect_sublayers:
                all_sublayer_attn = torch.stack(
                    plays_sublayer_attn[play_idx]
                )  # (num_steps, num_layers, hidden_dim)
                all_sublayer_mlp = torch.stack(plays_sublayer_mlp[play_idx])
                num_sublayers = all_sublayer_attn.shape[1]
                for layer_idx in range(num_sublayers):
                    save_dict[f"play_{play_idx}_sublayer_attn_layer_{layer_idx}"] = (
                        all_sublayer_attn[:, layer_idx, :]
                    )
                    save_dict[f"play_{play_idx}_sublayer_mlp_layer_{layer_idx}"] = (
                        all_sublayer_mlp[:, layer_idx, :]
                    )

            # Behavioral data
            save_dict[f"play_{play_idx}_behavioral_win"] = play_data["win"]
            save_dict[f"play_{play_idx}_behavioral_score"] = play_data["score"]
            save_dict[f"play_{play_idx}_behavioral_timestamps"] = torch.stack(
                plays_step_nums[play_idx]
            )
            save_dict[f"play_{play_idx}_behavioral_realworld_ts"] = torch.stack(
                plays_realworld_ts[play_idx]
            )
            save_dict[f"play_{play_idx}_behavioral_num_states"] = num_steps

            # Metadata
            save_dict[f"play_{play_idx}_metadata_play_id"] = play_data["play_id"]
            save_dict[f"play_{play_idx}_metadata_run_id"] = play_data["run_id"]
            save_dict[f"play_{play_idx}_metadata_level_id"] = self.level_id
            save_dict[f"play_{play_idx}_metadata_model_type"] = model_type

        # Top-level metadata
        save_dict["play_ids"] = play_ids  # Keep as list of strings
        save_dict["num_plays"] = len(play_ids)
        save_dict["level_id"] = self.level_id
        save_dict["prompt_variant"] = prompt_variant
        if self.expect_sublayers:
            save_dict["has_sublayers"] = True
        if subject:
            save_dict["subject"] = subject
        if extraction_meta is not None:
            save_dict["extraction_meta"] = extraction_meta

        # Save with torch.save (bf16 tensors = 2x storage savings vs fp32)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, self.output_file)

        # Save prompt records as gzip-compressed JSONL for reproducibility
        if self.prompt_records:
            with gzip.open(self.prompts_file, "wt", encoding="utf-8") as f:
                for record in self.prompt_records:
                    f.write(json.dumps(record) + "\n")
            prompts_size_mb = self.prompts_file.stat().st_size / 1024 / 1024
            print(
                f"  Saved prompts: {self.prompts_file} ({prompts_size_mb:.2f} MB, {len(self.prompt_records)} records)"
            )

        # Mark as finalized
        self.metadata["finalized"] = True
        self._save_metadata()

        # Log file size
        file_size_mb = self.output_file.stat().st_size / 1024 / 1024
        print(
            f"Finalized level features: {self.output_file} ({file_size_mb:.2f} MB, {len(play_ids)} plays)"
        )

        return self.output_file

    def cleanup_chunks(self) -> None:
        """Clear in-memory chunks and any disk work directory after finalization."""
        if not self.metadata["finalized"]:
            print("Warning: Cannot cleanup chunks before finalization")
            return

        # Clear in-memory chunks and prompt records
        self.in_memory_chunks = []
        self.prompt_records = []

        # Also clean up any disk-based work directory (for backwards compatibility
        # or if disk-based chunks were loaded from a resumed session)
        import shutil

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
