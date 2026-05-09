# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
#!/usr/bin/env python
"""
Generate text observations using vLLM from pre-generated prompts.

This is Step 1 of the split pipeline:
1. generate_observations.py - vLLM text generation (this script)
2. extract_features.py - HF feature extraction

Usage:
    # Generate observations for a single prompts file
    python -m src.llm_eval.human_replay.generate_observations \
        --prompts out/prompts/sub-01_vgfmri4_bait_observer_non_wait.jsonl \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --batch-size 32

    # Generate observations for multiple prompts files
    python -m src.llm_eval.human_replay.generate_observations \
        --prompts out/prompts/sub-01_*.jsonl out/prompts/sub-02_*.jsonl \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --batch-size 32

Output:
    Creates: out/human_replay_observations/{prompt_type}/model-{model_id}/{subject}/{game}/observations.jsonl
    Each line is a JSON object with:
    - prompt_idx: index in original prompts file
    - prompt_text: formatted prompt from chat template
    - prompt_token_len: number of tokens in prompt
    - response: generated text
    - input_tokens: total input tokens
    - output_tokens: generated tokens
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add repo path for imports
repo_path = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/"
sys.path.insert(0, repo_path)

from src.llm_eval.human_replay.prompt_utils import load_prompts
from src.llm_eval.human_replay.feature_saver import sanitize_model_id


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    else:
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours}h {mins}m {secs}s"


def parse_prompts_filename(path: Path) -> dict:
    """Parse prompts filename into components.

    Format: {subject}_{game}_{prompt_type}.jsonl
    where prompt_type is 3 parts: first_person_all or third_person_all

    Example: sub-01_vgfmri3_zelda_first_person_all.jsonl
    Returns: {subject: 'sub-01', game: 'vgfmri3_zelda', prompt_type: 'first_person_all'}
    """
    name = path.stem
    parts = name.split("_")

    subject = parts[0]
    prompt_type = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
    game = "_".join(parts[1:-3])

    return {"subject": subject, "game": game, "prompt_type": prompt_type}


def get_output_path(prompts_file: Path, output_dir: str, model_id: str) -> Path:
    """Get output path for a prompts file.

    Structure: {output_dir}/{prompt_type}/model-{model_id}/{subject}/{game}/observations.jsonl
    """
    info = parse_prompts_filename(prompts_file)
    return (
        Path(output_dir)
        / info["prompt_type"]
        / f"model-{model_id}"
        / info["subject"]
        / info["game"]
        / "observations.jsonl"
    )


def truncate_messages_to_fit(
    messages: list[dict],
    tokenizer,
    max_tokens: int,
) -> list[dict]:
    """
    Truncate messages to fit within max_tokens.

    Truncates the content of the last user message if the total prompt
    exceeds max_tokens.

    Args:
        messages: List of {'role': str, 'content': str} dicts
        tokenizer: Tokenizer with apply_chat_template method
        max_tokens: Maximum allowed tokens for the prompt

    Returns:
        Possibly truncated messages list
    """
    # Check current length
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    current_len = len(token_ids)

    if current_len <= max_tokens:
        return messages

    # Need to truncate - find the last user message
    messages = [dict(m) for m in messages]  # Deep copy
    user_msg_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            user_msg_idx = i
            break

    if user_msg_idx is None:
        # No user message to truncate, return as-is
        return messages

    # Binary search to find the right truncation point
    original_content = messages[user_msg_idx]["content"]
    tokens_to_remove = current_len - max_tokens

    # Estimate chars per token (rough estimate, will refine)
    chars_per_token = len(original_content) / max(
        1, tokenizer.encode(original_content).__len__()
    )
    chars_to_remove = int(tokens_to_remove * chars_per_token * 1.2)  # 20% buffer

    # Truncate from the end of the content
    new_content = (
        original_content[:-chars_to_remove]
        if chars_to_remove < len(original_content)
        else ""
    )

    # Iteratively adjust if still too long
    truncation_marker = "\n[HISTORY TRUNCATED]"
    for _ in range(10):  # Max iterations
        messages[user_msg_idx]["content"] = new_content + truncation_marker
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(token_ids) <= max_tokens:
            break
        # Remove 10% more
        remove_chars = max(100, int(len(new_content) * 0.1))
        new_content = (
            new_content[:-remove_chars] if remove_chars < len(new_content) else ""
        )

    return messages


def find_existing_output(
    prompts_file: Path, output_dir: str, model_id: str
) -> Path | None:
    """
    Check if a prompts file has already been processed.

    Checks if the expected output path exists.

    Args:
        prompts_file: Path to the prompts JSONL file
        output_dir: Base output directory
        model_id: Sanitized model ID

    Returns:
        Path to existing output file if found, None otherwise
    """
    expected_path = get_output_path(prompts_file, output_dir, model_id)
    if expected_path.exists():
        return expected_path
    return None


def save_observations(
    output_path: Path,
    observations: list[dict],
    metadata: dict,
) -> None:
    """
    Save observations to JSONL file with metadata.

    Args:
        output_path: Path to output JSONL file
        observations: List of observation dicts
        metadata: Metadata dict to save alongside
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write observations as JSONL
    with open(output_path, "w") as f:
        for obs in observations:
            f.write(json.dumps(obs) + "\n")

    # Write metadata file
    metadata_path = output_path.with_suffix(".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(observations)} observations to {output_path}")
    print(f"Saved metadata to {metadata_path}")


def generate_observations_for_file(
    prompts_file: Path,
    vllm_client,
    model: str,
    batch_size: int,
    frame_stride: int,
    output_dir: str,
    max_prompt_tokens: int,
    verbose: bool,
) -> Path:
    """
    Generate text observations for a single prompts file using vLLM.

    Args:
        prompts_file: Path to prompts JSONL file
        vllm_client: Pre-initialized VLLMWrapper instance
        model: HuggingFace model path (for metadata)
        batch_size: vLLM batch size
        frame_stride: Sample every Nth prompt
        output_dir: Output directory
        max_prompt_tokens: Max tokens for prompt (truncate if exceeded)
        verbose: Verbose output

    Returns:
        Path to output JSONL file
    """
    import time

    # Load prompts
    print(f"\n=== Loading Prompts: {prompts_file.name} ===")
    all_prompts, prompts_metadata = load_prompts(prompts_file)
    print(f"Loaded {len(all_prompts)} prompts")

    subject = prompts_metadata.get("subject", "unknown")
    game_name = prompts_metadata.get("game_name", "unknown")

    # Apply frame stride
    if frame_stride > 1:
        # Group by play/trial, apply stride within each
        prompts_by_play = defaultdict(list)
        for idx, p in enumerate(all_prompts):
            p["_original_idx"] = idx
            if "trial_idx" in p:
                key = (p["game_name"], p["trial_idx"])
            else:
                key = (
                    p.get("level_key", f"{p['game_name']}_level_{p['level_id']}"),
                    p.get("play_within_level_idx", 0),
                )
            prompts_by_play[key].append(p)

        strided_prompts = []
        for key in sorted(prompts_by_play.keys()):
            play_prompts = prompts_by_play[key]
            strided_prompts.extend(play_prompts[::frame_stride])

        print(f"After stride={frame_stride}: {len(strided_prompts)} prompts")
        all_prompts = strided_prompts
    else:
        # Add original index for tracking
        for idx, p in enumerate(all_prompts):
            p["_original_idx"] = idx

    # Generate observations
    print("\n=== Generating Observations ===")
    observations = []
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    batch_times = []
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    for batch_idx, batch_start in enumerate(range(0, len(all_prompts), batch_size)):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch = all_prompts[batch_start:batch_end]
        current_batch_size = len(batch)

        # Truncate messages to fit within max_prompt_tokens
        messages_batch = []
        truncated_count = 0
        for p in batch:
            truncated_msgs = truncate_messages_to_fit(
                p["messages"],
                vllm_client.tokenizer,
                max_prompt_tokens,
            )
            if truncated_msgs != p["messages"]:
                truncated_count += 1
            messages_batch.append(truncated_msgs)

        if truncated_count > 0 and verbose:
            print(
                f"  Truncated {truncated_count}/{len(batch)} prompts to fit {max_prompt_tokens} tokens"
            )

        batch_start_time = time.time()
        batch_results = vllm_client.generate_batch(
            messages_batch, return_prompt_text=True
        )
        batch_elapsed = time.time() - batch_start_time
        batch_times.append(batch_elapsed)

        batch_input_tokens = 0
        batch_output_tokens = 0

        for i, (response, ctx) in enumerate(batch_results):
            prompt = batch[i]
            obs = {
                "prompt_idx": prompt["_original_idx"],
                "trial_idx": prompt.get("trial_idx"),
                "step_num": prompt.get("step_num"),
                "run": prompt.get("run"),
                "play_idx": prompt.get("play_idx"),
                "level_id": prompt.get("level_id"),
                "prompt_text": ctx.get("prompt_text", ""),
                "prompt_token_len": ctx.get("prompt_token_len", 0),
                "response": response,
                "input_tokens": ctx.get("input_tokens", 0),
                "output_tokens": ctx.get("output_tokens", 0),
            }
            observations.append(obs)
            batch_input_tokens += ctx.get("input_tokens", 0)
            batch_output_tokens += ctx.get("output_tokens", 0)

        total_input_tokens += batch_input_tokens
        total_output_tokens += batch_output_tokens

        # Progress stats
        avg_batch_time = sum(batch_times) / len(batch_times)
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = avg_batch_time * remaining_batches
        eta_str = (
            f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            if eta_seconds >= 60
            else f"{eta_seconds:.1f}s"
        )

        avg_input = batch_input_tokens / current_batch_size
        avg_output = batch_output_tokens / current_batch_size
        time_per_sample = batch_elapsed / current_batch_size

        print(
            f"[{batch_idx + 1}/{total_batches}] {batch_end}/{len(all_prompts)} samples | "
            f"in: {avg_input:.0f} out: {avg_output:.0f} tok | "
            f"batch: {batch_elapsed:.1f}s ({time_per_sample:.2f}s/sample) | "
            f"ETA: {eta_str}"
        )

    total_elapsed = time.time() - start_time
    print(f"\n--- Generation Summary ({prompts_file.name}) ---")
    print(
        f"Total: {len(observations)} samples in {total_elapsed:.1f}s ({total_elapsed / len(observations):.2f}s/sample)"
    )
    print(f"Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
    print(
        f"Avg: {total_input_tokens / len(observations):.0f} input, {total_output_tokens / len(observations):.0f} output per sample"
    )

    # Save observations using consistent path structure
    model_id = sanitize_model_id(model)
    output_path = get_output_path(prompts_file, output_dir, model_id)

    # Extract unique trials from prompts for metadata
    trials_dict = {}
    for prompt in all_prompts:
        trial_idx = prompt.get("trial_idx")
        if trial_idx is not None and trial_idx not in trials_dict:
            trials_dict[trial_idx] = {
                "trial_idx": trial_idx,
                "run": prompt.get("run"),
                "play_idx": prompt.get("play_idx"),
                "level_id": prompt.get("level_id"),
                "win": prompt.get("win"),
                "score": prompt.get("score"),
                "num_states": prompt.get("num_states"),
            }
    trials_list = [trials_dict[i] for i in sorted(trials_dict.keys())]

    output_metadata = {
        "prompts_file": str(prompts_file),
        "model": model,
        "model_id": model_id,
        "batch_size": batch_size,
        "frame_stride": frame_stride,
        "total_prompts": len(observations),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "generation_time_seconds": total_elapsed,
        "created_at": datetime.now().isoformat(),
        # Include original prompts metadata
        "subject": subject,
        "game_name": game_name,
        "prompt_variant": prompts_metadata.get("prompt_variant"),
        "sampling_strategy": prompts_metadata.get("sampling_strategy"),
        # Include trials list for viewer
        "trials": trials_list,
    }

    save_observations(output_path, observations, output_metadata)

    print(f"Output: {output_path}")

    return output_path


def generate_observations(
    prompts_files: list[Path],
    model: str,
    batch_size: int,
    max_model_len: int,
    n_gpus: int | None,
    frame_stride: int,
    output_dir: str,
    wandb_project: str | None,
    verbose: bool,
    overwrite: bool = False,
) -> list[Path]:
    """
    Generate text observations using vLLM for multiple prompts files.

    Uses a 3-phase approach to optimize batch processing:
    1. Load all prompts from all files (with stride and truncation)
    2. Process in full batches (only last batch may be partial)
    3. Distribute results back to per-file outputs

    Args:
        prompts_files: List of paths to prompts JSONL files
        model: HuggingFace model path
        batch_size: vLLM batch size
        max_model_len: Max context length
        n_gpus: Number of GPUs (None = auto)
        frame_stride: Sample every Nth prompt
        output_dir: Output directory
        wandb_project: Optional W&B project
        verbose: Verbose output
        overwrite: If True, reprocess files even if output exists

    Returns:
        List of paths to output JSONL files
    """
    import time

    # Filter out already-processed files (unless overwrite is set)
    model_id = sanitize_model_id(model)
    if not overwrite:
        files_to_process = []
        for prompts_file in prompts_files:
            existing = find_existing_output(prompts_file, output_dir, model_id)
            if existing:
                print(
                    f"Skipping {prompts_file.name} (already processed: {existing.name})"
                )
            else:
                files_to_process.append(prompts_file)

        skipped = len(prompts_files) - len(files_to_process)
        if skipped > 0:
            print(f"\nSkipped {skipped} already-processed file(s)")

        if not files_to_process:
            print("All files already processed. Use --overwrite to reprocess.")
            return []

        prompts_files = files_to_process

    # Initialize W&B if requested
    if wandb_project:
        import wandb

        wandb.init(
            project=wandb_project,
            config={
                "prompts_files": [str(p) for p in prompts_files],
                "num_prompts_files": len(prompts_files),
                "model": model,
                "batch_size": batch_size,
                "frame_stride": frame_stride,
            },
        )

    # Load vLLM once
    print("\n=== Loading vLLM ===")
    from src.llm_eval.shared.llm_wrapper import VLLMWrapper

    vllm_client = VLLMWrapper(
        model_path=model,
        max_model_len=max_model_len,
        n_gpus=n_gpus,
        max_num_seqs=batch_size,
    )

    # Calculate max prompt tokens (75% of max_model_len to leave room for generation)
    max_prompt_tokens = int(max_model_len * 0.75)
    print(f"Max prompt tokens: {max_prompt_tokens:,} (75% of {max_model_len:,})")

    # =========================================================================
    # PHASE 1: Load all prompts from all files
    # =========================================================================
    print(f"\n=== Phase 1: Loading Prompts from {len(prompts_files)} files ===")
    phase1_start = time.time()

    # Structure: list of (file_idx, prompt_dict, truncated_messages)
    all_prompts = []
    # Structure: list of {prompts_file, metadata, output_path, prompts_list}
    file_metadata = []

    for file_idx, prompts_file in enumerate(prompts_files):
        print(f"  [{file_idx + 1}/{len(prompts_files)}] Loading {prompts_file.name}...")

        # Load prompts
        prompts_list, prompts_meta = load_prompts(prompts_file)
        print(f"    Loaded {len(prompts_list)} prompts")

        # Apply frame stride (same logic as generate_observations_for_file)
        if frame_stride > 1:
            prompts_by_play = defaultdict(list)
            for idx, p in enumerate(prompts_list):
                p["_original_idx"] = idx
                if "trial_idx" in p:
                    key = (p["game_name"], p["trial_idx"])
                else:
                    key = (
                        p.get("level_key", f"{p['game_name']}_level_{p['level_id']}"),
                        p.get("play_within_level_idx", 0),
                    )
                prompts_by_play[key].append(p)

            strided_prompts = []
            for key in sorted(prompts_by_play.keys()):
                play_prompts = prompts_by_play[key]
                strided_prompts.extend(play_prompts[::frame_stride])

            print(f"    After stride={frame_stride}: {len(strided_prompts)} prompts")
            prompts_list = strided_prompts
        else:
            for idx, p in enumerate(prompts_list):
                p["_original_idx"] = idx

        # Truncate messages and add to all_prompts
        truncated_count = 0
        for prompt in prompts_list:
            truncated_msgs = truncate_messages_to_fit(
                prompt["messages"],
                vllm_client.tokenizer,
                max_prompt_tokens,
            )
            if truncated_msgs != prompt["messages"]:
                truncated_count += 1
            all_prompts.append((file_idx, prompt, truncated_msgs))

        if truncated_count > 0:
            print(
                f"    Truncated {truncated_count} prompts to fit {max_prompt_tokens} tokens"
            )

        # Store file metadata
        output_path = get_output_path(prompts_file, output_dir, model_id)
        file_metadata.append(
            {
                "prompts_file": prompts_file,
                "metadata": prompts_meta,
                "output_path": output_path,
                "subject": prompts_meta.get("subject", "unknown"),
                "game_name": prompts_meta.get("game_name", "unknown"),
            }
        )

    phase1_elapsed = time.time() - phase1_start
    print(
        f"\nPhase 1 complete: {len(all_prompts)} total prompts from {len(prompts_files)} files"
    )
    print(f"Phase 1 time: {format_duration(phase1_elapsed)}")

    # =========================================================================
    # PHASE 2: Process all prompts in full batches
    # =========================================================================
    print("\n=== Phase 2: Generating Observations ===")
    phase2_start = time.time()

    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    print(f"Total batches: {total_batches} (batch_size={batch_size})")

    # Results storage: list of (file_idx, observation_dict)
    all_results = []
    batch_times = []
    total_input_tokens = 0
    total_output_tokens = 0

    for batch_idx, batch_start in enumerate(range(0, len(all_prompts), batch_size)):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch = all_prompts[batch_start:batch_end]
        current_batch_size = len(batch)

        # Extract truncated messages for vLLM
        messages_batch = [
            item[2] for item in batch
        ]  # item = (file_idx, prompt, truncated_msgs)

        batch_start_time = time.time()
        batch_results = vllm_client.generate_batch(
            messages_batch, return_prompt_text=True
        )
        batch_elapsed = time.time() - batch_start_time
        batch_times.append(batch_elapsed)

        batch_input_tokens = 0
        batch_output_tokens = 0

        for i, (response, ctx) in enumerate(batch_results):
            file_idx, prompt, _ = batch[i]
            obs = {
                "prompt_idx": prompt["_original_idx"],
                "trial_idx": prompt.get("trial_idx"),
                "step_num": prompt.get("step_num"),
                "run": prompt.get("run"),
                "play_idx": prompt.get("play_idx"),
                "level_id": prompt.get("level_id"),
                "prompt_text": ctx.get("prompt_text", ""),
                "prompt_token_len": ctx.get("prompt_token_len", 0),
                "response": response,
                "input_tokens": ctx.get("input_tokens", 0),
                "output_tokens": ctx.get("output_tokens", 0),
            }
            all_results.append((file_idx, obs))
            batch_input_tokens += ctx.get("input_tokens", 0)
            batch_output_tokens += ctx.get("output_tokens", 0)

        total_input_tokens += batch_input_tokens
        total_output_tokens += batch_output_tokens

        # Progress stats
        avg_batch_time = sum(batch_times) / len(batch_times)
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = avg_batch_time * remaining_batches
        eta_str = (
            f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            if eta_seconds >= 60
            else f"{eta_seconds:.1f}s"
        )

        avg_input = batch_input_tokens / current_batch_size
        avg_output = batch_output_tokens / current_batch_size
        time_per_sample = batch_elapsed / current_batch_size

        print(
            f"[{batch_idx + 1}/{total_batches}] {batch_end}/{len(all_prompts)} samples | "
            f"in: {avg_input:.0f} out: {avg_output:.0f} tok | "
            f"batch: {batch_elapsed:.1f}s ({time_per_sample:.2f}s/sample) | "
            f"ETA: {eta_str}"
        )

    phase2_elapsed = time.time() - phase2_start
    print("\n--- Phase 2 Summary ---")
    print(f"Total: {len(all_results)} samples in {format_duration(phase2_elapsed)}")
    print(f"Tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
    if all_results:
        print(
            f"Avg: {total_input_tokens / len(all_results):.0f} input, {total_output_tokens / len(all_results):.0f} output per sample"
        )

    # =========================================================================
    # PHASE 3: Distribute results to per-file outputs
    # =========================================================================
    print("\n=== Phase 3: Saving Results ===")
    phase3_start = time.time()

    # Group results by file index
    results_by_file = defaultdict(list)
    for file_idx, obs in all_results:
        results_by_file[file_idx].append(obs)

    # Track per-file token stats for metadata
    file_token_stats = defaultdict(lambda: {"input": 0, "output": 0})
    for file_idx, obs in all_results:
        file_token_stats[file_idx]["input"] += obs.get("input_tokens", 0)
        file_token_stats[file_idx]["output"] += obs.get("output_tokens", 0)

    output_paths = []
    for file_idx, meta in enumerate(file_metadata):
        observations = results_by_file[file_idx]
        output_path = meta["output_path"]

        # Extract unique trials from prompts for metadata
        # We need to re-extract from all_prompts since we have the full prompt dicts there
        trials_dict = {}
        for f_idx, prompt, _ in all_prompts:
            if f_idx != file_idx:
                continue
            trial_idx = prompt.get("trial_idx")
            if trial_idx is not None and trial_idx not in trials_dict:
                trials_dict[trial_idx] = {
                    "trial_idx": trial_idx,
                    "run": prompt.get("run"),
                    "play_idx": prompt.get("play_idx"),
                    "level_id": prompt.get("level_id"),
                    "win": prompt.get("win"),
                    "score": prompt.get("score"),
                    "num_states": prompt.get("num_states"),
                }
        trials_list = [trials_dict[i] for i in sorted(trials_dict.keys())]

        output_metadata = {
            "prompts_file": str(meta["prompts_file"]),
            "model": model,
            "model_id": model_id,
            "batch_size": batch_size,
            "frame_stride": frame_stride,
            "total_prompts": len(observations),
            "total_input_tokens": file_token_stats[file_idx]["input"],
            "total_output_tokens": file_token_stats[file_idx]["output"],
            "generation_time_seconds": phase2_elapsed
            * len(observations)
            / max(1, len(all_results)),  # Proportional time
            "created_at": datetime.now().isoformat(),
            # Include original prompts metadata
            "subject": meta["subject"],
            "game_name": meta["game_name"],
            "prompt_variant": meta["metadata"].get("prompt_variant"),
            "sampling_strategy": meta["metadata"].get("sampling_strategy"),
            # Include trials list for viewer
            "trials": trials_list,
        }

        save_observations(output_path, observations, output_metadata)
        output_paths.append(output_path)

    phase3_elapsed = time.time() - phase3_start
    print(f"Phase 3 time: {format_duration(phase3_elapsed)}")

    if wandb_project:
        import wandb

        wandb.log(
            {
                "num_files_processed": len(output_paths),
                "total_prompts": len(all_results),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            }
        )
        wandb.finish()

    total_elapsed = phase1_elapsed + phase2_elapsed + phase3_elapsed
    print("\n=== All Done ===")
    print(f"Total time: {format_duration(total_elapsed)}")
    print(
        f"Processed {len(prompts_files)} prompts files ({len(all_results)} total samples)"
    )
    for p in output_paths:
        print(f"  - {p}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate text observations using vLLM from pre-generated prompts"
    )

    # Required arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to prompts JSONL file(s) (from run_replay.py). Supports multiple files and glob patterns.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model path"
    )

    # vLLM configuration
    parser.add_argument(
        "--batch-size", type=int, default=1, help="vLLM batch size (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max context length (default: 8192)",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None, help="Number of GPUs (default: auto)"
    )

    # Sampling
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Sample every Nth prompt (default: 1 = all)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/human_replay_observations",
        help="Output directory (default: out/human_replay_observations)",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name (optional)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip already-processed files)",
    )

    args = parser.parse_args()

    # Expand glob patterns and validate prompts files
    import glob

    prompts_files = []
    for pattern in args.prompts:
        matches = glob.glob(pattern)
        if matches:
            prompts_files.extend(Path(m) for m in matches)
        else:
            # Treat as literal path
            prompts_files.append(Path(pattern))

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts_files = []
    for p in prompts_files:
        if p not in seen:
            seen.add(p)
            unique_prompts_files.append(p)
    prompts_files = unique_prompts_files

    # Validate all files exist
    missing = [p for p in prompts_files if not p.exists()]
    if missing:
        print("Error: Prompts file(s) not found:")
        for p in missing:
            print(f"  - {p}")
        print("Run run_replay.py first to generate prompts.")
        return 1

    if not prompts_files:
        print("Error: No prompts files specified")
        return 1

    # Print configuration
    print(f"\n{'=' * 60}")
    print("Generate Observations (vLLM)")
    print(f"{'=' * 60}")
    print(f"Prompts files: {len(prompts_files)}")
    for p in prompts_files:
        print(f"  - {p}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Model Len: {args.max_model_len}")
    print(f"Frame Stride: {args.frame_stride}")
    print(f"Output Dir: {args.output_dir}")
    print(f"{'=' * 60}")

    # Run generation
    generate_observations(
        prompts_files=prompts_files,
        model=args.model,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        n_gpus=args.n_gpus,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
