# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
Utility functions for LLM feature saving.

The main feature saving logic is now in chunked_feature_saver.py (LevelFeatureSaver).
This module only contains the sanitize_model_id utility.
"""


def sanitize_model_id(model_path: str) -> str:
    """
    Convert model path to a valid directory name.

    Args:
        model_path: HuggingFace model path (e.g., 'meta-llama/Llama-3.2-1B-Instruct')

    Returns:
        Sanitized model ID (e.g., 'meta-llama_Llama-3.2-1B-Instruct')
    """
    # Replace path separators and special chars
    model_id = model_path.replace("/", "_").replace("\\", "_")
    # Remove any characters that might cause filesystem issues
    model_id = "".join(c for c in model_id if c.isalnum() or c in "-_.")
    return model_id
