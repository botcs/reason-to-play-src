# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""
File-based prompt loader.

Loads prompt text files from the prompts/ directory tree, replacing the
hardcoded string constants that were previously in prompt_builder.py.
"""

import os

from src.utils import get_repo_path


class PromptLoader:
    """Loads prompt text files by name from the prompts/ directory."""

    def __init__(self, prompts_dir: str | None = None):
        """Initialize PromptLoader.

        Args:
            prompts_dir: Root prompts directory. Defaults to <repo_root>/prompts/.
        """
        if prompts_dir is not None:
            self.prompts_dir = os.path.abspath(prompts_dir)
        else:
            repo_root = get_repo_path()
            self.prompts_dir = os.path.join(repo_root, "prompts")

        if not os.path.isdir(self.prompts_dir):
            raise FileNotFoundError(
                f"Prompts directory does not exist: {self.prompts_dir}"
            )

    def load(self, name: str, subdir: str | None = None) -> str:
        """Load a prompt by name.

        Search order:
        1. prompts/{subdir}/{name}.txt (if subdir given)
        2. prompts/gameplay/{name}.txt
        3. prompts/replay/{name}.txt
        4. prompts/{name}.txt
        5. Treat name as absolute/relative path

        Args:
            name: Prompt name (without .txt extension), or a file path.

        Returns:
            The prompt text content.

        Raises:
            FileNotFoundError: If no matching prompt file is found, with a
                message listing all searched paths.
        """
        searched: list[str] = []

        # 1. prompts/{subdir}/{name}.txt (if subdir given, e.g. 'replay')
        if subdir is not None:
            subdir_path = os.path.join(self.prompts_dir, subdir, f"{name}.txt")
            searched.append(subdir_path)
            if os.path.isfile(subdir_path):
                return self._read(subdir_path)

        # 2. prompts/gameplay/{name}.txt
        gameplay_path = os.path.join(self.prompts_dir, "gameplay", f"{name}.txt")
        searched.append(gameplay_path)
        if os.path.isfile(gameplay_path):
            return self._read(gameplay_path)

        # 2. prompts/replay/{name}.txt
        replay_path = os.path.join(self.prompts_dir, "replay", f"{name}.txt")
        searched.append(replay_path)
        if os.path.isfile(replay_path):
            return self._read(replay_path)

        # 3. prompts/{name}.txt
        root_path = os.path.join(self.prompts_dir, f"{name}.txt")
        searched.append(root_path)
        if os.path.isfile(root_path):
            return self._read(root_path)

        # 3. Treat name as absolute or relative path
        direct_path = os.path.abspath(name)
        searched.append(direct_path)
        if os.path.isfile(direct_path):
            return self._read(direct_path)

        raise FileNotFoundError(
            f"Prompt {name!r} not found. Searched paths:\n"
            + "\n".join(f"  - {p}" for p in searched)
        )

    def _read(self, path: str) -> str:
        """Read a text file and return its contents.

        Args:
            path: Absolute path to the file.

        Returns:
            File contents as a string.
        """
        with open(path, "r") as f:
            return f.read()
