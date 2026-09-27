#!/usr/bin/env python3
"""Fetch baseline sources at audited immutable revisions, without installing them.

Run --list to inspect the source lock; select named sources explicitly.
Existing checkouts must already be clean and pinned. This command never resets,
cleans, updates, or overwrites an existing checkout, nor follows submodules.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

LOCK = Path(__file__).with_name("sources.json")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE
    ).strip()


def verify_checkout(path: Path, revision: str) -> None:
    if git("rev-parse", "HEAD", cwd=path) != revision:
        raise ValueError(f"{path}: HEAD differs from locked revision {revision}")
    if git("status", "--porcelain", "--untracked-files=all", cwd=path):
        raise ValueError(f"{path}: checkout contains local changes; left untouched")


def fetch_source(name: str, source: dict, root: Path, transport: str = "https") -> Path:
    revision = source["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{name}: revision must be a full immutable Git SHA")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(f"Invalid source name: {name!r}")
    destination = root / name
    if destination.exists():
        verify_checkout(destination, revision)
    else:
        root.mkdir(parents=True, exist_ok=True)
        url = source["url"]
        if transport == "ssh" and url.startswith("https://github.com/"):
            url = "git@github.com:" + url.removeprefix("https://github.com/")
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as staging:
            staged = Path(staging) / "checkout"
            git("init", "--quiet", str(staged))
            git("remote", "add", "origin", url, cwd=staged)
            git("fetch", "--depth=1", "origin", revision, cwd=staged)
            git("checkout", "--detach", "--quiet", revision, cwd=staged)
            verify_checkout(staged, revision)
            staged.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--transport", choices=("https", "ssh"), default="https")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).with_name("checkouts")
    )
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if args.list:
        print(json.dumps(lock, indent=2))
        return
    if not args.sources:
        parser.error("select one or more source names, or use --list")
    unknown = set(args.sources) - lock["sources"].keys()
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")
    for name in args.sources:
        path = fetch_source(name, lock["sources"][name], args.root, args.transport)
        print(f"Verified {name}: {path} @ {lock['sources'][name]['revision']}")


if __name__ == "__main__":
    main()
