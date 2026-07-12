#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import PROJECT_ROOT

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=Path(PROJECT_ROOT),
        check=check,
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit and push completed GraphAtlas outputs.")
    parser.add_argument("--message", default=None)
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()

    if run("rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise RuntimeError("GraphAtlas is not initialized as a Git repository")
    if run("remote", "get-url", args.remote, check=False).returncode != 0:
        raise RuntimeError(f"Git remote {args.remote!r} is not configured")

    run("add", "-A")
    changed = run("diff", "--cached", "--quiet", check=False).returncode != 0
    if changed:
        message = args.message or f"Update GraphAtlas outputs {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}"
        run("commit", "-m", message)

    branch = run("branch", "--show-current").stdout.strip()
    if not branch:
        raise RuntimeError("Cannot push from a detached HEAD")
    run("push", "--set-upstream", args.remote, branch)
    print(f"GitHub synchronized: {args.remote}/{branch}")


if __name__ == "__main__":
    main()
