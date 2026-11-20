#!/usr/bin/env python3
"""Generate ALL-FILES.md, a machine-readable inventory of every file in this repo.

Usage:
    python make_all_files_md.py            # raw walk: every file except .git and this inventory
    python make_all_files_md.py --respect-gitignore   # only files git would track/see

The inventory lists the file set on disk. It never follows into .git, and it
never lists itself. Output is deterministic (sorted, POSIX separators).

Why this script exists: ALL-FILES.md was originally added as a one-off commit
(01fc975) with no generator, so it silently went stale when the tree changed and
was later left with unresolved merge-conflict markers. Regenerate with this
instead of hand-editing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
INVENTORY_NAME = "ALL-FILES.md"


def git_tracked_and_untracked() -> set[str]:
    """Files git considers in-tree: tracked + untracked-but-not-ignored."""
    r = subprocess.run(
        ["git", "-C", REPO, "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise SystemExit(f"git ls-files failed: {r.stderr.strip()}")
    return {p.replace("\\", "/") for p in r.stdout.splitlines() if p.strip()}


def disk_walk() -> list[str]:
    """Every file under REPO except .git and the inventory itself."""
    found: list[str] = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, REPO).replace("\\", "/")
            if rel == INVENTORY_NAME:
                continue
            found.append(rel)
    return sorted(found)


def title_for(repo: str) -> str:
    parent = os.path.basename(os.path.dirname(repo.rstrip("/")))
    return f"{parent}/{os.path.basename(repo.rstrip('/'))}"


def render(paths: list[str], title: str) -> str:
    lines = [
        f"# {title} — Full File Inventory",
        "",
        f"**Total entries:** {len(paths)}",
        "",
        "```",
    ]
    lines.extend(f"./{p}" for p in paths)
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--respect-gitignore", action="store_true",
                    help="list only files git tracks or would see (skip __pycache__, *.pyc, egg-info, output/)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the on-disk inventory is stale; do not write")
    args = ap.parse_args()

    if args.respect_gitignore:
        paths = sorted(git_tracked_and_untracked() - {INVENTORY_NAME})
    else:
        paths = disk_walk()

    target = os.path.join(REPO, INVENTORY_NAME)
    content = render(paths, title_for(REPO))

    if args.check:
        if not os.path.exists(target):
            print(f"STALE: {INVENTORY_NAME} missing")
            return 1
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            current = fh.read()
        if current != content:
            cur = {l[2:] for l in current.splitlines() if l.startswith("./")}
            new = set(paths)
            missing = sorted(new - cur)
            extra = sorted(cur - new)
            print(f"STALE: {len(missing)} missing, {len(extra)} extra")
            for p in missing:
                print("  +", p)
            for p in extra:
                print("  -", p)
            return 1
        print(f"OK: {len(paths)} entries match")
        return 0

    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    print(f"wrote {target}: {len(paths)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
