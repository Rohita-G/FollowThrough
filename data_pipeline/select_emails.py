#!/usr/bin/env python3
"""Deterministic email selection from raw Enron maildir with mailbox prioritization."""

import argparse
import os
from pathlib import Path
from typing import List, Optional

PRIORITY_MAILBOXES = ("kaminski-v", "dasovich-j", "shackleton-s", "germany-c", "beck-s")


def get_ordered_mailboxes(maildir_path: Path) -> List[str]:
    """Return deterministic list of mailbox directories prioritizing key actors."""
    if not maildir_path.exists():
        return []
    with os.scandir(maildir_path) as entries:
        all_dirs = sorted([
            entry.name for entry in entries
            if not entry.name.startswith(".") and entry.is_dir()
        ])
    present = set(all_dirs)
    ordered = [m for m in PRIORITY_MAILBOXES if m in present]
    ordered.extend([m for m in all_dirs if m not in PRIORITY_MAILBOXES])
    return ordered


def select_emails(
    maildir_path: Path = Path("data/raw/maildir"),
    max_emails: Optional[int] = 100_000,
) -> List[Path]:
    """Select email files deterministically across prioritized mailboxes."""
    maildir_path = Path(maildir_path)
    if not maildir_path.exists():
        raise FileNotFoundError(f"Maildir directory not found at {maildir_path}")

    mailboxes = get_ordered_mailboxes(maildir_path)
    selected_files: List[Path] = []

    for mb in mailboxes:
        mb_dir = maildir_path / mb
        # Traverse subfolders deterministically
        for root, dirs, files in os.walk(mb_dir):
            dirs.sort()
            for fname in sorted(files):
                if fname.startswith("."):
                    continue
                file_path = Path(root) / fname
                if file_path.is_file():
                    selected_files.append(file_path)
                    if max_emails is not None and len(selected_files) >= max_emails:
                        return selected_files

    return selected_files


def main():
    parser = argparse.ArgumentParser(description="Select Enron emails deterministically")
    parser.add_argument("--maildir", type=str, default="data/raw/maildir", help="Path to maildir directory")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum emails to select")
    args = parser.parse_args()

    files = select_emails(Path(args.maildir), max_emails=args.limit)
    print(f"Selected {len(files)} email files from {args.maildir}")


if __name__ == "__main__":
    main()
