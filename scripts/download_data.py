#!/usr/bin/env python3
"""Download and verify the project datasets required for FollowThrough.

The script is intentionally small and dependency-light so it runs on a normal laptop.
It supports resume downloads and validates downloaded sizes against the server's
Content-Length (or a conservative expected size when provided).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SOURCES = [
    {
        "name": "enron",
        "url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
        "dest": RAW_DIR / "enron_mail_20150507.tar.gz",
        "expected_size": 443254787,
    },
    {
        "name": "parakweet_repo",
        "url": "https://github.com/ParakweetLabs/EmailIntentDataSet/archive/refs/heads/master.zip",
        "dest": RAW_DIR / "EmailIntentDataSet-master.zip",
        "expected_size": None,
    },
    {
        "name": "parakweet_raw",
        "url": "https://raw.githubusercontent.com/ParakweetLabs/EmailIntentDataSet/master/src/resources/Ask0729-fixed.txt",
        "dest": RAW_DIR / "parakweet" / "Ask0729-fixed.txt",
        "expected_size": None,
    },
    {
        "name": "ami",
        "url": "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",
        "dest": RAW_DIR / "ami_public_manual_1.6.2.zip",
        "expected_size": 22000000,
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(total: int, completed: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(completed / total * 100):.1f}%"


def head_content_length(url: str) -> Optional[int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "FollowThroughDownloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def download_file(url: str, dest: Path, expected_size: Optional[int] = None, chunk_size: int = 65536) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    current_size = dest.stat().st_size if dest.exists() else 0
    remote_size = head_content_length(url)

    if dest.exists() and current_size > 0:
        if expected_size is not None and current_size == expected_size:
            print(f"[skip] {dest.name} already exists and matches the expected size ({expected_size} bytes).")
            return
        if remote_size is not None and current_size == remote_size:
            print(f"[skip] {dest.name} already exists and matches the remote size ({remote_size} bytes).")
            return

    offset = current_size
    req = urllib.request.Request(url, headers={"User-Agent": "FollowThroughDownloader/1.0"})
    if offset > 0:
        req.add_header("Range", f"bytes={offset}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            remote_total = remote_size
            if remote_total is None:
                remote_total = int(r.headers.get("Content-Length", "0")) or None
            mode = "ab" if offset > 0 else "wb"
            with dest.open(mode) as out:
                total_written = offset
                started = time.time()
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    total_written += len(chunk)
                    if total_written - offset > 0:
                        elapsed = max(1.0, time.time() - started)
                        rate = (total_written - offset) / elapsed
                        if remote_total:
                            print(f"[download] {dest.name}: {_pct(remote_total, total_written)} ({total_written}/{remote_total} bytes, {rate/1024:.1f} KiB/s)")
                        else:
                            print(f"[download] {dest.name}: {total_written} bytes downloaded ({rate/1024:.1f} KiB/s)")
        if remote_total is not None and abs(dest.stat().st_size - remote_total) > 1024 * 1024:
            raise RuntimeError(f"Downloaded {dest} has {dest.stat().st_size} bytes but the server reported {remote_total} bytes.")
        if expected_size is not None and dest.stat().st_size != expected_size and (remote_total is None or dest.stat().st_size != remote_total):
            raise RuntimeError(f"Downloaded {dest} has {dest.stat().st_size} bytes but expected {expected_size} bytes and remote file size {remote_total} bytes.")
        print(f"[ok] {dest}")
    except Exception:
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify FollowThrough datasets.")
    parser.add_argument("--list", action="store_true", help="List all configured sources and exit.")
    args = parser.parse_args()

    if args.list:
        for item in SOURCES:
            print(f"{item['name']}: {item['url']}")
        return 0

    for item in SOURCES:
        url = item["url"]
        dest = item["dest"]
        print(f"\n=== {item['name']} ===")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[exists] {dest} ({dest.stat().st_size} bytes)")
        download_file(url=url, dest=dest, expected_size=item["expected_size"])

    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command-line error reporting
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
