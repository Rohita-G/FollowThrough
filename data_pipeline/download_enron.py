#!/usr/bin/env python3
"""Download and extract the Enron Email dataset for FollowThrough."""

import argparse
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
EXPECTED_SIZE = 443254787


def download_enron(dest_dir: Path = Path("data/raw"), extract: bool = False) -> Path:
    """Download the Enron maildir tarball and optionally extract it."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = dest_dir / "enron_mail_20150507.tar.gz"

    if tarball_path.exists() and tarball_path.stat().st_size == EXPECTED_SIZE:
        print(f"[OK] Enron tarball already exists at {tarball_path} ({tarball_path.stat().st_size} bytes).")
    elif tarball_path.exists() and tarball_path.stat().st_size > 0:
        print(f"[INFO] Enron tarball exists at {tarball_path} ({tarball_path.stat().st_size} bytes).")
    else:
        print(f"Downloading Enron dataset from {ENRON_URL}...")
        req = urllib.request.Request(ENRON_URL, headers={"User-Agent": "FollowThroughDownloader/1.0"})
        with urllib.request.urlopen(req) as resp, open(tarball_path, "wb") as out:
            total = int(resp.headers.get("Content-Length", EXPECTED_SIZE))
            downloaded = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total) * 100
                print(f"\rDownloading: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)", end="")
        print(f"\n[OK] Download complete: {tarball_path}")

    maildir_path = dest_dir / "maildir"
    if extract and not maildir_path.exists():
        print(f"Extracting {tarball_path} into {dest_dir}...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print(f"[OK] Extracted to {maildir_path}")
    elif maildir_path.exists():
        print(f"[OK] maildir directory already exists at {maildir_path}.")

    return tarball_path


def main():
    parser = argparse.ArgumentParser(description="Download Enron Email Dataset")
    parser.add_argument("--dest", type=str, default="data/raw", help="Target raw directory")
    parser.add_argument("--extract", action="store_true", help="Extract maildir after download")
    args = parser.parse_args()

    download_enron(dest_dir=Path(args.dest), extract=args.extract)


if __name__ == "__main__":
    main()
