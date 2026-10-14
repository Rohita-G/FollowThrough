#!/usr/bin/env python3
"""Unified data preparation command for FollowThrough.

Supports:
  python scripts/prepare_data.py --full
  python scripts/prepare_data.py --smoke

--full runs the complete data-preparation workflow on up to 100,000 emails.
--smoke runs a tiny bounded sample (e.g. 50 emails) into data/processed/smoke_test
        and strictly avoids overwriting full processed outputs.
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.download_enron import download_enron
from data_pipeline.select_emails import select_emails
from data_pipeline.parse_emails import parse_email
from data_pipeline.clean_emails import clean_body
from data_pipeline.build_threads import build_threads
from data_pipeline.build_corpus_manifest import build_manifest


def run_data_preparation(is_smoke: bool = False, smoke_limit: int = 50, full_limit: int = 100_000):
    start_time = time.perf_counter()
    maildir_path = REPO_ROOT / "data" / "raw" / "maildir"

    # Step 0: Ensure raw maildir exists
    if not maildir_path.exists():
        print("Raw maildir not found. Checking tarball / downloading...")
        download_enron(dest_dir=REPO_ROOT / "data" / "raw", extract=True)

    limit = smoke_limit if is_smoke else full_limit
    output_dir = REPO_ROOT / "data" / "processed" / ("smoke_test" if is_smoke else "full_corpus_manifest")

    mode_label = "SMOKE-TEST SAMPLE" if is_smoke else "FULL CORPUS"
    print("=" * 70)
    print(f"FOLLOWTHROUGH DATA PREPARATION PIPELINE [{mode_label}]")
    print("=" * 70)

    # 1. Selection
    print("\n[Step 1/5] Selecting emails from raw maildir...")
    selected_files = select_emails(maildir_path=maildir_path, max_emails=limit)
    print(f"  - Raw files found: {len(selected_files)} (Selection limit: {limit:,})")
    print(f"  - Emails selected: {len(selected_files):,}")

    if not selected_files:
        print("Error: No email files found in maildir. Aborting.")
        return

    # 2. Parsing & Cleaning
    print("\n[Step 2/5] Parsing MIME headers and cleaning email bodies...")
    cleaned_records = []
    parsed_count = 0
    cleaned_count = 0

    for file_path in selected_files:
        try:
            parsed = parse_email(file_path, maildir_base=maildir_path)
            parsed_count += 1
            cleaned = clean_body(parsed.get("body", ""))
            cleaned_count += 1

            record = {**parsed, **cleaned}
            cleaned_records.append(record)
        except Exception as e:
            pass

    print(f"  - Emails parsed:  {parsed_count:,}")
    print(f"  - Emails cleaned: {cleaned_count:,}")

    # 3. Thread Reconstruction
    print("\n[Step 3/5] Reconstructing conversation threads...")
    threaded_records = build_threads(cleaned_records)
    distinct_threads = len(set(r["thread_id"] for r in threaded_records))
    print(f"  - Threads built: {distinct_threads:,}")

    # 4. Manifest & Disjoint Splits & Weak Labels
    print("\n[Step 4/5] Building corpus manifest, disjoint splits, and weak labels...")
    manifest_entries, summary = build_manifest(threaded_records, output_dir=output_dir)
    print(f"  - Manifest records created: {len(manifest_entries):,}")

    # 5. Summary & Class Distributions
    print("\n[Step 5/5] Final Pipeline Statistics:")
    print("-" * 70)
    sc = summary.get("split_counts", {})
    print(f"  - Train records:      {sc.get('train', 0):,}")
    print(f"  - Validation records: {sc.get('validation', 0):,}")
    print(f"  - Test records:       {sc.get('test', 0):,}")
    print(f"  - Excluded records:   {sc.get('excluded', 0):,}")

    lc = summary.get("label_counts", {})
    print("\nWeak-Label Class Distribution:")
    for split_name in ("train", "validation", "test"):
        counts = lc.get(split_name, {})
        print(f"  - {split_name.capitalize():<11}: Total={counts.get('total', 0):<6} | Request={counts.get('req', 0):<6} | Commitment={counts.get('com', 0):<6}")

    print("\nOutput Paths:")
    print(f"  - Manifest file: {summary.get('manifest_path')}")
    print(f"  - Summary file:  {output_dir / 'manifest_summary.json'}")

    elapsed = time.perf_counter() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Prepare FollowThrough Dataset and Manifest")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Run full data preparation workflow (up to 100k emails)")
    group.add_argument("--smoke", action="store_true", help="Run smoke test sample (50 emails, non-destructive)")
    parser.add_argument("--sample-size", type=int, default=50, help="Smoke test sample size (default: 50)")
    args = parser.parse_args()

    if args.smoke:
        run_data_preparation(is_smoke=True, smoke_limit=args.sample_size)
    elif args.full:
        run_data_preparation(is_smoke=False, full_limit=100_000)


if __name__ == "__main__":
    main()
