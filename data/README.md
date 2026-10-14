# Data Architecture & Preprocessing Guide

This directory documents the data acquisition, ingestion, and preprocessing pipeline for FollowThrough.

> [!NOTE]
> **Gitignore Notice:** All raw archives (`data/raw/`) and preprocessed dataset outputs (`data/processed/`) are strictly gitignored to preserve a clean repository and adhere to version control best practices.

---

## 1. Corpus Overview & Scale

FollowThrough is grounded in a large-scale real-world corporate email corpus derived from the public **Enron Email Dataset** (May 7, 2015 version published by Carnegie Mellon University).

### Processed Corpus Statistics:
- **Total Parsed Emails:** 100,000 raw messages ingested across user mailboxes.
- **Retained & Included Emails:** **97,988 emails** (after removing unparseable, empty, or sub-10-character fragments).
- **Cleaned Sentences:** **1,442,694 sentences**.
- **Token Count:** **29,556,736 tokens**.
- **Distinct Conversation Threads:** 42,317 reconstructed threads.

---

## 2. Obtaining Raw Enron Data

You can obtain and extract the raw Enron archive using either the automated pipeline script or manual commands:

### Automated Script:
```bash
python data_pipeline/download_enron.py
```
This script downloads `enron_mail_20150507.tar.gz` (~423 MB) from `https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz`, verifies integrity, and extracts it into `data/raw/maildir/`.

### Manual Download & Extraction:
```bash
mkdir -p data/raw
curl -O https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz -C data/raw/
```
The extracted directory will contain individual user mail directories at `data/raw/maildir/`.

---

## 3. Regenerating Processed Data

To regenerate processed data from raw inputs, use `scripts/prepare_data.py`:

### Full Pipeline Regeneration (Full 100k Emails):
```bash
python scripts/prepare_data.py --full
```
This executes the complete data pipeline:
1. `data_pipeline/select_emails.py`: Selects up to 100,000 emails prioritizing active mailboxes.
2. `data_pipeline/parse_emails.py`: Parses RFC 822 MIME headers and bodies.
3. `data_pipeline/clean_emails.py`: Strips reply/quote headers, forwards, and signatures.
4. `data_pipeline/build_threads.py`: Reconstructs conversation threads from normalized subjects.
5. `data_pipeline/generate_weak_labels.py`: Applies deterministic regex rules for request and commitment pseudo-labels.
6. `data_pipeline/build_corpus_manifest.py`: Partitions records into train (~80%), validation (~10%), and test (~10%) splits with zero thread/duplicate leakage using Union-Find clustering, writing `data/processed/full_corpus_manifest/manifest.jsonl`.

### Smoke Test (Non-Destructive Tiny Sample):
```bash
python scripts/prepare_data.py --smoke
```
Runs a bounded 50-email smoke test outputting to `data/processed/smoke_test/` without overwriting full processed outputs.

---

## 4. Pipeline Stages Summary

1. **RFC 822 Header Parsing:**
   - Extracts `Message-ID`, `Date`, `From`, `To`, `Subject`, and `Thread-Topic` metadata.
2. **Quotation & Forward Boundary Stripping:**
   - Detects boundary patterns (`-----Original Message-----`, `From: [Name]`, etc.). Isolates new contributions.
3. **Noise & Signature Trimming:**
   - Strips disclaimer boilerplate, system markers, and corporate sign-offs.
4. **Sentence Segmentation & Quality Filtering:**
   - Excludes empty bodies and short fragments (<10 characters).
5. **Connected-Thread Graph Partitioning:**
   - Clusters duplicate bodies and conversation threads via Union-Find graph partitioning to ensure **zero split leakage**:
     - **Train Split (~80%):** 78,385 messages
     - **Validation Split (~10%):** 9,946 messages
     - **Test Split (~10%):** 9,657 messages
