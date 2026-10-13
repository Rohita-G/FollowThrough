# Data Architecture & Preprocessing Guide

This directory documents the data acquisition, ingestion, and preprocessing pipeline for FollowThrough. Raw email archives and processed token files are excluded from version control via `.gitignore`.

---

## 1. Corpus Overview & Scale

FollowThrough is grounded in a large-scale real-world corporate email corpus derived from the public **Enron Email Dataset** (May 7, 2015 version published by Carnegie Mellon University).

### Processed Corpus Statistics:
- **Total Parsed Emails:** Approximately 100,000 raw messages ingested across user mailboxes.
- **Retained & Included Emails:** **97,988 emails** (after removing unparseable, empty, or sub-10-character fragments).
- **Cleaned Sentences:** **1.44 million sentences** (1,442,694).
- **Token Count:** **29.56 million tokens** (29,560,000).
- **Distinct Conversation Threads:** 42,317 reconstructed threads.

This real-world volume substantially exceeds the standard capstone minimum requirement of 50,000 sentences.

---

## 2. Obtaining the Enron Corpus

To obtain the raw Enron archive:
1. Download the gzipped tarball from the Carnegie Mellon University archive:
   ```bash
   curl -O https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz
   ```
2. Extract the archive into the local data directory:
   ```bash
   tar -xzf enron_mail_20150507.tar.gz -C data/raw/
   ```
   The extracted tree contains individual user mail directories (`maildir/`).

---

## 3. Preprocessing & Normalization Pipeline

Raw email records undergo a multi-stage deterministic cleaning pipeline:

1. **RFC 822 Header Extraction:**
   - Extracts `Message-ID`, `Date`, `From`, `To`, `Subject`, and `Thread-Topic` metadata.
   - Cleans mailbox paths to establish email sender provenance.

2. **Quotation and Forward Boundary Stripping:**
   - Detects boundary patterns such as `-----Original Message-----`, `From: [Name]`, and `---------------------- Forwarded by`.
   - Isolates the author's new contribution to prevent attributing tasks from previous turns to the current message sender.

3. **Noise and Signature Removal:**
   - Strips disclaimer boilerplate, system-generated tracking markers, and automated footer URLs.
   - Normalizes unicode whitespace and line-break artifacts.

4. **Sentence Segmentation & Exclusion Criteria:**
   - Segments cleaned text into sentences using standard punctuation heuristics.
   - **Exclusions:**
     - Empty or purely whitespace bodies.
     - Ultra-short fragments (<10 characters) containing no communicative context.
     - Pure pleasantries without substantive directives.

5. **Conservative Deduplication & Connected-Thread Graph Partitioning:**
   - Many corporate emails are duplicates across multiple inboxes (e.g., sent messages copied to multiple recipients).
   - FollowThrough computes a composite key: `SHA256(cleaned_body) || sender || normalized_subject`.
   - A DisjointSet (Union-Find) graph connects all duplicate groups and conversation thread IDs.
   - Splitting occurs on entire connected components:
     - **Train Split (~80%):** 78,385 messages
     - **Validation Split (~10%):** 9,946 messages
     - **Test Split (~10%):** 9,657 messages
   - This guarantees **zero cluster leakage** between training, validation, and testing.

---

## 4. Manifest Generation

To generate the canonical experiment manifest with transparent weak supervision:
```bash
python3 scripts/build_full_corpus_manifest.py
```
This produces `data/processed/full_corpus_manifest/manifest.jsonl`, where each record includes the message text, thread cluster ID, deterministic split partition, and transparent weak labels.
