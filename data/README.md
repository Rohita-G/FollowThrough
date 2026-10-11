# Data sources for FollowThrough

This project uses three public data sources for the task-tracking system. Raw downloads are stored in `data/raw/` and processed outputs belong in `data/processed/`.

## 1) Enron email corpus
- Source: https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
- License: CMU / research use (see the Enron corpus page and accompanying documentation)
- Download date: to be filled in by the local run
- Size: approximately 443 MB compressed (official page), exact size varies by mirror
- Description: Large corporate email corpus used for task mining, thread recovery, and realistic task-state transitions.

## 2) Parakweet Enron intent dataset
- Source: https://github.com/ParakweetLabs/EmailIntentDataSet/archive/refs/heads/master.zip
- Source raw file: https://raw.githubusercontent.com/ParakweetLabs/EmailIntentDataSet/master/src/resources/Ask0729-fixed.txt
- License: Apache 2.0
- Download date: to be filled in by the local run
- Size: varies by mirror; the raw sentence file is small compared with the full repository
- Description: Sentence-level speech-act / intent examples for request and related email acts to support action-sentence detection.

## 3) AMI Meeting Corpus annotations
- Source: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
- License: CC BY 4.0
- Download date: to be filled in by the local run
- Size: approximately 22 MB
- Description: Meeting transcripts and manual annotations for dialogue acts, decisions, and action summaries; useful for multi-party conversational tracking.

Credit: The AMI annotations are used under the CC BY 4.0 license. Please preserve attribution in the project report and documentation.

## Manual-download instructions
If a link fails in an automated run, download the file manually into the matching path under `data/raw/` and rerun the preprocessing scripts. The expected file names are:
- `data/raw/enron_mail_20150507.tar.gz`
- `data/raw/EmailIntentDataSet-master.zip`
- `data/raw/parakweet/Ask0729-fixed.txt`
- `data/raw/ami_public_manual_1.6.2.zip`
