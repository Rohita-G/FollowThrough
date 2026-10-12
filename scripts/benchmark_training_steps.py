#!/usr/bin/env python3
"""Benchmark 100 steps of DistilBERT training on full corpus to project runtime."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MANIFEST_PATH = Path("data/processed/full_corpus_manifest/manifest.jsonl")


class ManifestStreamDataset(IterableDataset):
    def __init__(self, manifest_path: Path, split: str = "train"):
        self.manifest_path = manifest_path
        self.split = split

    def __iter__(self):
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split") == self.split and not record.get("excluded"):
                    text = record.get("text", "")
                    labels = [float(record.get("weak_request", 0)), float(record.get("weak_commitment", 0))]
                    yield text, labels


def collate_batch(batch, tokenizer, max_len=128):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    enc["labels"] = labels
    return enc


def run_benchmark():
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2,
        problem_type="multi_label_classification",
    )
    model.to(DEVICE)
    model.train()

    dataset = ManifestStreamDataset(MANIFEST_PATH, split="train")
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_batch(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Warming up (5 steps)...")
    iterator = iter(loader)
    for _ in range(5):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        optimizer.step()

    print("Benchmarking 100 steps on MPS...")
    start_time = time.perf_counter()
    for step in range(1, 101):
        batch = next(iterator)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(out.logits, labels)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start_time
    sec_per_step = elapsed / 100.0
    print(f"100 steps completed in {elapsed:.2f}s ({sec_per_step:.4f}s / step).")

    total_train_examples = 78385
    steps_per_epoch = total_train_examples // 2
    proj_epoch_hours = (steps_per_epoch * sec_per_step) / 3600.0
    print(f"Projected 1 epoch ({steps_per_epoch} steps): {proj_epoch_hours:.2f} hours")


if __name__ == "__main__":
    run_benchmark()
