#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune DistilBERT on train, evaluate on val, save checkpoint.
- Reads Parquet from data/
- Saves model/tokenizer to models/distilbert/
- Prints validation metrics (accuracy, macro F1)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL2ID: Dict[str, int] = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run data.py first.")
    return pd.read_parquet(path)


def to_hf(ds_df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(ds_df[["text", "label"]], preserve_index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--modeldir", default="models/distilbert")
    ap.add_argument("--pretrained", default="distilbert-base-uncased")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "mps"),
        help="Use 'mps' on Apple Silicon if available",
    )
    args = ap.parse_args()

    _ensure_dir(args.modeldir)

    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    train_df = load_df(os.path.join(args.datadir, "train.parquet"))
    val_df = load_df(os.path.join(args.datadir, "val.parquet"))

    train_ds = to_hf(train_df)
    val_ds = to_hf(val_df)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast=True)

    def tok_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=args.max_length
        )

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    training_args = TrainingArguments(
        output_dir=os.path.join(args.modeldir, "runs"),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    model.to(device)

    print("Training…")
    trainer.train()

    print("Evaluating (VAL)…")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    print(f"Saving to {args.modeldir}")
    trainer.save_model(args.modeldir)
    tokenizer.save_pretrained(args.modeldir)


if __name__ == "__main__":
    main()