#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline: TF-IDF + Logistic Regression
- Trains on data/train.parquet, validates on data/val.parquet, tests on data/test.parquet
- Saves model to models/baseline/tfidf_lr.joblib
- Writes metrics to metrics/baseline_val.json and metrics/baseline_test.json
- Writes run metadata to metrics/run.json (seed, params, dataset source, git SHA)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

LABELS = ["neg", "neu", "pos"]  # fixed order

# ---- reproducibility
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_split(outdir: str, name: str) -> pd.DataFrame:
    p = os.path.join(outdir, f"{name}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing Parquet: {p} (run data.py first)")
    return pd.read_parquet(p)


def clf_pipeline(
    max_features: int = 200_000, ngram_max: int = 2, min_df: int = 2, c: float = 4.0
) -> Pipeline:
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
    )
    clf = LogisticRegression(
        solver="liblinear",
        C=c,
        max_iter=200,
        multi_class="ovr",
        random_state=SEED,          # ensure full determinism
        # class_weight="balanced",  # optional: enable if neutral class is very underrepresented
    )
    return Pipeline([("tfidf", vec), ("lr", clf)])


def evaluate(clf: Pipeline, X: pd.Series, y: pd.Series, out_json: str) -> Dict[str, Any]:
    pred = clf.predict(X)
    labels_order = LABELS
    rep = classification_report(
        y,
        pred,
        labels=labels_order,
        target_names=labels_order,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, pred, labels=labels_order).tolist()
    res = {"report": rep, "confusion_matrix": cm}
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    return res


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--modeldir", default="models/baseline")
    ap.add_argument("--metricsdir", default="metrics")
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--C", type=float, default=4.0)
    args = ap.parse_args()

    _ensure_dir(args.modeldir)
    _ensure_dir(args.metricsdir)

    train = load_split(args.datadir, "train")
    val = load_split(args.datadir, "val")
    test = load_split(args.datadir, "test")

    pipe = clf_pipeline(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        c=args.C,
    )

    print(f"Training TF-IDF+LR on {len(train):,} docs …")
    t0 = time.time()
    pipe.fit(train["review_body"], train["label"])
    train_secs = time.time() - t0
    print(f"Training done in {train_secs:.1f}s")

    # VAL
    val_json = os.path.join(args.metricsdir, "baseline_val.json")
    val_res = evaluate(pipe, val["review_body"], val["label"], val_json)
    print("=== VAL ===")
    print(json.dumps(val_res["report"], indent=2))

    # TEST
    test_json = os.path.join(args.metricsdir, "baseline_test.json")
    test_res = evaluate(pipe, test["review_body"], test["label"], test_json)
    print("=== TEST ===")
    print(json.dumps(test_res["report"], indent=2))

    # Save model
    model_path = os.path.join(args.modeldir, "tfidf_lr.joblib")
    joblib.dump(pipe, model_path)
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else None
    print(f"Saved model to {model_path} ({model_size or 0} bytes)")

    # Run metadata
    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "seed": SEED,
        "params": {
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "min_df": args.min_df,
            "C": args.C,
            "solver": "liblinear",
        },
        "data": {
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "labels": LABELS,
            "source": os.getenv("DATA_SRC", "") or f"yelp:{os.getenv('YELP_ROWS', '100000')}",
        },
        "training": {
            "seconds": round(train_secs, 3),
        },
        "artifacts": {
            "model": model_path,
            "model_size_bytes": model_size,
            "metrics_val": val_json,
            "metrics_test": test_json,
        },
    }
    with open(os.path.join(args.metricsdir, "run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print("Saved run metadata to metrics/run.json")


if __name__ == "__main__":
    main()