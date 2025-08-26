#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple CLI to run inference with the baseline model.

Examples:
  python -m src.infer --text "great product!"
  python -m src.infer --file path/to/reviews.txt
  python -m src.infer --file -               # read from STDIN
  python -m src.infer --file reviews.txt --proba
  python -m src.infer --file reviews.txt --proba --output preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

LABELS = ["neg", "neu", "pos"]


def load_model(path: str = "models/baseline/tfidf_lr.joblib"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"[infer] model not found: {path}. Train first: python src/train_sklearn.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[infer] failed to load model {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _read_lines_from_file(path: str) -> List[str]:
    if path == "-":  # stdin
        data = sys.stdin.read()
        return [ln.strip() for ln in data.splitlines() if ln.strip()]
    p = Path(path)
    if not p.exists():
        print(f"[infer] file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def predict_labels(model, texts: Iterable[str]) -> np.ndarray:
    return model.predict(pd.Series(list(texts)))


def predict_probas_ordered(model, texts: Iterable[str]) -> np.ndarray:
    """Return probabilities aligned to LABELS order."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("model has no predict_proba")
    probs = model.predict_proba(pd.Series(list(texts)))  # shape (n, k)
    # Map model.classes_ -> our LABELS order
    if not hasattr(model, "classes_"):
        # sklearn Pipeline exposes classes_ directly
        raise AttributeError("model has no classes_")
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    cols = []
    for c in LABELS:
        idx = class_to_idx.get(c, None)
        cols.append(probs[:, idx] if idx is not None else np.zeros((probs.shape[0],)))
    return np.vstack(cols).T  # (n, len(LABELS))


def main():
    ap = argparse.ArgumentParser(description="Run inference with baseline model")
    ap.add_argument("--model", default="models/baseline/tfidf_lr.joblib", help="Path to joblib model")
    ap.add_argument("--text", default=None, help="Single text to classify")
    ap.add_argument("--file", default=None, help="Path to a text file (one review per line). Use '-' for STDIN.")
    ap.add_argument("--proba", action="store_true", help="Print per-class probabilities")
    ap.add_argument("--output", default=None, help="Optional CSV output path for batch mode")
    args = ap.parse_args()

    if not args.text and not args.file:
        print("Provide --text or --file (use --file - to read from STDIN)", file=sys.stderr)
        sys.exit(2)

    model = load_model(args.model)

    # single text
    if args.text:
        texts = [args.text]
        preds = predict_labels(model, texts)
        if args.proba:
            try:
                prob = predict_probas_ordered(model, texts)[0]
                # print: label<TAB>p_neg<TAB>p_neu<TAB>p_pos<TAB>text
                print(f"{preds[0]}\t{prob[0]:.4f}\t{prob[1]:.4f}\t{prob[2]:.4f}\t{args.text}")
            except Exception as e:
                print(preds[0])
        else:
            print(preds[0])
        return

    # batch from file/stdin
    texts = _read_lines_from_file(args.file)
    if not texts:
        print("[infer] no non-empty lines found.", file=sys.stderr)
        sys.exit(3)

    preds = predict_labels(model, texts)

    if args.proba:
        try:
            probas = predict_probas_ordered(model, texts)  # (n, 3)
        except Exception:
            probas = None
    else:
        probas = None

    # output
    if args.output:
        df = pd.DataFrame({"text": texts, "pred": preds})
        if probas is not None:
            df["proba_neg"] = probas[:, 0]
            df["proba_neu"] = probas[:, 1]
            df["proba_pos"] = probas[:, 2]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[infer] wrote {len(df)} rows to {args.output}")
        return

    # print to stdout
    if probas is None:
        for t, p in zip(texts, preds):
            print(f"{p}\t{t}")
    else:
        for t, p, pr in zip(texts, preds, probas):
            print(f"{p}\t{pr[0]:.4f}\t{pr[1]:.4f}\t{pr[2]:.4f}\t{t}")


if __name__ == "__main__":
    main()