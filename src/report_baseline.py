#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate visual artifacts for the baseline:
- Confusion matrix (PNG)
- Classification report (JSON & CSV)
- Misclassified examples (CSV)
Reads:  models/baseline/tfidf_lr.joblib, data/test.parquet
Writes: reports/confusion_matrix_test.png, reports/classification_report_test.csv,
        reports/misclassified_test.csv
"""

from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

LABELS = ["neg", "neu", "pos"]

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_artifacts(datadir="data", modeldir="models/baseline"):
    test = pd.read_parquet(Path(datadir) / "test.parquet")
    pipe = joblib.load(Path(modeldir) / "tfidf_lr.joblib")
    return test, pipe

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_png: Path, title="Confusion Matrix (Test)"):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    reports_dir = Path("reports")
    ensure_dir(reports_dir)

    test, pipe = load_artifacts()
    y_true = test["label"].astype(str).values
    X = test["review_body"].astype(str).values
    y_pred = pipe.predict(X).astype(str)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_confusion_matrix(cm, LABELS, reports_dir / "confusion_matrix_test.png")

    # classification report (JSON & CSV)
    rep_dict = classification_report(
        y_true, y_pred, labels=LABELS, target_names=LABELS, digits=4, output_dict=True, zero_division=0
    )
    with open(reports_dir / "classification_report_test.json", "w") as f:
        json.dump(rep_dict, f, indent=2)
    # flat CSV
    rows = []
    for k, v in rep_dict.items():
        if isinstance(v, dict) and {"precision","recall","f1-score","support"} <= set(v.keys()):
            rows.append({"label": k, **v})
    pd.DataFrame(rows).to_csv(reports_dir / "classification_report_test.csv", index=False)

    # misclassified examples
    mis = test.loc[y_true != y_pred, ["review_body"]].copy()
    mis["true"] = y_true[y_true != y_pred]
    mis["pred"] = y_pred[y_true != y_pred]
    mis.head(200).to_csv(reports_dir / "misclassified_test.csv", index=False)

    print("✅ Saved:")
    print("  - reports/confusion_matrix_test.png")
    print("  - reports/classification_report_test.json")
    print("  - reports/classification_report_test.csv")
    print("  - reports/misclassified_test.csv")

if __name__ == "__main__":
    main()
