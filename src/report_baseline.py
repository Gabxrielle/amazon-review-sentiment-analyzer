#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate baseline visuals:
- Confusion matrix (raw & normalized)
- Classification report (JSON & CSV)
- Misclassified examples (CSV) with probabilities
- Top n-grams per class (TXT)
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

def load_artifacts(datadir: str = "data", modeldir: str = "models/baseline"):
    test = pd.read_parquet(Path(datadir) / "test.parquet")
    pipe = joblib.load(Path(modeldir) / "tfidf_lr.joblib")
    return test, pipe

def plot_cm(
    cm: np.ndarray,
    labels: list[str],
    out_png: Path,
    title: str = "Confusion Matrix",
    fmt: str = "raw",
) -> None:
    """
    fmt: "raw" -> integers; "norm" -> floats with 2 decimals
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if fmt == "norm" else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def save_top_features(pipe, out_dir: Path, k: int = 20) -> None:
    vec = pipe.named_steps["tfidf"]
    lr = pipe.named_steps["lr"]
    feats = np.array(vec.get_feature_names_out())
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, cls in enumerate(LABELS):
        # highest positive weights for class i
        idx = lr.coef_[i].argsort()[-k:]
        top = feats[idx]
        (out_dir / f"top_features_{cls}.txt").write_text("\n".join(top), encoding="utf-8")

def main():
    reports_dir = Path("reports")
    ensure_dir(reports_dir)

    test, pipe = load_artifacts()
    y_true = test["label"].astype(str).values
    X = test["review_body"].astype(str).values
    y_pred = pipe.predict(X).astype(str)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_cm(cm, LABELS, reports_dir / "confusion_matrix_test.png", "Confusion Matrix (Test)", fmt="raw")

    # normalized by true row (avoid division by zero)
    denom = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm.astype(float) / denom
    plot_cm(cm_norm, LABELS, reports_dir / "confusion_matrix_test_normalized.png", "Confusion Matrix (Normalized)", fmt="norm")

    # Classification report (JSON & CSV)
    rep_dict = classification_report(
        y_true, y_pred, labels=LABELS, target_names=LABELS, digits=4, output_dict=True, zero_division=0
    )
    (reports_dir / "classification_report_test.json").write_text(
        json.dumps(rep_dict, indent=2), encoding="utf-8"
    )

    rows = []
    for k, v in rep_dict.items():
        if isinstance(v, dict) and {"precision", "recall", "f1-score", "support"} <= set(v.keys()):
            rows.append({"label": k, **v})
    pd.DataFrame(rows).to_csv(reports_dir / "classification_report_test.csv", index=False)

    # Misclassified examples (with probabilities)
    mis_mask = y_true != y_pred
    mis = test.loc[mis_mask, ["review_body"]].copy()
    mis["true"] = y_true[mis_mask]
    mis["pred"] = y_pred[mis_mask]

    # Add per-class probabilities if available (LogReg has predict_proba)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X[mis_mask])  # shape: (n_mis, n_classes aligned to LABELS internally)
        # Map class order used by the classifier to our LABELS
        if hasattr(pipe.classes_, "__iter__"):
            # Build a matrix with columns in the order of LABELS
            class_to_idx = {c: i for i, c in enumerate(pipe.classes_)}
            cols = []
            for c in LABELS:
                idx = class_to_idx.get(c, None)
                cols.append(proba[:, idx] if idx is not None else np.zeros((proba.shape[0],)))
            proba_ordered = np.vstack(cols).T  # (n_mis, len(LABELS))
            for j, c in enumerate(LABELS):
                mis[f"proba_{c}"] = proba_ordered[:, j]

    mis.head(200).to_csv(reports_dir / "misclassified_test.csv", index=False)

    # Top features
    save_top_features(pipe, reports_dir / "top_features", k=20)

    print("✅ Saved visuals & reports in reports/")

if __name__ == "__main__":
    main()