# tests/test_smoke.py
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest


def _make_small_splits(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "review_body": ["bad", "ok", "great", "terrible", "nice", "meh"],
            "label": ["neg", "neu", "pos", "neg", "pos", "neu"],
        }
    )
    # 4 train / 1 val / 1 test
    df.iloc[:4].to_parquet(outdir / "train.parquet", index=False)
    df.iloc[4:5].to_parquet(outdir / "val.parquet", index=False)
    df.iloc[5:6].to_parquet(outdir / "test.parquet", index=False)


@pytest.mark.parametrize("py", ["python"])
def test_train_baseline_smoke(tmp_path: Path, py: str):
    datadir = tmp_path / "data"
    modeldir = tmp_path / "models" / "baseline"
    metricsdir = tmp_path / "metrics"

    _make_small_splits(datadir)

    env = os.environ.copy()
    # Ensure no external source interferes in the smoke test
    env.pop("DATA_SRC", None)
    env.pop("YELP_ROWS", None)
    env["SEED"] = "42"

    # Train
    subprocess.run(
        [
            py,
            "src/train_sklearn.py",
            "--datadir",
            str(datadir),
            "--modeldir",
            str(modeldir),
            "--metricsdir",
            str(metricsdir),
        ],
        check=True,
        env=env,
    )

    # Artifacts existence
    model_path = modeldir / "tfidf_lr.joblib"
    val_json = metricsdir / "baseline_val.json"
    test_json = metricsdir / "baseline_test.json"
    run_json = metricsdir / "run.json"

    assert model_path.exists(), "model not saved"
    assert val_json.exists() and test_json.exists(), "metrics JSON not saved"
    assert run_json.exists(), "run metadata not saved"

    # Model size > 0
    assert model_path.stat().st_size > 0

    # Metrics schema quick check
    m = json.loads(test_json.read_text(encoding="utf-8"))
    assert "report" in m and "confusion_matrix" in m
    # At least the labels keys should exist in the classification report
    for lbl in ["neg", "neu", "pos"]:
        assert lbl in m["report"]

    # Optional: run report script to ensure it works with the trained model
    subprocess.run(
        [py, "src/report_baseline.py"],
        check=True,
        cwd=Path.cwd(),  # report reads default paths, fine to run in repo root
        env=env,
    )
