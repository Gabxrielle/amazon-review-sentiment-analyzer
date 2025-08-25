# src/data.py
import os, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")
DATA_SRC = os.getenv("DATA_SRC", "").strip()  # e.g. data/raw/amazon_reviews_us_Beauty_v1_00.tsv.gz
YELP_ROWS = int(os.getenv("YELP_ROWS", "100000"))

def map_label_from_stars(star):
    try:
        s = int(float(star))
    except Exception:
        return None
    if s in (1, 2): return "neg"
    if s == 3:      return "neu"
    if s in (4, 5): return "pos"
    return None

def pick_cols(df):
    cols = [c.lower().strip() for c in df.columns]
    if "review_body" in cols and "star_rating" in cols:
        return df.columns[cols.index("review_body")], df.columns[cols.index("star_rating")]
    if "reviewtext" in cols and "overall" in cols:
        return df.columns[cols.index("reviewtext")], df.columns[cols.index("overall")]
    text_col = None
    rating_col = None
    for c in df.columns:
        lc = c.lower()
        if text_col is None and ("review" in lc and ("text" in lc or "body" in lc or "content" in lc)):
            text_col = c
        if rating_col is None and (("star" in lc and "rating" in lc) or "overall" in lc or lc == "rating" or "stars" in lc):
            rating_col = c
    return text_col, rating_col

def from_amazon_local(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"DATA_SRC points to a non-existent path: {path}")
    with open(path, "rb") as f:
        head = f.read(128)
    if head.startswith(b"<?xml") or b"<Error>" in head:
        raise ValueError(
            f"File {path} seems to be an XML error (S3 AccessDenied). "
            "Download the correct Amazon TSV/TSV.GZ file."
        )

    print(f"📥 Loading local Amazon file: {path}")
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip", low_memory=False, compression="infer")
    text_col, rating_col = pick_cols(df)
    if not text_col or not rating_col:
        raise KeyError(f"Could not find text/rating columns. Columns: {list(df.columns)}")
    df = df[[text_col, rating_col]].dropna()
    df = df.rename(columns={text_col: "review_body", rating_col: "star_rating"})
    df["label"] = df["star_rating"].apply(map_label_from_stars)
    return df.dropna(subset=["label", "review_body"]).reset_index(drop=True)

def from_yelp(rows: int) -> pd.DataFrame:
    print(f"📥 Loading Yelp (yelp_review_full) with {rows} rows …")
    from datasets import load_dataset
    ds = load_dataset("yelp_review_full", split=f"train[:{rows}]")
    df = pd.DataFrame(ds)
    df = df.rename(columns={"text": "review_body", "label": "yelp_label"})
    df["star_rating"] = (df["yelp_label"].astype(int) + 1).astype(int)
    df["label"] = df["star_rating"].apply(map_label_from_stars)
    return df[["review_body", "star_rating", "label"]].dropna().reset_index(drop=True)

def save_splits(df: pd.DataFrame):
    print("📊 Class distribution:")
    print(df["label"].value_counts())
    print("✂️ Stratified split (80/10/10)…")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df,  test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUT_DIR / "val.parquet",   index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
    print("✅ Saved: data/train.parquet, data/val.parquet, data/test.parquet")

def main():
    try:
        if DATA_SRC:
            df = from_amazon_local(DATA_SRC)
        else:
            print("ℹ️ DATA_SRC not defined — using Yelp fallback.")
            df = from_yelp(YELP_ROWS)
    except Exception as e:
        print(f"⚠️ Amazon local load failed ({e}). Falling back to Yelp.")
        df = from_yelp(YELP_ROWS)

    save_splits(df)

if __name__ == "__main__":
    main()