def evaluate(clf: Pipeline, X: pd.Series, y: pd.Series, out_json: str) -> dict:
    pred = clf.predict(X)
    labels_order = LABELS  # ["neg", "neu", "pos"]

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
