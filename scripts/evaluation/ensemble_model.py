"""
Script name:
    evaluate_ensemble_strict_consensus_single.py

Purpose:
    Evaluate a strict-consensus ensemble of three models (ds, qw, and one
    third model) for a single experiment using a two-step classification
    strategy.

    Step 1:
        Convert original labels:
        - original classes 1 and 2 -> step1 class 1
        - original class 3         -> step1 class 2

        Ensemble rule:
        Only if all three models map to step1 class 1, the final step1
        prediction is 1. Otherwise, the final step1 prediction is 2.

    Step 2:
        Applied only to samples predicted as step1 class 1.

        Ensemble rule:
        Only keep samples where all three models exactly agree on the same
        original class (1/2/3). Otherwise, discard the sample from step2
        evaluation.

Output:
    1. A combined confusion matrix figure containing:
       - step1 confusion matrix
       - step1 metrics text
       - step2 confusion matrix
    2. A CSV file containing:
       - step1 metrics
       - step1 confusion matrix
    3. A CSV file containing:
       - step2 misclassified cases
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

# ============================================================
# Paths (edit here)
# ============================================================
THIRD_MODEL_NAME = "gpt5"

LABELLED_FILE = "path/to/labelled_dataset.csv"
DS_FILE = "path/to/ds_predictions.csv"
QW_FILE = "path/to/qw_predictions.csv"
THIRD_FILE = "path/to/third_model_predictions.csv"

# Optional:
# If you need to restrict evaluation to a predefined ID set, provide a CSV path
# containing an 'id' column. Otherwise set to None.
ID_REFERENCE_FILE = None

OUT_DIR = "path/to/output"

# ============================================================
# Constants
# ============================================================
LABELS_3 = [1, 2, 3]
LABELS_2 = [1, 2]


# ============================================================
# Utility functions
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def robust_read_csv(path: str, dtype=None) -> pd.DataFrame:
    """
    Read a CSV file using a list of common encodings.

    Args:
        path: Path to the CSV file.
        dtype: Optional dtype mapping passed to pandas.read_csv().

    Returns:
        A pandas DataFrame.

    Raises:
        UnicodeDecodeError: If all attempted encodings fail.
    """
    encodings_to_try = [
        "utf-8",
        "utf-8-sig",
        "gbk",
        "gb18030",
        "latin1",
        "cp1252",
    ]

    last_error = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, dtype=dtype, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"Failed to read file with attempted encodings {encodings_to_try}: {path}",
    )


def load_id_reference(path: str) -> set:
    """
    Load an optional reference ID file.

    Expected column:
        - id

    Args:
        path: Path to the reference CSV.

    Returns:
        A set of normalized IDs.
    """
    df = robust_read_csv(path, dtype={"id": str})
    df = normalize_columns(df)

    if "id" not in df.columns:
        raise ValueError(f"Reference file must contain an 'id' column: {path}")

    ids = normalize_id_series(df["id"].dropna())
    return set(ids.tolist())


def read_labelled_data(path: str) -> pd.DataFrame:
    """
    Read human-labeled data.

    Expected columns:
        - id
        - label

    Returns:
        DataFrame with valid labels in {1, 2, 3}.
    """
    df = robust_read_csv(path, dtype={"id": str})
    df = normalize_columns(df)

    if not {"id", "label"}.issubset(df.columns):
        raise ValueError(f"Label file must contain 'id' and 'label': {path}")

    df = df[["id", "label"]].copy()
    df = df.dropna(subset=["id"]).copy()
    df["id"] = normalize_id_series(df["id"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["label"]).copy()
    df = df[df["label"].isin(LABELS_3)].copy()

    return df.reset_index(drop=True)


def read_result_csv(path: str, allowed_ids: set | None = None) -> dict:
    """
    Read one model prediction CSV.

    Expected columns:
        - id
        - result_cls
      or
        - id
        - result

    Args:
        path: Path to prediction CSV.
        allowed_ids: Optional set of IDs to keep.

    Returns:
        A dictionary mapping id -> predicted class.
    """
    df = robust_read_csv(path, dtype={"id": str})
    df = normalize_columns(df)

    if "id" not in df.columns:
        raise ValueError(f"Prediction file is missing 'id': {path}")

    if "result_cls" in df.columns:
        result_column = "result_cls"
    elif "result" in df.columns:
        result_column = "result"
    else:
        raise ValueError(f"Prediction file must contain 'result_cls' or 'result': {path}")

    df = df.dropna(subset=["id"]).copy()
    df["id"] = normalize_id_series(df["id"])
    df[result_column] = pd.to_numeric(df[result_column], errors="coerce").astype("Int64")
    df.loc[~df[result_column].isin(LABELS_3), result_column] = pd.NA

    if allowed_ids is not None:
        df = df[df["id"].isin(allowed_ids)].copy()

    return dict(zip(df["id"], df[result_column]))


def map_step1_label(x):
    """
    Map original 3-class labels to step1 binary labels.

    Original:
        1 -> 1
        2 -> 1
        3 -> 2
    """
    if x is None or pd.isna(x):
        return None

    x = int(x)
    if x in (1, 2):
        return 1
    if x == 3:
        return 2
    return None


def fuse_step1_strict(model_results: dict, ids: list[str]) -> dict:
    """
    Strict-consensus fusion for step1.

    Final prediction is 1 only if all three models map to step1 class 1.
    Otherwise the final prediction is 2.
    """
    fused = {}
    for sample_id in ids:
        values = []
        for model_key in ("ds", "qw", "third"):
            raw_value = model_results[model_key].get(sample_id, pd.NA)
            if pd.isna(raw_value):
                values.append(None)
            else:
                values.append(map_step1_label(int(raw_value)))

        fused[sample_id] = 1 if values == [1, 1, 1] else 2

    return fused


def fuse_step2_strict_original_123(model_results: dict, ids_from_step1_pred_1: list[str]) -> dict:
    """
    Strict-consensus fusion for step2.

    Among samples predicted as step1 class 1, only keep a final prediction if
    all three models exactly agree on the same original class (1/2/3).
    """
    fused = {}
    for sample_id in ids_from_step1_pred_1:
        values = []
        missing = False

        for model_key in ("ds", "qw", "third"):
            raw_value = model_results[model_key].get(sample_id, pd.NA)
            if pd.isna(raw_value):
                missing = True
                values.append(None)
            else:
                values.append(int(raw_value))

        if (not missing) and (values[0] == values[1] == values[2]):
            fused[sample_id] = values[0]

    return fused


def get_basic_metrics(y_true, y_pred, average="binary", pos_label=1):
    """
    Compute standard evaluation metrics.

    Returns:
        A dictionary with accuracy, precision, recall, and f1_score.
    """
    accuracy = accuracy_score(y_true, y_pred)

    if average == "binary":
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=pos_label,
            zero_division=0,
        )
    else:
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def evaluate_step1(labelled_eval: pd.DataFrame, pred_map_step1: dict):
    """
    Evaluate step1 binary predictions.
    """
    df = labelled_eval.copy()
    df["step1_label"] = df["label"].map(map_step1_label).astype("Int64")
    df = df[df["id"].isin(pred_map_step1.keys())].copy()

    ids = df["id"].tolist()
    y_true = [int(x) for x in df["step1_label"].tolist()]
    y_pred = [int(pred_map_step1[sample_id]) for sample_id in ids]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_2)
    metrics = get_basic_metrics(y_true, y_pred, average="binary", pos_label=1)

    return {
        "eval_df": df,
        "ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "metrics": metrics,
    }


def evaluate_step2(labelled_eval: pd.DataFrame, pred_map_step1: dict, pred_map_step2: dict):
    """
    Evaluate step2 three-class predictions.
    """
    step1_pred_1_ids = [sample_id for sample_id, pred in pred_map_step1.items() if pred == 1]

    df = labelled_eval[labelled_eval["id"].isin(step1_pred_1_ids)].copy()
    df = df[df["id"].isin(pred_map_step2.keys())].copy()

    ids = df["id"].tolist()
    y_true = [int(x) for x in df["label"].tolist()]
    y_pred = [int(pred_map_step2[sample_id]) for sample_id in ids]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS_3)
    metrics_macro = get_basic_metrics(y_true, y_pred, average="macro")
    metrics_weighted = get_basic_metrics(y_true, y_pred, average="weighted")

    return {
        "eval_df": df,
        "ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "metrics_macro": metrics_macro,
        "metrics_weighted": metrics_weighted,
    }


def cm_to_long_df(cm, labels: list[int], section_name: str) -> pd.DataFrame:
    """
    Convert a confusion matrix to long-format DataFrame.
    """
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "section": section_name,
                "true_label": true_label,
                "pred_label": pred_label,
                "count": int(cm[i][j]),
            })
    return pd.DataFrame(rows)


def step1_metrics_row(metrics: dict) -> pd.DataFrame:
    """
    Build a one-row DataFrame for step1 metrics.
    """
    return pd.DataFrame([{
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }])


def step2_mislabel_df(ids: list[str], y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    """
    Build a mislabel DataFrame for step2.
    """
    rows = []
    for sample_id, true_label, pred_label in zip(ids, y_true, y_pred):
        if true_label != pred_label:
            rows.append({
                "id": sample_id,
                "result_cls": pred_label,
                "label": true_label,
            })
    return pd.DataFrame(rows)


def draw_metrics_text(ax, metrics: dict, title: str):
    """
    Draw metrics text in the same visual style as the original script.
    """
    ax.axis("off")
    lines = [
        title,
        "",
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1-score : {metrics['f1_score']:.4f}",
    ]
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=11,
        family="monospace",
    )


def plot_confusion_matrices(step1_res: dict, step2_res: dict, out_png: str, third_model_name: str):
    """
    Plot the step1 confusion matrix, step1 metrics text, and step2 confusion
    matrix in a layout that matches the original style as closely as possible.
    """
    fig = plt.figure(figsize=(6.5, 10))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[3.0, 1.2, 3.0],
        hspace=0.35,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=step1_res["cm"],
        display_labels=LABELS_2,
    )
    disp1.plot(ax=ax1, cmap="Reds", colorbar=False, values_format="d")
    ax1.set_title("Step1 (2x2)", fontsize=12)

    ax2 = fig.add_subplot(gs[1, 0])
    draw_metrics_text(ax2, step1_res["metrics"], "Step1 metrics")

    ax3 = fig.add_subplot(gs[2, 0])
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=step2_res["cm"],
        display_labels=LABELS_3,
    )
    disp2.plot(ax=ax3, cmap="Blues", colorbar=False, values_format="d")
    ax3.set_title("Step2 (3x3)", fontsize=12)

    fig.suptitle(f"Ensemble evaluation: ds + qw + {third_model_name}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    labelled_raw = read_labelled_data(LABELLED_FILE)

    if ID_REFERENCE_FILE is not None:
        allowed_ids = load_id_reference(ID_REFERENCE_FILE)
        labelled_eval = labelled_raw[labelled_raw["id"].isin(allowed_ids)].copy()
    else:
        allowed_ids = set(labelled_raw["id"].tolist())
        labelled_eval = labelled_raw.copy()

    model_results = {
        "ds": read_result_csv(DS_FILE, allowed_ids=allowed_ids),
        "qw": read_result_csv(QW_FILE, allowed_ids=allowed_ids),
        "third": read_result_csv(THIRD_FILE, allowed_ids=allowed_ids),
    }

    labelled_ids = labelled_eval["id"].tolist()

    # Step 1
    pred_map_step1 = fuse_step1_strict(model_results, labelled_ids)
    step1_res = evaluate_step1(labelled_eval, pred_map_step1)

    # Step 2
    ids_step1_pred_1 = [sample_id for sample_id, pred in pred_map_step1.items() if pred == 1]
    pred_map_step2 = fuse_step2_strict_original_123(model_results, ids_step1_pred_1)
    step2_res = evaluate_step2(labelled_eval, pred_map_step1, pred_map_step2)

    # Output 1: figure
    figure_path = os.path.join(
        OUT_DIR,
        f"ds+qw+{THIRD_MODEL_NAME}_step1_step2_confusion_matrices.png",
    )
    plot_confusion_matrices(step1_res, step2_res, figure_path, THIRD_MODEL_NAME)

    # Output 2: step1 metrics + step1 confusion matrix CSV
    step1_csv_path = os.path.join(
        OUT_DIR,
        f"ds+qw+{THIRD_MODEL_NAME}_step1_confusion_matrix_and_metrics.csv",
    )
    step1_metrics_df = step1_metrics_row(step1_res["metrics"])
    step1_cm_df = cm_to_long_df(step1_res["cm"], LABELS_2, "step1_confusion_matrix")

    with open(step1_csv_path, "w", encoding="utf-8-sig", newline="") as file:
        file.write("### step1_metrics ###\n")
        step1_metrics_df.to_csv(file, index=False)
        file.write("\n")
        file.write("### step1_confusion_matrix ###\n")
        step1_cm_df.to_csv(file, index=False)

    # Output 3: step2 mislabels CSV
    step2_mislabel_csv_path = os.path.join(
        OUT_DIR,
        f"ds+qw+{THIRD_MODEL_NAME}_step2_mislabels.csv",
    )
    step2_mislabel_df_all = step2_mislabel_df(
        step2_res["ids"],
        step2_res["y_true"],
        step2_res["y_pred"],
    )
    step2_mislabel_df_all.to_csv(step2_mislabel_csv_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Figure saved to: {figure_path}")
    print(f"Step1 CSV saved to: {step1_csv_path}")
    print(f"Step2 mislabels saved to: {step2_mislabel_csv_path}")


if __name__ == "__main__":
    main()
