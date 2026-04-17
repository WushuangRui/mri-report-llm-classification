"""
Script name:
    evaluate_single_model.py

Purpose:
    Evaluate a single model's classification results against human labels.
    Computes metrics, confusion matrix, and misclassified cases.

Input:
    1. Prediction CSV with columns:
        - id
        - result_cls
    2. Human label CSV with columns:
        - id
        - label

Output:
    1. evaluation_summary.csv
    2. confusion_matrix.png
    3. misclassified_cases.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================
# Paths (EDIT HERE)
# =========================
PRED_CSV = "path/to/predictions.csv"
LABEL_CSV = "path/to/labels.csv"
OUTPUT_DIR = "path/to/output"

# =========================
# Constants
# =========================
LABELS = [1, 2, 3]
LABEL_NAMES = {
    1: "normal",
    2: "non-specific WM lesion",
    3: "abnormal",
}


# =========================
# Utilities
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_columns(df):
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df


def load_label_data(path):
    df = pd.read_csv(path)
    df = clean_columns(df)

    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("Label CSV must contain 'id' and 'label' columns.")

    df["id"] = df["id"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin(LABELS)].copy()
    df["label"] = df["label"].astype(int)

    return df


def load_prediction_data(path):
    df = pd.read_csv(path)
    df = clean_columns(df)

    if "id" not in df.columns or "result_cls" not in df.columns:
        raise ValueError("Prediction CSV must contain 'id' and 'result_cls'.")

    df["id"] = df["id"].astype(str).str.strip()
    df["result_cls"] = pd.to_numeric(df["result_cls"], errors="coerce")
    df = df[df["result_cls"].isin(LABELS)].copy()
    df["result_cls"] = df["result_cls"].astype(int)

    return df


def evaluate(df):
    y_true = df["label"]
    y_pred = df["result_cls"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "support": len(y_true),
    }

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return metrics, report, cm


def save_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS)
    plt.yticks(tick_marks, LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics_csv(metrics, report, path):
    rows = []

    rows.append(["Metric", "Value"])
    for k, v in metrics.items():
        rows.append([k, v])

    rows.append([])
    rows.append(["Class", "Precision", "Recall", "F1"])

    for cls in LABELS:
        r = report[str(cls)]
        rows.append([
            cls,
            r["precision"],
            r["recall"],
            r["f1-score"]
        ])

    pd.DataFrame(rows).to_csv(path, index=False, header=False)


def save_misclassified(df, path):
    mis_df = df[df["label"] != df["result_cls"]]
    mis_df.to_csv(path, index=False)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading data...")
    label_df = load_label_data(LABEL_CSV)
    pred_df = load_prediction_data(PRED_CSV)

    print("Merging...")
    merged = pd.merge(label_df, pred_df, on="id", how="inner")

    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check IDs.")

    print(f"Total samples: {len(merged)}")

    print("Evaluating...")
    metrics, report, cm = evaluate(merged)

    print("Saving outputs...")

    save_metrics_csv(
        metrics,
        report,
        os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    )

    save_confusion_matrix(
        cm,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    save_misclassified(
        merged,
        os.path.join(OUTPUT_DIR, "misclassified_cases.csv")
    )

    print("Done.")
    print(metrics)


if __name__ == "__main__":
    main()
