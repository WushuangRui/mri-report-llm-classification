#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script name:
    analysis_unlabeled_icd10.py

Purpose:
    This script performs two-stage strict-consensus ensemble analysis on
    unlabeled MRI report predictions from three models, and links the
    resulting groups to ICD10 clinical diagnosis data for downstream
    descriptive analysis.

Workflow:
    1. Read prediction CSV files from three models.
    2. Apply a two-stage strict-consensus ensemble strategy:
       - Step 1:
         original class 1/2 -> step1 class 1
         original class 3   -> step1 class 2
         Final step1 prediction is 1 only if all three models map to 1;
         otherwise it is 2.
       - Step 2:
         applied only to step1 class 1 cases;
         a final 3-class prediction is kept only if all three models
         exactly agree on the same original label (1/2/3).
    3. Read and filter ICD10 diagnosis records.
    4. Summarize ICD10 distributions for different predicted groups.
    5. Export prediction tables, stage summary tables, and ICD10 frequency tables.

Input:
    1. Three model prediction CSV files, each containing:
       - mid
       - result_cls   (or result)

    2. One merged clinical diagnosis CSV file containing at least:
       - m_id
       - icd10
       - dx_name
       - dx_type
       - status_comment
       - scan_date
       - entry_date

    3. One ICD10 mapping CSV file containing at least:
       - icd10_category
       - category_name

    4. One raw report CSV file containing at least:
       - mid
       - raw_report
       (or report / report_text / rawreport)

    5. Optional reference CSV file containing:
       - mid
       Used only if you want to restrict analysis to a predefined subset.

Output:
    In OUTPUT_DIR, this script generates:

    1. step1_predictions.csv
       Columns:
       - mid
       - ds_raw
       - qw_raw
       - third_raw
       - step1_pred

    2. step2_predictions.csv
       Columns:
       - mid
       - ds_raw
       - qw_raw
       - third_raw
       - step2_pred
       - step2_has_consensus

    3. stage_counts_summary.csv
       Summary counts for total reports, step1 counts, and step2 counts.

    4. step1_class1_mids_with_icd10_and_raw_report.csv
       Columns:
       - mid
       - raw_report
       - icd10_list
       - agg_icd10_key_list
       - agg_category_name_list

    5. step1_icd10/
       - step1_class1_raw_icd10.csv
       - step1_class1_agg_icd10.csv
       - step1_class2_raw_icd10.csv
       - step1_class2_agg_icd10.csv
       - step1_icd10_rank.csv

    6. step2_icd10/
       - step2_consensus1_raw_icd10.csv
       - step2_consensus1_agg_icd10.csv
       - step2_consensus2_raw_icd10.csv
       - step2_consensus2_agg_icd10.csv
       - step2_consensus12_icd10_rank.csv

Notes:
    - All file paths are defined at the top of the script for easy editing.
    - This GitHub version avoids hard-coded personal paths.
"""

import os
import re
from typing import Dict, Tuple, List, Optional

import pandas as pd


# =========================================================
# Paths to edit
# =========================================================
MODEL_NAME = "gpt5"
MODE = "cns"
MODEL_TAG = f"ds+qw3+{MODEL_NAME}_{MODE}"

DS_FILE = "path/to/ds_predictions.csv"
QW_FILE = "path/to/qwen_predictions.csv"
THIRD_FILE = "path/to/third_model_predictions.csv"

MID_REF_FILE = None  # e.g. "path/to/reference_mids.csv"

MERGED_FILE = "path/to/merged_clinical_diagnosis.csv"
ALL_ICD10_FILE = "path/to/icd10_mapping.csv"
RAW_REPORT_FILE = "path/to/raw_report_file.csv"

OUTPUT_DIR = "path/to/output_directory"

WINDOW_DAYS = 10
LABELS_3 = [1, 2, 3]


# =========================================================
# Exclusion keywords
# =========================================================
EXCLUDE_KEYWORDS = [
    r"\bhistory\b",
    r"\bhx\b",
    r"\bh/o\b",
    r"\bpossible\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\bsuspected\b",
    r"\bsuspect\b",
    r"\brule\s+out\b",
    r"\br/o\b",
    r"\bquestionable\b",
    r"\bpresumptive\b",
    r"\bpresumably\b",
    r"\blikely\b",
    r"疑似",
    r"可疑",
    r"排除",
    r"病史",
]
EXCLUDE_PATTERN = re.compile("|".join(EXCLUDE_KEYWORDS), flags=re.IGNORECASE)


# =========================================================
# Basic utilities
# =========================================================
def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lower().strip().replace("\ufeff", "") for col in df.columns]
    return df


def normalize_mid_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def robust_read_csv(path: str, dtype=None, usecols=None) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1", "cp1252"]
    last_error = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, dtype=dtype, encoding=encoding, usecols=usecols)
        except UnicodeDecodeError as error:
            last_error = error
            continue

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        f"Failed to read CSV file with attempted encodings {encodings_to_try}: {path}",
    )


def load_mid_reference(mid_ref_path: str) -> set:
    df = robust_read_csv(mid_ref_path, dtype={"mid": str})
    df = normalize_columns(df)

    if "mid" not in df.columns:
        raise ValueError(f"Reference file is missing 'mid' column: {mid_ref_path}")

    mids = normalize_mid_series(df["mid"].dropna())
    return set(mids.tolist())


# =========================================================
# Model prediction loading and two-stage fusion
# =========================================================
def read_result_csv(path: str, allowed_mids: Optional[set] = None) -> dict:
    """
    Read one model prediction CSV.

    Supported columns:
        - mid
        - result_cls or result

    Returns:
        {mid: label}
    """
    df = robust_read_csv(path, dtype={"mid": str})
    df = normalize_columns(df)

    if "mid" not in df.columns:
        raise ValueError(f"Prediction file is missing 'mid' column: {path}")

    if "result_cls" in df.columns:
        result_col = "result_cls"
    elif "result" in df.columns:
        result_col = "result"
    else:
        raise ValueError(f"Prediction file must contain 'result_cls' or 'result': {path}")

    df = df.dropna(subset=["mid"]).copy()
    df["mid"] = normalize_mid_series(df["mid"])
    df[result_col] = pd.to_numeric(df[result_col], errors="coerce").astype("Int64")
    df.loc[~df[result_col].isin(LABELS_3), result_col] = pd.NA

    if allowed_mids is not None:
        df = df[df["mid"].isin(allowed_mids)].copy()

    return dict(zip(df["mid"], df[result_col]))


def collect_union_mids(*result_dicts: dict) -> List[str]:
    mids = set()
    for result_dict in result_dicts:
        mids.update(result_dict.keys())
    return sorted(mids)


def map_step1_label(x):
    """
    Map original 3-class labels to step1 binary labels:
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


def fuse_step1_strict(model_results: dict, mids: List[str]) -> dict:
    """
    Step1 strict consensus rule:
    final prediction is 1 only if all three models map to step1 class 1;
    otherwise final prediction is 2.
    """
    fused = {}
    for mid in mids:
        values = []
        for model_key in ("ds", "qw", "third"):
            raw_value = model_results[model_key].get(mid, pd.NA)
            if pd.isna(raw_value):
                values.append(None)
            else:
                values.append(map_step1_label(int(raw_value)))

        fused[mid] = 1 if values == [1, 1, 1] else 2

    return fused


def fuse_step2_strict_original_123(model_results: dict, mids_from_step1_pred_1: List[str]) -> dict:
    """
    Step2 strict consensus rule:
    among step1 class 1 cases, keep a final label only if all three models
    exactly agree on the same original class (1/2/3).
    """
    fused = {}
    for mid in mids_from_step1_pred_1:
        values = []
        missing = False

        for model_key in ("ds", "qw", "third"):
            raw_value = model_results[model_key].get(mid, pd.NA)
            if pd.isna(raw_value):
                missing = True
                values.append(None)
            else:
                values.append(int(raw_value))

        if (not missing) and (values[0] == values[1] == values[2]):
            fused[mid] = values[0]

    return fused


def build_prediction_tables(model_results: dict, all_mids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    step1_rows = []
    for mid in all_mids:
        ds_raw = model_results["ds"].get(mid, pd.NA)
        qw_raw = model_results["qw"].get(mid, pd.NA)
        third_raw = model_results["third"].get(mid, pd.NA)

        step1_pred = fuse_step1_strict(model_results, [mid])[mid]

        step1_rows.append({
            "mid": mid,
            "ds_raw": ds_raw,
            "qw_raw": qw_raw,
            "third_raw": third_raw,
            "step1_pred": step1_pred,
        })

    step1_df = pd.DataFrame(step1_rows).sort_values("mid").reset_index(drop=True)

    mids_step1_1 = step1_df.loc[step1_df["step1_pred"] == 1, "mid"].tolist()
    step2_map = fuse_step2_strict_original_123(model_results, mids_step1_1)

    step2_rows = []
    for mid in mids_step1_1:
        ds_raw = model_results["ds"].get(mid, pd.NA)
        qw_raw = model_results["qw"].get(mid, pd.NA)
        third_raw = model_results["third"].get(mid, pd.NA)
        step2_pred = step2_map.get(mid, pd.NA)

        step2_rows.append({
            "mid": mid,
            "ds_raw": ds_raw,
            "qw_raw": qw_raw,
            "third_raw": third_raw,
            "step2_pred": step2_pred,
            "step2_has_consensus": 0 if pd.isna(step2_pred) else 1,
        })

    step2_df = pd.DataFrame(step2_rows).sort_values("mid").reset_index(drop=True)
    return step1_df, step2_df


def save_stage_counts(step1_df: pd.DataFrame, step2_df: pd.DataFrame, out_dir: str) -> None:
    total_reports = int(step1_df["mid"].nunique())
    step1_count_1 = int((step1_df["step1_pred"] == 1).sum())
    step1_count_2 = int((step1_df["step1_pred"] == 2).sum())

    step2_total_input = int(step2_df["mid"].nunique())
    step2_count_1 = int((step2_df["step2_pred"] == 1).sum()) if not step2_df.empty else 0
    step2_count_2 = int((step2_df["step2_pred"] == 2).sum()) if not step2_df.empty else 0
    step2_count_3 = int((step2_df["step2_pred"] == 3).sum()) if not step2_df.empty else 0
    step2_no_consensus = int(step2_df["step2_pred"].isna().sum()) if not step2_df.empty else 0

    summary_df = pd.DataFrame([{
        "total_reports": total_reports,
        "step1_class1_count": step1_count_1,
        "step1_class2_count": step1_count_2,
        "step2_input_count": step2_total_input,
        "step2_consensus_1_count": step2_count_1,
        "step2_consensus_2_count": step2_count_2,
        "step2_consensus_3_count": step2_count_3,
        "step2_no_consensus_count": step2_no_consensus,
    }])

    summary_path = os.path.join(out_dir, "stage_counts_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    info(f"Saved: {summary_path}")


# =========================================================
# ICD10 logic
# =========================================================
def normalize_alnum(code: str) -> str:
    s = str(code).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def build_exact_matcher(all_df: pd.DataFrame) -> Tuple[Dict[str, Tuple[str, str]], set, pd.DataFrame]:
    """
    Build an exact ICD10 matcher based on normalized alphanumeric codes.
    """
    all_df = all_df.dropna(subset=["icd10_category", "category_name"]).copy()
    all_df["icd10_category"] = all_df["icd10_category"].astype(str).str.strip().str.upper()
    all_df["category_name"] = all_df["category_name"].astype(str).str.strip()

    all_df["norm_key"] = all_df["icd10_category"].map(normalize_alnum)
    all_df = all_df[all_df["norm_key"].astype(str).str.len() > 0].copy()

    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, row in all_df.iterrows():
        norm_key = row["norm_key"]
        if norm_key not in norm_map:
            norm_map[norm_key] = (row["icd10_category"], row["category_name"])

    norm_set = set(norm_map.keys())

    skeleton = (
        all_df[["icd10_category", "category_name"]]
        .drop_duplicates()
        .rename(columns={"icd10_category": "icd10_key"})
        .reset_index(drop=True)
    )

    return norm_map, norm_set, skeleton


def exact_match_icd10(code: str, norm_set: set) -> Optional[str]:
    normalized = normalize_alnum(code)
    if not normalized:
        return None
    return normalized if normalized in norm_set else None


def text_contains_exclude_keyword(text: str) -> bool:
    if not isinstance(text, str) or text.strip() == "":
        return False
    return bool(EXCLUDE_PATTERN.search(text))


def load_and_filter_merged(merged_file: str, window_days: int) -> pd.DataFrame:
    """
    Filter merged clinical diagnosis data by:
    1. dx_type == Encounter Diagnosis
    2. |entry_date - scan_date| <= window_days
    3. Excluding uncertain/history-like diagnosis rows
    """
    merged = robust_read_csv(
        merged_file,
        dtype=str,
        usecols=["m_id", "icd10", "dx_name", "dx_type", "status_comment", "scan_date", "entry_date"],
    ).dropna(subset=["m_id", "icd10"]).copy()

    merged["m_id"] = merged["m_id"].astype(str).str.strip()
    merged["icd10"] = merged["icd10"].astype(str).str.strip().str.upper()
    merged["dx_type"] = merged["dx_type"].astype(str).str.strip()
    merged["dx_name"] = merged["dx_name"].fillna("").astype(str)
    merged["status_comment"] = merged["status_comment"].fillna("").astype(str)

    merged = merged[merged["dx_type"].str.lower() == "encounter diagnosis"].copy()

    merged["scan_date_parsed"] = pd.to_datetime(merged["scan_date"], format="%m/%d/%y", errors="coerce")
    merged["entry_date_parsed"] = pd.to_datetime(merged["entry_date"], errors="coerce")

    valid_mask = merged["scan_date_parsed"].notna() & merged["entry_date_parsed"].notna()
    diff = (merged["entry_date_parsed"] - merged["scan_date_parsed"]).abs()
    merged = merged[valid_mask & (diff <= pd.Timedelta(days=window_days))].copy()

    exclude_mask = (
        merged["dx_name"].apply(text_contains_exclude_keyword) |
        merged["status_comment"].apply(text_contains_exclude_keyword)
    )
    merged = merged[~exclude_mask].copy()

    merged = merged.drop(columns=["scan_date_parsed", "entry_date_parsed"])
    merged = merged.drop_duplicates(subset=["m_id", "icd10"]).reset_index(drop=True)

    return merged


def build_outputs_for_class_keep_full_icd(
    class_mids: set,
    unique_mid_icd: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int, float, int]:
    used_mids = sorted(class_mids & set(unique_mid_icd["m_id"]))
    usable = len(used_mids)
    sub_mid_icd = unique_mid_icd[unique_mid_icd["m_id"].isin(used_mids)].copy()

    if usable > 0 and not sub_mid_icd.empty:
        freq_series = sub_mid_icd.groupby("icd10")["m_id"].nunique()
    else:
        freq_series = pd.Series(dtype=int)

    order = freq_series.sort_values(ascending=False)
    out = pd.DataFrame({"icd10": order.index, "frequency": order.values})
    out["mean_frequency"] = out["frequency"] / usable if usable > 0 else 0.0
    out = out.sort_values(["mean_frequency", "icd10"], ascending=[False, True])

    if usable > 0 and not sub_mid_icd.empty:
        icd_total = int(sub_mid_icd.groupby("m_id")["icd10"].nunique().sum())
        avg_icd_per_mid = icd_total / usable
    else:
        icd_total = 0
        avg_icd_per_mid = 0.0

    uniq_icd_count = len(out)
    return out, usable, icd_total, avg_icd_per_mid, uniq_icd_count


def apply_all_icd10_exact_filter_and_aggregate(
    out_df: pd.DataFrame,
    usable: int,
    norm_map: Dict[str, Tuple[str, str]],
    norm_set: set,
) -> pd.DataFrame:
    if out_df.empty or usable == 0:
        return pd.DataFrame(columns=["icd10_key", "category_name", "frequency", "mean_frequency"])

    rows: List[Dict] = []
    for _, row in out_df.iterrows():
        code = str(row["icd10"]).strip().upper()
        frequency = int(row["frequency"])
        norm_key = exact_match_icd10(code, norm_set)

        if norm_key is None:
            continue

        original_key, category_name = norm_map[norm_key]
        rows.append({
            "icd10_key": original_key,
            "category_name": category_name,
            "frequency": frequency,
        })

    if not rows:
        return pd.DataFrame(columns=["icd10_key", "category_name", "frequency", "mean_frequency"])

    df = pd.DataFrame(rows).groupby(["icd10_key", "category_name"], as_index=False)["frequency"].sum()
    df["mean_frequency"] = df["frequency"] / usable
    df = df.sort_values(["mean_frequency", "icd10_key"], ascending=[False, True]).reset_index(drop=True)
    return df


def build_rank(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    skeleton: pd.DataFrame,
    name_a: str = "cls1",
    name_b: str = "cls2",
) -> pd.DataFrame:
    base = skeleton.copy()

    df_rank = base.merge(
        df_a.rename(columns={"frequency": f"count_{name_a}", "mean_frequency": f"freq_{name_a}"}),
        on=["icd10_key", "category_name"],
        how="left",
    ).merge(
        df_b.rename(columns={"frequency": f"count_{name_b}", "mean_frequency": f"freq_{name_b}"}),
        on=["icd10_key", "category_name"],
        how="left",
    )

    for col in [f"count_{name_a}", f"freq_{name_a}", f"count_{name_b}", f"freq_{name_b}"]:
        df_rank[col] = pd.to_numeric(df_rank[col], errors="coerce").fillna(0)

    df_rank["total_count"] = df_rank[f"count_{name_a}"] + df_rank[f"count_{name_b}"]
    return df_rank.sort_values(["total_count", "icd10_key"], ascending=[False, True]).reset_index(drop=True)


# =========================================================
# Group-level ICD exports
# =========================================================
def analyze_one_group_icd(
    group_name: str,
    mids: set,
    unique_mid_icd: pd.DataFrame,
    norm_map: Dict[str, Tuple[str, str]],
    norm_set: set,
    out_dir: str,
) -> pd.DataFrame:
    raw_out, usable, icd_total, avg_icd_per_mid, uniq_icd = build_outputs_for_class_keep_full_icd(
        mids, unique_mid_icd
    )

    info(f"[{group_name}] total mids: {len(mids)}")
    info(f"[{group_name}] usable mids: {usable}")
    info(f"[{group_name}] total raw ICD10 count: {icd_total}")
    info(f"[{group_name}] avg ICD10 per mid: {avg_icd_per_mid:.3f}")
    info(f"[{group_name}] unique raw ICD10 count: {uniq_icd}")

    aggregated = apply_all_icd10_exact_filter_and_aggregate(raw_out, usable, norm_map, norm_set)

    raw_csv = os.path.join(out_dir, f"{group_name}_raw_icd10.csv")
    agg_csv = os.path.join(out_dir, f"{group_name}_agg_icd10.csv")

    raw_out.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    aggregated.to_csv(agg_csv, index=False, encoding="utf-8-sig")

    info(f"Saved: {raw_csv}")
    info(f"Saved: {agg_csv}")

    return aggregated


def export_step1_class1_mids_with_icd_and_report(
    step1_df: pd.DataFrame,
    unique_mid_icd: pd.DataFrame,
    raw_report_file: str,
    out_dir: str,
    norm_map: Dict[str, Tuple[str, str]],
    norm_set: set,
) -> None:
    """
    Export step1 class 1 cases that have at least one matched ICD10 category,
    together with raw report and aggregated ICD10 information.
    """
    step1_class1_mids = set(step1_df.loc[step1_df["step1_pred"] == 1, "mid"].astype(str))
    sub = unique_mid_icd[unique_mid_icd["m_id"].astype(str).isin(step1_class1_mids)].copy()

    out_path = os.path.join(out_dir, "step1_class1_mids_with_icd10_and_raw_report.csv")

    if sub.empty:
        pd.DataFrame(columns=[
            "mid", "raw_report", "icd10_list", "agg_icd10_key_list", "agg_category_name_list"
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        info(f"No step1 class1 cases with ICD10 data. Saved empty file: {out_path}")
        return

    rows = []
    for mid, group in sub.groupby("m_id"):
        mid = str(mid).strip()
        raw_codes = sorted(set(group["icd10"].astype(str).tolist()))

        agg_keys = []
        agg_names = []

        for code in raw_codes:
            norm_key = exact_match_icd10(code, norm_set)
            if norm_key is None:
                continue
            original_key, category_name = norm_map[norm_key]
            agg_keys.append(original_key)
            agg_names.append(category_name)

        agg_keys = sorted(set(agg_keys))
        agg_names = sorted(set(agg_names))

        if len(agg_keys) == 0:
            continue

        rows.append({
            "mid": mid,
            "icd10_list": "; ".join(raw_codes),
            "agg_icd10_key_list": "; ".join(agg_keys),
            "agg_category_name_list": "; ".join(agg_names),
        })

    export_df = pd.DataFrame(rows)

    if export_df.empty:
        pd.DataFrame(columns=[
            "mid", "raw_report", "icd10_list", "agg_icd10_key_list", "agg_category_name_list"
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        info(f"No step1 class1 cases matched ICD10 mapping. Saved empty file: {out_path}")
        return

    export_df = export_df.sort_values("mid").reset_index(drop=True)

    raw_df = robust_read_csv(raw_report_file, dtype=str)
    raw_df = normalize_columns(raw_df)

    if "mid" not in raw_df.columns:
        raise ValueError(f"Raw report file is missing 'mid' column: {raw_report_file}")

    raw_report_col = None
    for candidate in ["raw_report", "report", "report_text", "rawreport"]:
        if candidate in raw_df.columns:
            raw_report_col = candidate
            break

    if raw_report_col is None:
        raise ValueError(
            f"Could not find raw report column in {raw_report_file}. "
            f"Available columns: {list(raw_df.columns)}"
        )

    raw_df["mid"] = raw_df["mid"].astype(str).str.strip()
    raw_keep = raw_df[["mid", raw_report_col]].copy().drop_duplicates(subset=["mid"])

    export_df = export_df.merge(raw_keep, on="mid", how="left")
    export_df = export_df.rename(columns={raw_report_col: "raw_report"})
    export_df = export_df[
        ["mid", "raw_report", "icd10_list", "agg_icd10_key_list", "agg_category_name_list"]
    ]

    export_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    info(f"Saved: {out_path}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    allowed_mids = load_mid_reference(MID_REF_FILE) if MID_REF_FILE else None

    model_results = {
        "ds": read_result_csv(DS_FILE, allowed_mids=allowed_mids),
        "qw": read_result_csv(QW_FILE, allowed_mids=allowed_mids),
        "third": read_result_csv(THIRD_FILE, allowed_mids=allowed_mids),
    }

    all_mids = collect_union_mids(
        model_results["ds"],
        model_results["qw"],
        model_results["third"],
    )

    if allowed_mids is not None:
        all_mids = [mid for mid in all_mids if mid in allowed_mids]

    info(f"Total mids included in analysis: {len(all_mids)}")

    step1_df, step2_df = build_prediction_tables(model_results, all_mids)

    step1_pred_path = os.path.join(OUTPUT_DIR, "step1_predictions.csv")
    step2_pred_path = os.path.join(OUTPUT_DIR, "step2_predictions.csv")

    step1_df.to_csv(step1_pred_path, index=False, encoding="utf-8-sig")
    step2_df.to_csv(step2_pred_path, index=False, encoding="utf-8-sig")

    info(f"Saved: {step1_pred_path}")
    info(f"Saved: {step2_pred_path}")

    save_stage_counts(step1_df, step2_df, OUTPUT_DIR)

    merged = load_and_filter_merged(MERGED_FILE, WINDOW_DAYS)
    unique_mid_icd = merged[["m_id", "icd10"]].drop_duplicates().copy()

    all_icd_df = robust_read_csv(ALL_ICD10_FILE, dtype=str).dropna(
        subset=["icd10_category", "category_name"]
    ).copy()

    norm_map, norm_set, skeleton = build_exact_matcher(all_icd_df)

    export_step1_class1_mids_with_icd_and_report(
        step1_df=step1_df,
        unique_mid_icd=unique_mid_icd,
        raw_report_file=RAW_REPORT_FILE,
        out_dir=OUTPUT_DIR,
        norm_map=norm_map,
        norm_set=norm_set,
    )

    # Step1 ICD10 analysis
    step1_dir = os.path.join(OUTPUT_DIR, "step1_icd10")
    ensure_dir(step1_dir)

    step1_cls1_mids = set(step1_df.loc[step1_df["step1_pred"] == 1, "mid"].astype(str))
    step1_cls2_mids = set(step1_df.loc[step1_df["step1_pred"] == 2, "mid"].astype(str))

    step1_cls1_agg = analyze_one_group_icd(
        group_name="step1_class1",
        mids=step1_cls1_mids,
        unique_mid_icd=unique_mid_icd,
        norm_map=norm_map,
        norm_set=norm_set,
        out_dir=step1_dir,
    )
    step1_cls2_agg = analyze_one_group_icd(
        group_name="step1_class2",
        mids=step1_cls2_mids,
        unique_mid_icd=unique_mid_icd,
        norm_map=norm_map,
        norm_set=norm_set,
        out_dir=step1_dir,
    )

    step1_rank = build_rank(
        step1_cls1_agg,
        step1_cls2_agg,
        skeleton,
        name_a="step1_cls1",
        name_b="step1_cls2",
    )
    step1_rank_path = os.path.join(step1_dir, "step1_icd10_rank.csv")
    step1_rank.to_csv(step1_rank_path, index=False, encoding="utf-8-sig")
    info(f"Saved: {step1_rank_path}")

    # Step2 ICD10 analysis
    step2_dir = os.path.join(OUTPUT_DIR, "step2_icd10")
    ensure_dir(step2_dir)

    if step2_df.empty:
        info("step2_df is empty. Skipping step2 ICD10 analysis.")
    else:
        step2_cons1_mids = set(step2_df.loc[step2_df["step2_pred"] == 1, "mid"].astype(str))
        step2_cons2_mids = set(step2_df.loc[step2_df["step2_pred"] == 2, "mid"].astype(str))

        step2_cons1_agg = analyze_one_group_icd(
            group_name="step2_consensus1",
            mids=step2_cons1_mids,
            unique_mid_icd=unique_mid_icd,
            norm_map=norm_map,
            norm_set=norm_set,
            out_dir=step2_dir,
        )
        step2_cons2_agg = analyze_one_group_icd(
            group_name="step2_consensus2",
            mids=step2_cons2_mids,
            unique_mid_icd=unique_mid_icd,
            norm_map=norm_map,
            norm_set=norm_set,
            out_dir=step2_dir,
        )

        step2_rank = build_rank(
            step2_cons1_agg,
            step2_cons2_agg,
            skeleton,
            name_a="step2_cons1",
            name_b="step2_cons2",
        )
        step2_rank_path = os.path.join(step2_dir, "step2_consensus12_icd10_rank.csv")
        step2_rank.to_csv(step2_rank_path, index=False, encoding="utf-8-sig")
        info(f"Saved: {step2_rank_path}")

    info("All analysis finished.")


if __name__ == "__main__":
    main()
