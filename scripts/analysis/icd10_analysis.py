#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage ensemble + ICD10 analysis (unlabeled data)

Usage:
python analysis_unlabeled.py \
    --ds_file path/to/ds.csv \
    --qw_file path/to/qw.csv \
    --third_file path/to/model.csv \
    --merged_file path/to/merged.csv \
    --icd_map path/to/icd10_map.csv \
    --raw_report path/to/raw_report.csv \
    --out_dir outputs/
"""

import os
import re
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 参数解析（🔥 GitHub关键）
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_file", required=True)
    parser.add_argument("--qw_file", required=True)
    parser.add_argument("--third_file", required=True)

    parser.add_argument("--merged_file", required=True)
    parser.add_argument("--icd_map", required=True)
    parser.add_argument("--raw_report", required=True)

    parser.add_argument("--out_dir", default="outputs/")
    parser.add_argument("--window_days", type=int, default=10)

    return parser.parse_args()


# =========================================================
# 工具函数
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_columns(df):
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def read_csv(path):
    return pd.read_csv(path, dtype=str)


# =========================================================
# 读取模型结果
# =========================================================
def read_result(path):
    df = read_csv(path)
    df = normalize_columns(df)

    col = "result_cls" if "result_cls" in df.columns else "result"
    df["mid"] = df["mid"].astype(str).str.strip()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    return dict(zip(df["mid"], df[col]))


# =========================================================
# Step1 + Step2
# =========================================================
def map_step1(x):
    if pd.isna(x):
        return None
    return 1 if x in [1, 2] else 2


def step1_fuse(ds, qw, third, mids):
    out = {}
    for m in mids:
        vals = [map_step1(ds.get(m)), map_step1(qw.get(m)), map_step1(third.get(m))]
        out[m] = 1 if vals == [1, 1, 1] else 2
    return out


def step2_fuse(ds, qw, third, mids):
    out = {}
    for m in mids:
        vals = [ds.get(m), qw.get(m), third.get(m)]
        if all(v == vals[0] for v in vals):
            out[m] = vals[0]
    return out


# =========================================================
# ICD过滤
# =========================================================
EXCLUDE = re.compile(
    r"history|possible|suspected|rule out|r/o|疑似|可疑",
    re.I
)


def filter_merged(df, window_days):
    df = normalize_columns(df)

    df = df[df["dx_type"].str.lower() == "encounter diagnosis"]

    df["scan"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df["entry"] = pd.to_datetime(df["entry_date"], errors="coerce")

    df = df[(df["entry"] - df["scan"]).abs() <= pd.Timedelta(days=window_days)]

    mask = df["dx_name"].str.contains(EXCLUDE, na=False) | \
           df["status_comment"].str.contains(EXCLUDE, na=False)

    df = df[~mask]

    return df[["m_id", "icd10"]].drop_duplicates()


# =========================================================
# ICD统计
# =========================================================
def icd_stats(mids, merged):
    sub = merged[merged["m_id"].isin(mids)]
    freq = sub.groupby("icd10")["m_id"].nunique().reset_index()
    freq.columns = ["icd10", "count"]
    return freq.sort_values("count", ascending=False)


# =========================================================
# 主函数
# =========================================================
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # 1️⃣ 读取模型
    ds = read_result(args.ds_file)
    qw = read_result(args.qw_file)
    third = read_result(args.third_file)

    mids = sorted(set(ds) | set(qw) | set(third))

    # 2️⃣ step1
    step1 = step1_fuse(ds, qw, third, mids)

    # 3️⃣ step2
    mids_s1 = [m for m in mids if step1[m] == 1]
    step2 = step2_fuse(ds, qw, third, mids_s1)

    # 保存
    pd.DataFrame({"mid": list(step1.keys()), "step1": list(step1.values())}) \
        .to_csv(os.path.join(args.out_dir, "step1.csv"), index=False)

    pd.DataFrame({"mid": list(step2.keys()), "step2": list(step2.values())}) \
        .to_csv(os.path.join(args.out_dir, "step2.csv"), index=False)

    # 4️⃣ ICD
    merged = filter_merged(read_csv(args.merged_file), args.window_days)

    s1_cls1 = [m for m in mids if step1[m] == 1]
    s1_cls2 = [m for m in mids if step1[m] == 2]

    icd1 = icd_stats(s1_cls1, merged)
    icd2 = icd_stats(s1_cls2, merged)

    icd1.to_csv(os.path.join(args.out_dir, "step1_cls1_icd.csv"), index=False)
    icd2.to_csv(os.path.join(args.out_dir, "step1_cls2_icd.csv"), index=False)

    print("✅ Done")


if __name__ == "__main__":
    main()
