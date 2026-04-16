#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_classification_gemma_27B.py
--------------------------------
Classify brain MRI reports using Gemma-3-27B-IT (AWQ-INT4) via vLLM.
Valid output labels: 1, 2, 3

Input  (INPUT_CSV_PATH):      CSV with columns [id, json_report]
Output (OUTPUT_CSV_PATH):     CSV with columns [id, result_cls, raw_output]
       (INVALID_LABELS_PATH): CSV listing IDs whose label fell outside {1, 2, 3}

All paths and runtime parameters can be overridden via environment variables:
    python run_classification_gemma_27B.py
    MODEL_PATH=/new/path python run_classification_gemma_27B.py
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import csv
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Paths (override via environment variables or edit defaults below) ──────────
MODEL_PATH = os.environ.get("MODEL_PATH",
    "/gpfs/data/shenlab/LLMs/gemma-3-27b-it-int4-awq")
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv")
PROMPT_FILE_PATH = os.environ.get("PROMPT_FILE_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt")
OUTPUT_CSV_PATH = os.environ.get("OUTPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_gemma_27B/processed_reports_classification_gemma3_27b_awq_14_680_1.csv")
INVALID_LABELS_PATH = os.environ.get("INVALID_LABELS_PATH",
    "/gpfs/home/wr2215/ms_mri_gemma_27B/invalid_labels_gemma3_27b_awq_14_680_1.csv")
# ─────────────────────────────────────────────────────────────────────────────

# ── Runtime parameters (overridable via environment variables) ────────────────
TP_SIZE       = int(os.environ.get("TP_SIZE", "1"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "8"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL  = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
SEED          = int(os.environ.get("SEED", "1"))

# Sampling: greedy / deterministic (stable for classification)
sampling_params = SamplingParams(
    max_tokens  = int(os.environ.get("MAX_TOKENS", "128")),
    temperature = float(os.environ.get("TEMPERATURE", "0.0")),
    top_p       = float(os.environ.get("TOP_P", "1.0")),
    top_k       = int(os.environ.get("TOP_K", "-1")),
    seed        = SEED,
)

VALID_LABELS = {1, 2, 3}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_system_message(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def sanitize_one_line(s: str) -> str:
    """Collapse a multi-line string to one line to prevent CSV row misalignment."""
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(system_message: str, report_id, report_input):
    example_user = """Here's an example of how to process a report:

Example Input:
ID: 6
input:
Clinical indication:
39-year-old female headache.

Impression:
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left
precentral gyral cavernous malformation.

Example Output:
"""
    # Keep as-is; do not modify the few-shot example fed to the LLM
    example_assistant = "3"

    task_user = f"Now, please process this report:\n\nID: {report_id}\ninput: {report_input}"

    return [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": system_message + "\n\n" + example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user",      "content": task_user},
    ]


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks to avoid capturing chain-of-thought digits."""
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in t.lower():
        t = t.split("</think>", 1)[1]
    return t.strip()


def extract_label(text: str):
    """Return 1, 2, or 3; None if no valid label found."""
    t = strip_think(text)
    if t.strip() in {"1", "2", "3"}:
        return int(t.strip())
    m = re.findall(r"(?<!\d)([1-3])(?!\d)", t)
    if m:
        return int(m[-1])
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────
def batch_process_reports(llm: LLM, sampling_params: SamplingParams,
                          system_message: str, reports, batch_size=8):
    results, bad_ids, raw_full_jsonl = [], [], []

    for i in tqdm(range(0, len(reports), batch_size), desc="Processing reports"):
        batch = reports[i:i + batch_size]
        batch_prompts = [build_prompt(system_message, row["id"], row["json_report"]) for row in batch]

        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)
            for j, output in enumerate(outputs):
                try:
                    result_text = output.outputs[0].text.strip() if output.outputs else ""
                except Exception:
                    result_text = str(output)

                cls = extract_label(result_text)
                results.append({
                    "id": batch[j]["id"],
                    "result_cls": cls,
                    "raw_output": sanitize_one_line(result_text),
                })
                raw_full_jsonl.append({"id": batch[j]["id"], "raw_output_full": result_text})

                if cls not in VALID_LABELS:
                    print(f"[WARN] Invalid output: {batch[j]['id']} — raw: {sanitize_one_line(result_text)[:200]}")
                    bad_ids.append(batch[j]["id"])

        except Exception as e:
            err = f"[ERROR] {e}"
            for row in batch:
                results.append({"id": row["id"], "result_cls": None, "raw_output": err})
                raw_full_jsonl.append({"id": row["id"], "raw_output_full": err})
                bad_ids.append(row["id"])

    return results, bad_ids, raw_full_jsonl


def main():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=False,
        seed=SEED,
        quantization="awq",
        dtype="float16",
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )

    df = pd.read_csv(INPUT_CSV_PATH)
    need_cols = {"id", "json_report"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {need_cols}")

    system_message = load_system_message(PROMPT_FILE_PATH)
    results, bad_ids, _ = batch_process_reports(
        llm, sampling_params, system_message, df.to_dict(orient="records"), BATCH_SIZE)

    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(results)
    if "result_cls" in df_out.columns:
        # Use nullable integer to avoid 1.0 / 2.0 / 3.0 formatting
        df_out["result_cls"] = df_out["result_cls"].astype("Int64")

    df_out.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_ALL,
                  escapechar="\\", lineterminator="\n", encoding="utf-8")
    print(f"[INFO] Done. Results saved to {OUTPUT_CSV_PATH}")

    bad_dir = os.path.dirname(INVALID_LABELS_PATH)
    if bad_dir:
        os.makedirs(bad_dir, exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)
    print(f"[INFO] Invalid label count: {len(bad_ids)}, saved to {INVALID_LABELS_PATH}")


if __name__ == "__main__":
    main()
