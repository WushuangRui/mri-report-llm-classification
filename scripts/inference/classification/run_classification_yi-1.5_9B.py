#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_classification_yi-1.5_9B.py
---------------------------------
Classify brain MRI reports using Yi-1.5-9B-Chat-16K via vLLM.
Valid output labels: 1, 2, 3
The model is prompted to return a single digit; max_tokens is set to 4.

Input  (INPUT_CSV_PATH):      CSV with columns [id, json_report]
Output (OUTPUT_CSV_PATH):     CSV with columns [id, result, raw_output]
       (INVALID_LABELS_PATH): CSV listing IDs whose label fell outside {1, 2, 3}

All paths and runtime parameters can be overridden via environment variables:
    python run_classification_yi-1.5_9B.py
    MODEL_PATH=/new/path python run_classification_yi-1.5_9B.py
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
    "/gpfs/data/shenlab/LLMs/Yi-1.5-9B-Chat-16K")
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_reporttojson/input&result/processed_reports_7074_age_ds_split_utf8.csv")
PROMPT_FILE_PATH = os.environ.get("PROMPT_FILE_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt")
OUTPUT_CSV_PATH = os.environ.get("OUTPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_yi-1.5_9B/processed_reports_classification_yi15_9b16k_7074_14.csv")
INVALID_LABELS_PATH = os.environ.get("INVALID_LABELS_PATH",
    "/gpfs/home/wr2215/ms_mri_yi-1.5_9B/invalid_labels_yi15_9b16k_7074.csv")
# ─────────────────────────────────────────────────────────────────────────────

# ── Runtime parameters (overridable via environment variables) ────────────────
TP_SIZE       = int(os.environ.get("TP_SIZE", "1"))         # set to GPU count for multi-GPU
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "8"))      # reduce to 4/2/1 if VRAM is tight
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192")) # supports up to 16384
GPU_MEM_UTIL  = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
SEED          = int(os.environ.get("SEED", "1"))

# Sampling: deterministic; very short output (model returns a single digit)
sampling_params = SamplingParams(
    max_tokens  = int(os.environ.get("MAX_TOKENS", "4")),
    temperature = float(os.environ.get("TEMPERATURE", "0.0")),
    top_p       = float(os.environ.get("TOP_P", "1.0")),
    top_k       = int(os.environ.get("TOP_K", "-1")),
    seed        = SEED,
)

VALID_LABELS = {1, 2, 3}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_system_message(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def sanitize_one_line(s: str) -> str:
    """Collapse a multi-line string to one line to prevent CSV row misalignment."""
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(system_message: str, report_id, report_input):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You must output only one single digit: 1 or 2 or 3. "
                "Do not output any explanation, words, punctuation, or extra text."
            )
        },
        {"role": "user", "content": system_message},
        {"role": "user", "content": "Here's an example of how to process a report:"},
        {"role": "user", "content": """
Example Input:
ID: 6
input:
Clinical indication:
39-year-old female headache.

Findings:
...

Impression:
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left precentral
gyral cavernous malformation.

Example Output:
3
"""},
        {
            "role": "user",
            "content": (
                f"ID: {report_id}\n"
                f"input: {report_input}\n\n"
                "IMPORTANT:\n"
                "- Output ONLY one digit\n"
                "- Allowed outputs: 1, 2, or 3\n"
                "- Do NOT explain\n"
                "- Do NOT output any other text\n\n"
                "Answer:"
            )
        },
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
        batch_prompts = [build_prompt(system_message, row['id'], row['json_report']) for row in batch]

        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)
            for j, output in enumerate(outputs):
                try:
                    result_text = output.outputs[0].text.strip() if output.outputs else ""
                except Exception:
                    result_text = str(output)

                label = extract_label(result_text)
                results.append({
                    "id": batch[j]["id"],
                    "result": label,
                    "raw_output": sanitize_one_line(result_text),
                })
                raw_full_jsonl.append({"id": batch[j]["id"], "raw_output_full": result_text})

                if label not in VALID_LABELS:
                    print(f"[WARN] Invalid output: {batch[j]['id']} — raw: {sanitize_one_line(result_text)[:200]}")
                    bad_ids.append(batch[j]["id"])

        except Exception as e:
            err = f"[ERROR] {e}"
            for row in batch:
                results.append({"id": row["id"], "result": None, "raw_output": err})
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
    )

    df = pd.read_csv(INPUT_CSV_PATH)
    need_cols = {"id", "json_report"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {need_cols}")

    system_message = load_system_message(PROMPT_FILE_PATH)
    results, bad_ids, _ = batch_process_reports(
        llm, sampling_params, system_message, df.to_dict(orient="records"), BATCH_SIZE)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_out = pd.DataFrame(results)
    if "result" in df_out.columns:
        # Use nullable integer to avoid 3.0 formatting
        df_out["result"] = df_out["result"].astype("Int64")

    df_out.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_ALL,
                  escapechar="\\", lineterminator="\n", encoding="utf-8")
    print(f"[INFO] Done. Results saved to {OUTPUT_CSV_PATH}")

    os.makedirs(os.path.dirname(INVALID_LABELS_PATH), exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)
    print(f"[INFO] Invalid/failed count: {len(bad_ids)}, saved to {INVALID_LABELS_PATH}")


if __name__ == "__main__":
    main()
