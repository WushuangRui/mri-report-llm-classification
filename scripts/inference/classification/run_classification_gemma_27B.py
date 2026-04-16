#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_classification_gemma_27B.py
--------------------------------
Classify brain MRI reports using Gemma-3-27B-IT (AWQ-INT4) via vLLM.
Valid output labels: 1, 2, 3

Input  (--in_csv):      CSV with columns [id, json_report]
Output (--out_csv):     CSV with columns [id, result_cls, raw_output]
       (--invalid_csv): CSV listing IDs whose label fell outside {1, 2, 3}

Usage:
    python run_classification_gemma_27B.py \
        --model       /path/to/gemma-3-27b-it-int4-awq \
        --in_csv      /path/to/input.csv \
        --prompt_path /path/to/prompt.txt \
        --out_csv     /path/to/output.csv \
        --invalid_csv /path/to/invalid.csv
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import csv
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

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


def parse_args():
    parser = argparse.ArgumentParser(description="MRI report classification with Gemma-27B via vLLM")
    parser.add_argument("--model",       required=True, help="Path to model weights")
    parser.add_argument("--in_csv",      required=True, help="Input CSV path")
    parser.add_argument("--prompt_path", required=True, help="Prompt txt path")
    parser.add_argument("--out_csv",     required=True, help="Output CSV path")
    parser.add_argument("--invalid_csv", required=True, help="Invalid labels CSV path")
    return parser.parse_args()


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
ID: M0005
input:
{
  "quality_dec": false,
  "clinical_indication": [
    {
      "age": null,
      "gender": null,
      "symptom": null,
      "history": null
    }
  ],
  "comparison": [],
  "lesions": [
    {
      "type": "demyelinating lesions",
      "location": [
        {
          "side": null,
          "region": [
            "periventricular",
            "juxtacortical",
            "infratentorial white matter"
          ]
        }
      ],
      "size": null,
      "radiological_changes": {
        "T2/FLAIR": [
          "hyperintensity"
        ]
      },
      "characteristics": [
        "ovoid",
        "Dawson fingers"
      ],
      "compare_pre": null,
      "diagnosis": {
        "multiple sclerosis": [
          "highly suggestive"
        ]
      }
    }
  ],
  "additional_findings": null
}

Example Output:3
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


def batch_process_reports(llm: LLM, sampling_params: SamplingParams,
                          system_message: str, reports, batch_size=8):
    results, bad_ids = [], []

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
                    "id":         batch[j]["id"],
                    "result_cls": cls,
                    "raw_output": sanitize_one_line(result_text),
                })
                if cls not in VALID_LABELS:
                    print(f"[WARN] Invalid output: {batch[j]['id']} — raw: {sanitize_one_line(result_text)[:200]}")
                    bad_ids.append(batch[j]["id"])
        except Exception as e:
            err = f"[ERROR] {e}"
            for row in batch:
                results.append({"id": row["id"], "result_cls": None, "raw_output": err})
                bad_ids.append(row["id"])

    return results, bad_ids


def main():
    args = parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=False,
        seed=SEED,
        quantization="awq",
        dtype="float16",
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )

    df = pd.read_csv(args.in_csv)
    need_cols = {"id", "json_report"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {need_cols}")

    system_message = load_system_message(args.prompt_path)
    results, bad_ids = batch_process_reports(
        llm, sampling_params, system_message, df.to_dict(orient="records"), BATCH_SIZE)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out = pd.DataFrame(results)
    # Use nullable integer to avoid 1.0 / 2.0 / 3.0 formatting
    df_out["result_cls"] = df_out["result_cls"].astype("Int64")
    df_out.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_ALL,
                  escapechar="\\", lineterminator="\n", encoding="utf-8")
    print(f"[INFO] Done. Results saved to {args.out_csv}")

    os.makedirs(os.path.dirname(args.invalid_csv) or ".", exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(args.invalid_csv, index=False)
    print(f"[INFO] Invalid label count: {len(bad_ids)}, saved to {args.invalid_csv}")


if __name__ == "__main__":
    main()
