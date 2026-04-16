"""
run_classification_qwen3.py
----------------------------
Classify brain MRI reports using Qwen3-32B via vLLM.
Valid output labels: 1, 2, 3

Input  (INPUT_CSV_PATH):      CSV with columns [id, json_report]
Output (OUTPUT_CSV_PATH):     CSV with columns [id, result_cls, raw_output]
       (INVALID_LABELS_PATH): CSV listing IDs whose label fell outside {1, 2, 3}

Configure paths in the constants block below, then run:
    python run_classification_qwen3.py
"""

import re
import pandas as pd
from tqdm import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from vllm import LLM, SamplingParams

# ── Paths (edit before running) ───────────────────────────────────────────────
MODEL_PATH          = "/gpfs/data/shenlab/LLMs/Qwen3-32B"
INPUT_CSV_PATH      = "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv"
PROMPT_FILE_PATH    = "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt"
OUTPUT_CSV_PATH     = "/gpfs/home/wr2215/ms_mri_qwen3/processed_reports_classification_qw_14_680.csv"
INVALID_LABELS_PATH = "/gpfs/home/wr2215/ms_mri_qwen3/invalid_labels_qwen3_14_680.csv"
# ─────────────────────────────────────────────────────────────────────────────

VALID_LABELS = {1, 2, 3}


def load_system_message(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_prompt(system_message, report_id, report_input):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": system_message},
        {"role": "user",   "content": "Here's an example of how to process a report:"},
        {"role": "user",   "content": """
Example Input:
ID: 6
input:
Clinical indication:
39-year-old female headache.

Technique:
Multiplanar multi-sequence MR images of the brain were obtained without intravenous contrast administration.

Findings:
...

Impression:
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left precentral
gyral cavernous malformation.

Example Output:
3
"""},
        {"role": "user", "content": f"Now, please process this report:\n\nID: {report_id}\ninput: {report_input}"}
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


def batch_process_reports(llm, sampling_params, system_message, reports, batch_size=8):
    results = []
    bad_ids = []
    for i in tqdm(range(0, len(reports), batch_size), desc="Processing reports"):
        batch = reports[i:i + batch_size]
        batch_prompts = [build_prompt(system_message, row['id'], row['json_report']) for row in batch]
        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)
            for j, output in enumerate(outputs):
                result_text = output.outputs[0].text.strip()
                label = extract_label(result_text)
                results.append({'id': batch[j]['id'], 'result_cls': label, 'raw_output': result_text})
                if label not in VALID_LABELS:
                    print(f"[WARN] Invalid label: {batch[j]['id']} — raw: {result_text}")
                    bad_ids.append(batch[j]['id'])
        except Exception as e:
            print(f"[ERROR] Batch {i} failed: {e}")
            for row in batch:
                results.append({'id': row['id'], 'result_cls': None, 'raw_output': f"[ERROR] {e}"})
    return results, bad_ids


def main():
    llm = LLM(model=MODEL_PATH, max_model_len=8192, gpu_memory_utilization=0.95)

    # Qwen3 is a reasoning model; keep max_tokens high for chain-of-thought,
    # use temperature=0.0 for deterministic classification output.
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, top_k=-1)

    df = pd.read_csv(INPUT_CSV_PATH)
    system_message = load_system_message(PROMPT_FILE_PATH)
    results, bad_ids = batch_process_reports(llm, sampling_params, system_message,
                                             df.to_dict(orient="records"), batch_size=8)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[INFO] Done. Results saved to {OUTPUT_CSV_PATH}")

    os.makedirs(os.path.dirname(INVALID_LABELS_PATH), exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)
    print(f"[INFO] Invalid label count: {len(bad_ids)}, saved to {INVALID_LABELS_PATH}")


if __name__ == "__main__":
    main()
