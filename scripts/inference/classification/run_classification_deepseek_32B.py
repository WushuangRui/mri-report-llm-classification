"""
run_classification_deepseek.py
-------------------------------
Classify brain MRI reports using DeepSeek-R1-Distill-Qwen-32B via vLLM.
Valid output labels: 1, 2, 3

Input  (--in_csv):      CSV with columns [id, json_report]
Output (--out_csv):     CSV with columns [id, result_cls, raw_output]
       (--invalid_csv): CSV listing IDs whose label fell outside {1, 2, 3}

Usage:
    python run_classification_deepseek.py \
        --model       /path/to/DeepSeek-R1-Distill-Qwen-32B \
        --in_csv      /path/to/input.csv \
        --prompt_path /path/to/prompt.txt \
        --out_csv     /path/to/output.csv \
        --invalid_csv /path/to/invalid.csv
"""

import re
import argparse
import pandas as pd
from tqdm import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from vllm import LLM, SamplingParams

VALID_LABELS = {1, 2, 3}


def parse_args():
    parser = argparse.ArgumentParser(description="MRI report classification with DeepSeek via vLLM")
    parser.add_argument("--model",       required=True, help="Path to model weights")
    parser.add_argument("--in_csv",      required=True, help="Input CSV path")
    parser.add_argument("--prompt_path", required=True, help="Prompt txt path")
    parser.add_argument("--out_csv",     required=True, help="Output CSV path")
    parser.add_argument("--invalid_csv", required=True, help="Invalid labels CSV path")
    return parser.parse_args()


def load_system_message(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_prompt(system_message, report_id, report_input):
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": system_message},
        {"role": "user", "content": "Here's an example of how to process a report:"},
        {"role": "user", "content": """
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
    results, bad_ids = [], []
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
    args = parse_args()

    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=0.95)

    # DeepSeek-R1 is a reasoning model; keep max_tokens high for chain-of-thought,
    # use temperature=0.0 for deterministic classification output.
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, top_k=-1)

    df = pd.read_csv(args.in_csv)
    system_message = load_system_message(args.prompt_path)
    results, bad_ids = batch_process_reports(llm, sampling_params, system_message,
                                             df.to_dict(orient="records"), batch_size=8)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"[INFO] Done. Results saved to {args.out_csv}")

    os.makedirs(os.path.dirname(args.invalid_csv) or ".", exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(args.invalid_csv, index=False)
    print(f"[INFO] Invalid label count: {len(bad_ids)}, saved to {args.invalid_csv}")


if __name__ == "__main__":
    main()
