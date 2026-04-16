import csv
import pandas as pd
import re
from tqdm import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from vllm import LLM, SamplingParams

# ✅ 修改为Qwen3格式模型路径
MODEL_PATH = "/gpfs/data/shenlab/LLMs/Qwen3-32B"  # ← 修改此处为Qwen3模型路径

# ✅ 修改为你的输入CSV路径
INPUT_CSV_PATH = "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv"

# ✅ 修改为你的Prompt文件路径
PROMPT_FILE_PATH = "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt"

# ✅ 输出文件路径
OUTPUT_CSV_PATH = "/gpfs/home/wr2215/ms_mri_qwen3/processed_reports_classification_qw_14_680.csv"
INVALID_LABELS_PATH = "/gpfs/home/wr2215/ms_mri_qwen3/invalid_labels_qwen3_14_680.csv"

def load_system_message(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def build_prompt(system_message, report_id, report_input):
    return [
        {"role": "system", "content":  "You are a helpful assistant."},
        {"role": "user", "content": system_message},
        {"role": "user", "content": "Here's an example of how to process a report:"},
        {"role": "user", "content": """
Example Input:
ID: 6
input: 
Clinical indication:
39-year-old female headache.

Technique:
Multiplanar multi-sequence MR images of the brain were obtained without intravenous contrast administration.

Comparison:
Brain MRI 3/21/2013

Findings:
...

Impression:
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left precentral gyral cavernous malformation.

Others:
Final Report: Dictated by Resident Lindsay Griffin MD and Signed by Attending Mari Hagiwara MD 11/12/2014 11:31 AM

Example Output:
3
"""},
        {"role": "user", "content": f"Now, please process this report:\n\nID: {report_id}\ninput: {report_input}"}
    ]

def extract_numeric_label(text):
    text = str(text).lower()
    if '</think>' in text:
        post_think = text.split('</think>')[1]
        match = re.search(r'\d+', post_think)
        if match:
            return int(match.group())

    match = re.search(r"(answer|select|choice|prediction|category)[^\d]{0,5}(\d)", text)
    if match:
        return int(match.group(2))

    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[-1])

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
                label = extract_numeric_label(result_text)
                results.append({
                    'id': batch[j]['id'],
                    'result': label,
                    'raw_output': result_text
                })
                if label is None or label not in [1, 2, 3]:
                    print(f"⚠️ Invalid label (not in 1–3): {batch[j]['id']} — raw: {result_text}")
                    bad_ids.append(batch[j]['id'])
        except Exception as e:
            print(f"[❌ ERROR] Batch {i} failed: {e}")
            for row in batch:
                results.append({
                    'id': row['id'],
                    'result': None,
                    'raw_output': f"[ERROR] {str(e)}"
                })
    return results, bad_ids

def main():
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=8192,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        top_k=40
    )

    df = pd.read_csv(INPUT_CSV_PATH)
    reports = df.to_dict(orient="records")

    system_message = load_system_message(PROMPT_FILE_PATH)

    results, bad_ids = batch_process_reports(llm, sampling_params, system_message, reports, batch_size=8)

    pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH, index=False)
    print("✅ 所有报告处理完成，结果已保存。")

    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)

if __name__ == "__main__":
    main()
