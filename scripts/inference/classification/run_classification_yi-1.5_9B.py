#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import csv
import json
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ========= 路径（按需修改或用环境变量覆盖）=========
MODEL_PATH = os.environ.get("MODEL_PATH", "/gpfs/data/shenlab/LLMs/Yi-1.5-9B-Chat-16K")

INPUT_CSV_PATH = os.environ.get(
    "INPUT_CSV_PATH",
    #"/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset.csv"
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_reporttojson/input&result/processed_reports_7074_age_ds_split_utf8.csv"
)
PROMPT_FILE_PATH = os.environ.get(
    "PROMPT_FILE_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt"
)

OUTPUT_CSV_PATH = os.environ.get(
    "OUTPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_yi-1.5_9B/processed_reports_classification_yi15_9b16k_7074_14.csv"
)
INVALID_LABELS_PATH = os.environ.get(
    "INVALID_LABELS_PATH",
    "/gpfs/home/wr2215/ms_mri_yi-1.5_9B/invalid_labels_yi15_9b16k_7074.csv"
)

# ========= 运行参数（可被环境变量覆盖）=========
TP_SIZE        = int(os.environ.get("TP_SIZE", "1"))          # 多卡时设为 GPU 张数
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "8"))       # 显存吃紧可降到 4/2/1
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "8192")) # Yi-1.5-9B-Chat-16K 可设到 16384
GPU_MEM_UTIL   = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
SEED           = int(os.environ.get("SEED", "1"))

# 采样设置：改成更适合“只输出一位数”
sampling_params = SamplingParams(
    max_tokens   = int(os.environ.get("MAX_TOKENS", "4")),
    temperature  = float(os.environ.get("TEMPERATURE", "0.0")),
    top_p        = float(os.environ.get("TOP_P", "1.0")),
    top_k        = int(os.environ.get("TOP_K", "-1")),
    seed=SEED,
)

# ========= 工具函数 =========
def load_system_message(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def sanitize_one_line(s: str) -> str:
    """把长文本压成一行，便于 CSV 不错位"""
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_prompt(system_message: str, report_id, report_input):
    # 按你的模板：固定 system，然后把自定义 system_message 作为 user 发送
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

VALID_LABELS = {1, 2, 3}

def strip_think(text: str) -> str:
    """移除 <think>...</think> 内容，避免抓到思维链里的数字。"""
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in t.lower():
        t = t.split("</think>", 1)[1]
    return t.strip()

def extract_label(text: str):
    """
    只返回 1/2/3 或 None
    """
    t = strip_think(text)

    # 先看整个输出是不是单独的 1/2/3
    t_clean = t.strip()
    if t_clean in {"1", "2", "3"}:
        return int(t_clean)

    # 再找独立出现的 1/2/3
    m = re.findall(r"(?<!\d)([1-3])(?!\d)", t)
    if m:
        return int(m[-1])

    return None

# ========= 主流程 =========
def batch_process_reports(llm: LLM, sampling_params: SamplingParams, system_message: str, reports, batch_size=8):
    results = []
    bad_ids = []
    raw_full_jsonl = []

    for i in tqdm(range(0, len(reports), batch_size), desc="Processing reports"):
        batch = reports[i:i + batch_size]
        batch_prompts = [build_prompt(system_message, row['id'], row['json_report']) for row in batch]

        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)
            for j, output in enumerate(outputs):
                result_text = ""
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

                raw_full_jsonl.append({
                    "id": batch[j]["id"],
                    "raw_output_full": result_text,
                })

                if label not in VALID_LABELS:
                    preview = sanitize_one_line(result_text)[:200]
                    print(f"⚠️ Invalid output: {batch[j]['id']} — raw: {preview}")
                    bad_ids.append(batch[j]["id"])

        except Exception as e:
            err = f"[ERROR] {e}"
            for row in batch:
                results.append({
                    "id": row["id"],
                    "result": None,
                    "raw_output": err,
                })
                raw_full_jsonl.append({
                    "id": row["id"],
                    "raw_output_full": err,
                })
                bad_ids.append(row["id"])

    return results, bad_ids, raw_full_jsonl

def main():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=False,  # Yi 官方权重一般不需要 remote code
        seed=SEED,
    )

    df = pd.read_csv(INPUT_CSV_PATH)
    need_cols = {"id", "json_report"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"输入 CSV 需要包含列：{need_cols}")
    reports = df.to_dict(orient="records")

    system_message = load_system_message(PROMPT_FILE_PATH)

    results, bad_ids, raw_full_jsonl = batch_process_reports(
        llm=llm,
        sampling_params=sampling_params,
        system_message=system_message,
        reports=reports,
        batch_size=BATCH_SIZE
    )

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_out = pd.DataFrame(results)

    # 结果列用可空整数，避免 3.0
    if "result" in df_out.columns:
        df_out["result"] = df_out["result"].astype("Int64")

    df_out.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        encoding="utf-8"
    )
    print("✅ 所有报告处理完成，结果已保存：", OUTPUT_CSV_PATH)

    os.makedirs(os.path.dirname(INVALID_LABELS_PATH), exist_ok=True)
    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)
    print(f"⚠️ 无效/失败条数：{len(bad_ids)}，清单已保存：{INVALID_LABELS_PATH}")

if __name__ == "__main__":
    main()