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
MODEL_PATH = os.environ.get("MODEL_PATH", "/gpfs/data/shenlab/LLMs/gemma-3-27b-it-int4-awq")

INPUT_CSV_PATH = os.environ.get(
    "INPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv"
)
PROMPT_FILE_PATH = os.environ.get(
    "PROMPT_FILE_PATH",
    "/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt"
)

OUTPUT_CSV_PATH = os.environ.get(
    "OUTPUT_CSV_PATH",
    "/gpfs/home/wr2215/ms_mri_gemma_27B/processed_reports_classification_gemma3_27b_awq_14_680_1.csv"
)
INVALID_LABELS_PATH = os.environ.get(
    "INVALID_LABELS_PATH",
    "/gpfs/home/wr2215/ms_mri_gemma_27B/invalid_labels_gemma3_27b_awq_14_680_1.csv"
)

# ========= 运行参数（可被环境变量覆盖）=========
TP_SIZE        = int(os.environ.get("TP_SIZE", "1"))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "8"))
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL   = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
SEED           = int(os.environ.get("SEED", "1"))

# ---- 采样设置（分类更稳，默认为贪心/确定性）----
sampling_params = SamplingParams(
    max_tokens   = int(os.environ.get("MAX_TOKENS", "128")),
    temperature  = float(os.environ.get("TEMPERATURE", "0.0")),
    top_p        = float(os.environ.get("TOP_P", "1.0")),
    top_k        = int(os.environ.get("TOP_K", "-1")),
    seed=SEED,
)

# ========= 工具函数 =========
def load_system_message(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def sanitize_one_line(s: str) -> str:
    """把长文本压成一行，便于 CSV 不错位"""
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
Mild interval increase in size with associated minimal vasogenic edema of the paramedian left precentral gyral cavernous malformation.

Example Output:
"""
    # 保持原样，不改 LLM 输入示例
    example_assistant = "30"

    task_user = f"""Now, please process this report:

ID: {report_id}
input: {report_input}"""

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": system_message + "\n\n" + example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": task_user},
    ]

# ========= 三分类解析逻辑：最终只输出 1/2/3 =========
VALID_CLASSES = {1, 2, 3}

def strip_think(text: str) -> str:
    """移除 <think>...</think>，避免抓到思维链里的数字。"""
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in t.lower():
        t = t.split("</think>", 1)[1]
    return t.strip()

def extract_class(text: str):
    """
    返回单一类别 1/2/3 或 None

    兼容：
    - 10 / 11 -> 1
    - 20 / 21 -> 2
    - 30      -> 3
    - 1 / 2 / 3 -> 1 / 2 / 3
    """
    t = strip_think(text)

    # 优先匹配两位码，避免把 10 先匹配成 1
    m = re.findall(r"(?<!\d)(10|11|20|21|30)(?!\d)", t)
    if m:
        code = m[-1]
        mapping = {
            "10": 1,
            "11": 1,
            "20": 2,
            "21": 2,
            "30": 3,
        }
        return mapping[code]

    # 兜底：兼容模型直接输出单个 1/2/3
    m1 = re.search(r"(?<!\d)([1-3])(?!\d)", t.strip())
    if m1:
        return int(m1.group(1))

    return None

# ========= 主流程 =========
def batch_process_reports(llm: LLM, sampling_params: SamplingParams, system_message: str, reports, batch_size=8):
    results = []
    bad_ids = []
    raw_full_jsonl = []

    for i in tqdm(range(0, len(reports), batch_size), desc="Processing reports"):
        batch = reports[i:i + batch_size]
        batch_prompts = [build_prompt(system_message, row["id"], row["json_report"]) for row in batch]

        try:
            outputs = llm.chat(batch_prompts, sampling_params, use_tqdm=False)
            for j, output in enumerate(outputs):
                result_text = ""
                try:
                    result_text = output.outputs[0].text.strip() if output.outputs else ""
                except Exception:
                    result_text = str(output)

                cls = extract_class(result_text)

                results.append({
                    "id": batch[j]["id"],
                    "result_cls": cls,
                    "raw_output": sanitize_one_line(result_text),
                })

                raw_full_jsonl.append({
                    "id": batch[j]["id"],
                    "raw_output_full": result_text,
                })

                if cls not in VALID_CLASSES:
                    preview = sanitize_one_line(result_text)[:200]
                    print(f"⚠️ Invalid output: {batch[j]['id']} — raw: {preview}")
                    bad_ids.append(batch[j]["id"])

        except Exception as e:
            err = f"[ERROR] {e}"
            for row in batch:
                results.append({
                    "id": row["id"],
                    "result_cls": None,
                    "raw_output": err,
                })
                raw_full_jsonl.append({
                    "id": row["id"],
                    "raw_output_full": err,
                })
                bad_ids.append(row["id"])

    return results, bad_ids, raw_full_jsonl

def main():
    # 初始化 vLLM —— 保持原样
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=False,
        seed=SEED,

        # AWQ: 通常 FP16
        quantization="awq",   # 或 "awq_marlin"
        dtype="float16",

        # 禁用多模态
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )

    # 读入数据
    df = pd.read_csv(INPUT_CSV_PATH)
    need_cols = {"id", "json_report"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"输入 CSV 需要包含列：{need_cols}")
    reports = df.to_dict(orient="records")

    system_message = load_system_message(PROMPT_FILE_PATH)

    # 执行
    results, bad_ids, raw_full_jsonl = batch_process_reports(
        llm=llm,
        sampling_params=sampling_params,
        system_message=system_message,
        reports=reports,
        batch_size=BATCH_SIZE
    )

    # 落盘：CSV
    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(results)

    # 结果列用可空整数，避免 1.0/2.0/3.0
    if "result_cls" in df_out.columns:
        df_out["result_cls"] = df_out["result_cls"].astype("Int64")

    df_out.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        encoding="utf-8"
    )
    print("✅ 所有报告处理完成，结果已保存：", OUTPUT_CSV_PATH)

    # 无效标签清单
    bad_dir = os.path.dirname(INVALID_LABELS_PATH)
    if bad_dir:
        os.makedirs(bad_dir, exist_ok=True)

    pd.DataFrame({"bad_id": bad_ids}).to_csv(INVALID_LABELS_PATH, index=False)
    print(f"⚠️ 无效标签条数：{len(bad_ids)}，清单已保存：{INVALID_LABELS_PATH}")

if __name__ == "__main__":
    main()