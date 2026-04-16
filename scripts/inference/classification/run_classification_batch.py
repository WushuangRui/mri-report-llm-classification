import csv
import pandas as pd
import re
from tqdm import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from vllm import LLM, SamplingParams

def load_system_message(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def build_prompt(system_message, report_id, report_input):
    return [
        {"role": "system", "content":  "You are a helpful assistant"},
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
The left mesial precentral gyrus cavernoma shows ill-defined T2/FLAIR hyperintense and T1-isotense signal extending inferolaterally, which may represent edema. There is mild interval increase in size of the lesion, currently measuring 1.5 x 1.3 cm, previously 1.3 x 1.0 cm. The lesion remains heterogenous centrally, demonstrating T1 and T2 hyperintensity , a T1-isointense/T2-hypointense rim, and susceptibility. Mild associated gyral expansion is similar to prior MRI. There is no midline shift. These findings are most compatible cavernoma which has subacute hemorrhage with new mild edema in the adjacent parenchyma. Linear T1 isointense focus just lateral to the cavernoma is compatible with a developmental venous anomaly, seen on the postcontrast images of the prior exam. The ventricular system is of normal size and configuration. There is no midline shift or herniation pattern. No extra axial collections are identified. Normal flow voids of the major intracranial vessels are seen on T2 weighted spin echo imaging. There is no evidence of a diffusion abnormality to suggest acute or subacute infarction. Small right maxillary retention cyst. The paranasal sinuses, mastoid air cells and orbits are unremarkable.

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

    # Step 1: 查找 </think> 后的第一个数字
    think_split = text.split('</think>')
    if len(think_split) > 1:
        post_think = think_split[1]
        match = re.search(r'\d+', post_think)
        if match:
            return int(match.group())

    # Step 2: 查找关键词后最多5个非数字字符后跟的第一个数字
    match = re.search(r"(answer|select|choice|prediction|category)[^\d]{0,5}(\d)", text)
    if match:
        return int(match.group(2))

    # Step 3: 提取所有数字，返回最后一个
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
                if label is None or label not in [1, 2, 3, 4]:
                    print(f"⚠️ Invalid label (not in 1–4): {batch[j]['id']} — raw: {result_text}")
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
    # ✅ 手动构建模型参数（不使用 argparse 和 EngineArgs）
    llm = LLM(
        model="/gpfs/data/shenlab/LLMs/DeepSeek/distilled_model/DeepSeek-R1-Distill-Qwen-32B",
        max_model_len=8192,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        top_k=40
    )

    # 读取数据和 prompt
    df = pd.read_csv("/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv")
    reports = df.to_dict(orient="records")

    system_message = load_system_message("/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/prompt&result/prompt_classification_14.txt")

    # 执行推理
    results, bad_ids = batch_process_reports(llm, sampling_params, system_message, reports, batch_size=8)

    # 保存结果
    pd.DataFrame(results).to_csv("processed_reports_classification_ds_14_680.csv", index=False)
    print("✅ 所有报告处理完成，结果已保存。")

    pd.DataFrame({"bad_id": bad_ids}).to_csv("invalid_labels.csv", index=False)


if __name__ == "__main__":
    main()
