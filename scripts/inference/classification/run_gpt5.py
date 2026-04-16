import os
import re
import sys
import time
import argparse
import pandas as pd

from openai import OpenAI

QUIET = True
VALID_CLASSES = {1, 2, 3}


def structured_output_llm(
    messages,
    model_version: str = "gpt-5/v1.0.0",
    temperature: float = 0.2,
    top_p: float = 1.0
):
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

    endpoint = f"https://kong-api.prod1.nyumc.org/{model_version}"

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
        default_headers={"api-key": api_key},
        timeout=120.0,
    )

    is_gpt5 = "gpt-5" in model_version

    try:
        if is_gpt5:
            print(f"[INFO] Using GPT-5 via chat.completions: {model_version}", flush=True)
            resp = client.chat.completions.create(
                model="anything",
                messages=messages,
                top_p=top_p,
                max_completion_tokens=512,
                # 如果你们接口支持，可以打开这一行：
                # reasoning={"effort": "low"},
            )

            # 关键：把“length + 空内容”当成失败
            choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
            msg = getattr(choice0, "message", None) if choice0 else None
            content = getattr(msg, "content", None) if msg else None
            finish_reason = getattr(choice0, "finish_reason", None) if choice0 else None

            if finish_reason == "length" and (content is None or str(content).strip() == ""):
                raise RuntimeError("GPT-5 returned empty output due to finish_reason='length'")

            return resp

        print(f"[INFO] Using GPT-4o-style: {model_version}", flush=True)
        return client.chat.completions.create(
            model="anything",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=128,
        )

    except Exception as e:
        if is_gpt5:
            print(f"[WARN] GPT-5 failed → fallback GPT-4o: {e}", flush=True)

            fallback_client = OpenAI(
                base_url="https://kong-api.prod1.nyumc.org/gpt-4o/v1.0.0",
                api_key=api_key,
                default_headers={"api-key": api_key},
                timeout=120.0,
            )

            return fallback_client.chat.completions.create(
                model="anything",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=128,
            )

        raise


def parse_args():
    parser = argparse.ArgumentParser(description="MRI report classification with GPT via api_manager5.")
    parser.add_argument("--in_csv", required=True, help="Input CSV path")
    parser.add_argument("--prompt_path", required=True, help="Prompt txt path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--max_retries", type=int, default=4, help="Max retries per request")
    parser.add_argument("--base_sleep", type=float, default=2.0, help="Initial retry sleep seconds")
    return parser.parse_args()


def load_system_message(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def strip_think(text: str) -> str:
    """
    移除 <think>...</think>，避免误抓数字。
    """
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", t, flags=re.IGNORECASE):
        t = re.split(r"</think>", t, maxsplit=1, flags=re.IGNORECASE)[-1]
    return t.strip()


def extract_class(text: str):
    """
    只接受 1 / 2 / 3，返回 int 或 None。
    假设 prompt 已明确要求模型只输出一个分类数字。
    """
    t = strip_think(text)
    matches = re.findall(r"(?<!\d)([1-3])(?!\d)", t)
    if matches:
        return int(matches[-1])
    return None


def process_report(system_message: str, report_id: str, report_input: str) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"ID: {report_id}\ninput: {report_input}"},
    ]

    raw_response = structured_output_llm(messages=messages)

    if not raw_response or not getattr(raw_response, "choices", None):
        raise RuntimeError(f"Empty response object for ID={report_id}")

    message = raw_response.choices[0].message
    content = message.content if message and getattr(message, "content", None) is not None else ""

    if str(content).strip() == "":
        raise RuntimeError(f"Empty message.content for ID={report_id}")

    return content


def process_report_with_retry(
    system_message: str,
    report_id: str,
    report_input: str,
    max_retries: int = 4,
    base_sleep: float = 2.0
) -> str:
    """
    单条请求自动重试。
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            return process_report(system_message, report_id, report_input)
        except Exception as e:
            last_err = e
            wait_s = base_sleep * (2 ** (attempt - 1))
            print(f"[WARN] ID={report_id} attempt {attempt}/{max_retries} failed: {e}", flush=True)

            if attempt < max_retries:
                print(f"[WARN] sleeping {wait_s:.1f}s then retry...", flush=True)
                time.sleep(wait_s)

    raise last_err


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗列名，兼容 BOM / 空格 / mid -> id / json_report -> input
    """
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    rename_map = {}
    if "mid" in df.columns and "id" not in df.columns:
        rename_map["mid"] = "id"
    if "json_report" in df.columns and "input" not in df.columns:
        rename_map["json_report"] = "input"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_done_ids(out_csv: str) -> set:
    """
    从已有输出中读取已经成功完成的 id。
    只有 result_cls 为 1/2/3 的才算真正完成。
    """
    if not os.path.exists(out_csv):
        return set()

    try:
        old = pd.read_csv(out_csv, dtype={"id": str})
        old = normalize_columns(old)

        if "id" not in old.columns:
            print(f"[WARN] Existing output file has no 'id' column: {out_csv}", flush=True)
            return set()

        old["id"] = old["id"].fillna("").astype(str).str.strip()
        old = old[old["id"] != ""].copy()

        if "result_cls" in old.columns:
            old["result_cls"] = pd.to_numeric(old["result_cls"], errors="coerce")
            success_mask = old["result_cls"].isin([1, 2, 3])
            done_ids = set(old.loc[success_mask, "id"].tolist())
        else:
            done_ids = set(old["id"].tolist())

        if not QUIET:
            print(f"Found existing output with {len(done_ids)} successful IDs.", flush=True)
        return done_ids

    except Exception as e:
        print(f"[WARN] Failed to read existing output file: {e}", flush=True)
        return set()


def append_result_row(out_csv: str, row_dict: dict):
    """
    单条结果立刻写盘。不存在则写表头，存在则追加。
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row_df = pd.DataFrame([row_dict])
    file_exists = os.path.exists(out_csv)

    row_df.to_csv(
        out_csv,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )


def rebuild_invalid_file(out_csv: str):
    """
    跑完后根据总输出重建 invalid 文件。
    """
    if not os.path.exists(out_csv):
        return

    out_df = pd.read_csv(out_csv, dtype={"id": str})
    out_df = normalize_columns(out_df)

    if "result_cls" not in out_df.columns:
        return

    out_df["id"] = out_df["id"].fillna("").astype(str).str.strip()
    out_df["result_cls"] = pd.to_numeric(out_df["result_cls"], errors="coerce")

    invalid = out_df[~out_df["result_cls"].isin([1, 2, 3])]

    if len(invalid) > 0:
        invalid_path = out_csv.replace(".csv", ".invalid_ids.csv")
        cols_to_save = [c for c in ["id", "raw_output"] if c in invalid.columns]
        invalid[cols_to_save].to_csv(invalid_path, index=False, encoding="utf-8")
        print(f"Invalid outputs: {len(invalid)}; saved: {invalid_path}", flush=True)
    else:
        print("No invalid outputs found.", flush=True)


def main():
    args = parse_args()

    if not QUIET:
        print("[INFO] main() started", flush=True)

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

    if not QUIET:
        print("[INFO] API key found", flush=True)
        print(f"[INFO] Reading input CSV: {args.in_csv}", flush=True)

    df = pd.read_csv(args.in_csv, dtype=str, encoding="utf-8-sig")

    if not QUIET:
        print(f"[INFO] Input CSV loaded. shape={df.shape}", flush=True)

    df = normalize_columns(df)

    if not QUIET:
        print(f"[INFO] Columns after normalize: {list(df.columns)}", flush=True)

    if "input" not in df.columns:
        raise KeyError(f"找不到 input 列。现有列：{list(df.columns)}")
    if "id" not in df.columns:
        raise KeyError(f"找不到 id 列。现有列：{list(df.columns)}")

    df["id"] = df["id"].fillna("").astype(str).str.strip()
    df["input"] = df["input"].fillna("").astype(str)
    df = df[df["id"] != ""].copy()

    if not QUIET:
        print(f"[INFO] Non-empty id rows: {len(df)}", flush=True)
        print(f"[INFO] Loading prompt: {args.prompt_path}", flush=True)

    system_message = load_system_message(args.prompt_path)

    if not QUIET:
        print(f"[INFO] Prompt loaded. length={len(system_message)}", flush=True)
        print(f"[INFO] Loading done IDs from: {args.out_csv}", flush=True)

    done_ids = load_done_ids(args.out_csv)

    if not QUIET:
        print(f"[INFO] Loaded done IDs: {len(done_ids)}", flush=True)

    total = len(df)
    remaining_df = df[~df["id"].isin(done_ids)].copy()

    if not QUIET:
        print(f"Input CSV: {args.in_csv}", flush=True)
        print(f"Prompt path: {args.prompt_path}", flush=True)
        print(f"Output CSV: {args.out_csv}", flush=True)
        print(f"Total rows in input: {total}", flush=True)
        print(f"Already done: {len(done_ids)}", flush=True)
        print(f"Remaining to process: {len(remaining_df)}", flush=True)

    processed_now = 0

    for _, row in remaining_df.iterrows():
        report_id = row["id"]
        report_input = row["input"]

        try:
            raw_out = process_report_with_retry(
                system_message=system_message,
                report_id=report_id,
                report_input=report_input,
                max_retries=args.max_retries,
                base_sleep=args.base_sleep
            )

            cls = extract_class(raw_out)

            result_row = {
                "id": report_id,
                "result_cls": cls,
                "raw_output": raw_out,
            }

        except Exception as e:
            result_row = {
                "id": report_id,
                "result_cls": None,
                "raw_output": f"[ERROR] {type(e).__name__}: {e}",
            }
            print(f"[ERROR] ID={report_id} failed after retries: {e}", flush=True)

        append_result_row(args.out_csv, result_row)
        processed_now += 1

        print(f"{processed_now}/{len(remaining_df)} ID={report_id}", flush=True)

    print(f"Run finished. Saved: {args.out_csv}", flush=True)

    rebuild_invalid_file(args.out_csv)


if __name__ == "__main__":
    main()
