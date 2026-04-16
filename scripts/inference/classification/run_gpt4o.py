"""
run_gpt4o.py
-------------
Classify brain MRI reports using GPT-4o via the NYU Kong API gateway.
Valid output labels: 1, 2, 3

Results are written row-by-row so the job can be safely interrupted and resumed.

Usage:
    export AZURE_OPENAI_API_KEY=<your-key>
    python run_gpt4o.py \
        --in_csv      /path/to/input.csv \
        --prompt_path /path/to/prompt.txt \
        --out_csv     /path/to/output.csv
"""

import os
import re
import time
import argparse
import pandas as pd
from openai import OpenAI

QUIET = True
VALID_CLASSES = {1, 2, 3}


def structured_output_llm(messages, model_version="gpt-4o/v1.0.0", temperature=0.2, top_p=1.0):
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not set")
    client = OpenAI(
        base_url=f"https://kong-api.prod1.nyumc.org/{model_version}",
        api_key=api_key,
        default_headers={"api-key": api_key},
    )
    return client.chat.completions.create(
        model="anything", messages=messages, temperature=temperature, top_p=top_p)


def parse_args():
    parser = argparse.ArgumentParser(description="MRI report classification with GPT-4o")
    parser.add_argument("--in_csv",      required=True, help="Input CSV path")
    parser.add_argument("--prompt_path", required=True, help="Prompt txt path")
    parser.add_argument("--out_csv",     required=True, help="Output CSV path")
    parser.add_argument("--max_retries", type=int,   default=4,   help="Max retries per request")
    parser.add_argument("--base_sleep",  type=float, default=2.0, help="Initial retry sleep seconds")
    return parser.parse_args()


def load_system_message(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks to avoid capturing chain-of-thought digits."""
    if text is None:
        return ""
    t = str(text)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", t, flags=re.IGNORECASE):
        t = re.split(r"</think>", t, maxsplit=1, flags=re.IGNORECASE)[-1]
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


def process_report(system_message: str, report_id: str, report_input: str) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": f"ID: {report_id}\ninput: {report_input}"},
    ]
    raw = structured_output_llm(messages=messages, model_version="gpt-4o/v1.0.0",
                                 temperature=0.2, top_p=1.0)
    if not raw or not getattr(raw, "choices", None):
        raise RuntimeError(f"Empty response object for ID={report_id}")
    content = raw.choices[0].message.content or ""
    if not str(content).strip():
        raise RuntimeError(f"Empty message.content for ID={report_id}")
    return content


def process_report_with_retry(system_message, report_id, report_input,
                               max_retries=4, base_sleep=2.0) -> str:
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
    """Strip BOM / whitespace from column names; rename mid->id and json_report->input."""
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    rename_map = {}
    if "mid" in df.columns and "id" not in df.columns:
        rename_map["mid"] = "id"
    if "json_report" in df.columns and "input" not in df.columns:
        rename_map["json_report"] = "input"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_done_ids(out_csv: str) -> set:
    """Return IDs already successfully classified (result_cls in {1,2,3})."""
    if not os.path.exists(out_csv):
        return set()
    try:
        old = normalize_columns(pd.read_csv(out_csv, dtype={"id": str}))
        if "id" not in old.columns:
            print(f"[WARN] Existing output has no 'id' column: {out_csv}", flush=True)
            return set()
        old["id"] = old["id"].fillna("").astype(str).str.strip()
        old = old[old["id"] != ""]
        if "result_cls" in old.columns:
            old["result_cls"] = pd.to_numeric(old["result_cls"], errors="coerce")
            return set(old.loc[old["result_cls"].isin(list(VALID_CLASSES)), "id"])
        return set()
    except Exception as e:
        print(f"[WARN] Failed to read existing output: {e}", flush=True)
        return set()


def append_result_row(out_csv: str, row_dict: dict):
    """Write a single result row immediately to disk (resume-safe)."""
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    expected_columns = ["id", "result_cls", "raw_output"]
    row_df = pd.DataFrame([row_dict], columns=expected_columns)
    file_exists = os.path.exists(out_csv)
    if file_exists:
        old_header = pd.read_csv(out_csv, nrows=0).columns.tolist()
        if old_header != expected_columns:
            raise ValueError(
                f"Output file header mismatch.\n"
                f"Expected: {expected_columns}\nFound: {old_header}\n"
                f"Delete the old output file or use a new --out_csv path.")
    row_df.to_csv(out_csv, mode="a", header=not file_exists, index=False, encoding="utf-8")


def rebuild_invalid_file(out_csv: str):
    """Write a sidecar .invalid_ids.csv listing rows with no valid classification."""
    if not os.path.exists(out_csv):
        return
    out_df = normalize_columns(pd.read_csv(out_csv, dtype={"id": str}))
    if "result_cls" not in out_df.columns:
        return
    out_df["id"] = out_df["id"].fillna("").astype(str).str.strip()
    out_df["result_cls"] = pd.to_numeric(out_df["result_cls"], errors="coerce")
    invalid = out_df[~out_df["result_cls"].isin(list(VALID_CLASSES))]
    if len(invalid) > 0:
        invalid_path = out_csv.replace(".csv", ".invalid_ids.csv")
        cols = [c for c in ["id", "raw_output"] if c in invalid.columns]
        invalid[cols].to_csv(invalid_path, index=False, encoding="utf-8")
        print(f"[INFO] Invalid outputs: {len(invalid)}, saved to {invalid_path}", flush=True)
    else:
        print("[INFO] No invalid outputs found.", flush=True)


def main():
    args = parse_args()
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

    df = normalize_columns(pd.read_csv(args.in_csv, dtype=str, encoding="utf-8-sig"))
    if "input" not in df.columns:
        raise KeyError(f"'input' column not found. Available columns: {list(df.columns)}")
    if "id" not in df.columns:
        raise KeyError(f"'id' column not found. Available columns: {list(df.columns)}")

    df["id"]    = df["id"].fillna("").astype(str).str.strip()
    df["input"] = df["input"].fillna("").astype(str)
    df = df[df["id"] != ""].copy()

    system_message = load_system_message(args.prompt_path)
    done_ids       = load_done_ids(args.out_csv)
    remaining_df   = df[~df["id"].isin(done_ids)].copy()
    processed_now  = 0

    for _, row in remaining_df.iterrows():
        report_id, report_input = row["id"], row["input"]
        try:
            raw_out = process_report_with_retry(
                system_message, report_id, report_input,
                args.max_retries, args.base_sleep)
            result_row = {"id": report_id, "result_cls": extract_label(raw_out), "raw_output": raw_out}
        except Exception as e:
            result_row = {"id": report_id, "result_cls": None,
                          "raw_output": f"[ERROR] {type(e).__name__}: {e}"}
            print(f"[ERROR] ID={report_id} failed after retries: {e}", flush=True)

        append_result_row(args.out_csv, result_row)
        processed_now += 1
        print(f"{processed_now}/{len(remaining_df)} ID={report_id}", flush=True)

    print(f"[INFO] Run finished. Saved to {args.out_csv}", flush=True)
    rebuild_invalid_file(args.out_csv)


if __name__ == "__main__":
    main()
