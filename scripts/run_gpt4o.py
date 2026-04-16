"""
run_gpt4o.py
Batch-classify MRI reports using Azure OpenAI GPT-4o.

Usage:
    python scripts/run_gpt4o.py \
        --input  data/example_input.csv \
        --prompt prompts/prompt_classification_14.txt \
        --output outputs/results.csv \
        --deployment gpt-4o
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from api_manager import get_client, call_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs") / "run_gpt4o.log"),
    ],
)
logger = logging.getLogger(__name__)

REPORT_COLUMN = "report_text"
ID_COLUMN = "report_id"


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def classify_report(client, deployment: str, system_prompt: str, report: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": report},
    ]
    return call_api(client, deployment, messages)


def main():
    parser = argparse.ArgumentParser(description="GPT-4o MRI report classifier")
    parser.add_argument("--input",      required=True, help="Path to input CSV")
    parser.add_argument("--prompt",     required=True, help="Path to system prompt file")
    parser.add_argument("--output",     required=True, help="Path to output CSV")
    parser.add_argument("--deployment", default="gpt-4o", help="Azure deployment name")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading prompt from %s", args.prompt)
    system_prompt = load_prompt(args.prompt)

    logger.info("Loading input from %s", args.input)
    df = pd.read_csv(args.input)
    if REPORT_COLUMN not in df.columns:
        raise ValueError(f"Input CSV must contain a '{REPORT_COLUMN}' column.")

    client = get_client()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        report_id = row.get(ID_COLUMN, "N/A")
        report_text = str(row[REPORT_COLUMN])
        try:
            response = classify_report(client, args.deployment, system_prompt, report_text)
            results.append({ID_COLUMN: report_id, "llm_output": response, "status": "ok"})
        except Exception as e:
            logger.error("Failed on report %s: %s", report_id, e)
            results.append({ID_COLUMN: report_id, "llm_output": None, "status": f"error: {e}"})

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    logger.info("Saved %d results to %s", len(out_df), args.output)


if __name__ == "__main__":
    main()
