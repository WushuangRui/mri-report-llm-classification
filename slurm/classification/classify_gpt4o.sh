#!/bin/bash
#SBATCH --job-name=cls_gpt4o
#SBATCH --output=logs/cls_gpt4o_%j.out
#SBATCH --error=logs/cls_gpt4o_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu             # API-based; no GPU needed

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/mri-report-llm-classification"
SCRIPT="$REPO_DIR/scripts/inference/classification/run_gpt4o.py"

IN_CSV="$REPO_DIR/data/example/example_json_report"
PROMPT="$REPO_DIR/prompts/prompt_mri_classification.txt"
OUT_CSV="path/to/output/classification_gpt4o.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Required: set your API key here or export it before submitting
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:?AZURE_OPENAI_API_KEY is not set}"

cd "$REPO_DIR" || exit 1
source .venv/bin/activate

mkdir -p logs "$(dirname "$OUT_CSV")"

python "$SCRIPT" \
    --in_csv      "$IN_CSV" \
    --prompt_path "$PROMPT" \
    --out_csv     "$OUT_CSV" \
    --max_retries 4 \
    --base_sleep  2.0

echo "[INFO] Job finished: $(date)"
