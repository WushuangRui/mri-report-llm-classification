#!/bin/bash
#SBATCH --job-name=cls_yi_9B
#SBATCH --output=logs/cls_yi_%j.out
#SBATCH --error=logs/cls_yi_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1          # 9B model fits comfortably in 1x A100-40G
#SBATCH --partition=gpu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/mri-report-llm-classification"
SCRIPT="$REPO_DIR/scripts/inference/classification/run_classification_yi-1.5_9B.py"

MODEL="path/to/model/Yi-1.5-9B-Chat-16K"
IN_CSV="$REPO_DIR/data/example/example_json_report"
PROMPT="$REPO_DIR/prompts/prompt_mri_classification.txt"
OUT_CSV="path/to/output/classification_yi_9B.csv"
INVALID_CSV="path/to/output/invalid_yi_9B.csv"
# ─────────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR" || exit 1
source .venv/bin/activate

mkdir -p logs "$(dirname "$OUT_CSV")"

python "$SCRIPT" \
    --model       "$MODEL" \
    --in_csv      "$IN_CSV" \
    --prompt_path "$PROMPT" \
    --out_csv     "$OUT_CSV" \
    --invalid_csv "$INVALID_CSV"

echo "[INFO] Job finished: $(date)"
