#!/bin/bash
#SBATCH --job-name=cls_gemma_27B
#SBATCH --output=logs/cls_gemma_%j.out
#SBATCH --error=logs/cls_gemma_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1          # AWQ-INT4 quantization fits in ~14GB; 1x A100-40G is sufficient
#SBATCH --partition=gpu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/mri-report-llm-classification"
SCRIPT="$REPO_DIR/scripts/inference/classification/run_classification_gemma_27B.py"

MODEL="path/to/model/gemma-3-27b-it-int4-awq"
IN_CSV="$REPO_DIR/data/example/example_json_report"
PROMPT="$REPO_DIR/prompts/prompt_mri_classification.txt"
OUT_CSV="path/to/output/classification_gemma_27B.csv"
INVALID_CSV="path/to/output/invalid_gemma_27B.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Runtime overrides (optional; defaults are set inside the Python script)
export BATCH_SIZE=8
export TP_SIZE=1

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
