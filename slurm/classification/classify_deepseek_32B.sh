#!/bin/bash
#SBATCH --job-name=cls_deepseek_32B
#SBATCH --output=logs/cls_deepseek_%j.out
#SBATCH --error=logs/cls_deepseek_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2          # 32B model needs ~60GB VRAM; use 2x A100-40G or 1x A100-80G
#SBATCH --partition=gpu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/mri-report-llm-classification"
SCRIPT="$REPO_DIR/scripts/inference/classification/run_classification_deepseek_32B.py"

MODEL="/gpfs/data/shenlab/LLMs/DeepSeek/distilled_model/DeepSeek-R1-Distill-Qwen-32B"
IN_CSV="/gpfs/home/wr2215/ms_mri_deepseek/mri_classification/processed_json_test_subset680.csv"
PROMPT="$REPO_DIR/prompts/prompt_classification_14.txt"
OUT_CSV="/gpfs/home/wr2215/ms_mri_deepseek/results/classification_deepseek_32B.csv"
INVALID_CSV="/gpfs/home/wr2215/ms_mri_deepseek/results/invalid_deepseek_32B.csv"
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
