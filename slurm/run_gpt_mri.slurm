#!/bin/bash
#SBATCH --job-name=mri_gpt4o
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11   # adjust to your HPC module name

REPO_DIR="$HOME/mri-report-llm-classification"
cd "$REPO_DIR" || exit 1

source .venv/bin/activate   # activate your virtual environment

# ── Required secrets (set these in your shell or via HPC secret manager) ─────
# export AZURE_OPENAI_API_KEY=<your-key>
# export AZURE_OPENAI_ENDPOINT=<your-endpoint>
# export AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ── Parameters ────────────────────────────────────────────────────────────────
INPUT="data/example_input.csv"
PROMPT="prompts/prompt_classification_14.txt"
OUTPUT="outputs/results_$(date +%Y%m%d_%H%M%S).csv"
DEPLOYMENT="gpt-4o"

mkdir -p outputs logs

# ── Run ───────────────────────────────────────────────────────────────────────
python scripts/run_gpt4o.py \
    --input      "$INPUT" \
    --prompt     "$PROMPT" \
    --output     "$OUTPUT" \
    --deployment "$DEPLOYMENT"

echo "Done. Results written to $OUTPUT"
