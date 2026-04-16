#!/bin/bash
#SBATCH --job-name=mmri_json
#SBATCH --output=logs/mri_json_%j.out
#SBATCH --error=logs/mri_json_%j.err
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=32G

module purge

source ~/.bashrc
conda activate vllm

PROJECT_DIR="/path/to/your/project"
cd "${PROJECT_DIR}"

mkdir -p logs

python scripts/raw_report2json.py \
    --max-tokens 2048 \
    --temperature 0.7 \
    --top-p 0.95 \
    --top-k 40
