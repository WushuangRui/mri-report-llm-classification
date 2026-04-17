# MRI Report LLM Classification

This project implements an LLM-based pipeline for brain MRI report classification. 
It includes structured report parsing, ICD-10-based clinical filtering, and multi-model ensemble prediction.

## Features
- JSON-based report structuring
- Multi-LLM ensemble classification
- ICD-10 validation with clinical constraints
- Temporal filtering of diagnoses

## Usage
1. Prepare input CSV
2. Set API key:
   export AZURE_OPENAI_API_KEY=...
3. Run:
   sbatch slurm/run_gpt_mri.slurm

Stage 1: Brain MRI raw report structuring
raw report -> LLM-based JSON extraction

Stage 2: Brain MRI report classification
structured JSON -> LLM-based label prediction

Stage 3: Evaluation and downstream clinical analysis
predictions -> metrics, ensemble analysis, ICD-10 analysis
