#!/bin/bash
# Full experiment: Qwen3.5-4B (bf16) and Qwen3.5-9B (4-bit NF4) x the 2023-2024 and 2026 datasets on one GPU,
# then local Qwen3Guard judging and a results summary. Models are downloaded from Hugging Face on first use.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs

echo "=== PHASE 1: Qwen3.5-4B $(date +%H:%M)"
./run_model.sh Qwen/Qwen3.5-4B 16

echo "=== PHASE 2: Qwen3.5-9B NF4 $(date +%H:%M)"
./run_model.sh Qwen/Qwen3.5-9B 16 --load_in_4bit

echo "=== PHASE 3: Qwen3Guard judging + summary $(date +%H:%M)"
.venv/bin/python -m pipeline.judge_all > logs/judge.log 2>&1 && echo "=== JUDGE DONE $(date +%H:%M)" || echo "=== JUDGE FAILED (logs/judge.log)"
.venv/bin/python -m pipeline.summarize > logs/summary.log 2>&1 && echo "=== SUMMARY DONE" || echo "=== SUMMARY FAILED"
echo "=== ALL FINISHED $(date +%H:%M)"
