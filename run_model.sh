#!/bin/bash
# usage: run_model.sh <hf_model_path> <gen_batch_size> [--load_in_4bit]
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs
model=$1; bs=$2; extra=${3:-}
for version in 2023-2024 2026; do
    tag="$(basename "$model")${extra:+-nf4}_${version}"
    echo "=== START $tag $(date +%H:%M)"
    python -m pipeline.run_pipeline --model_path "$model" --dataset_version "$version" --gen_batch_size "$bs" $extra > "logs/$tag.log" 2>&1 \
        && echo "=== DONE $tag $(date +%H:%M)" || echo "=== FAILED $tag $(date +%H:%M) (see logs/$tag.log)"
done
