#!/bin/bash

# Configuration
PENALTIES=("fgan_js" "fgan_sqHellinger")
ANNEAL_OPTIONS=(0.0 1.0)
RESULT_DIR="results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULT_DIR"

for anneal in "${ANNEAL_OPTIONS[@]}"; do
    for penalty in "${PENALTIES[@]}"; do
        python main.py \
            --penalty "$penalty" \
            --anneal "$anneal" \
            --current "$RESULT_DIR/"
    done
done