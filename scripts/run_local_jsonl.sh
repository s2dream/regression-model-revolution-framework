#!/bin/bash
# =====================================================================
# 🚀 AutoML Regression Framework Runner - Local JSONL Dataset
# =====================================================================

# Resolve the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================================"
echo "🏃 Running AutoML Pipeline with Local JSONL dataset..."
echo "========================================================"

python "$PROJECT_ROOT/main.py" \
  --config configs/default.yml \
  --dataset-path data/synthetic_regression.jsonl \
  --target Target_Y \
  --turn 1

echo "========================================================"
echo "✅ Pipeline run completed!"
echo "========================================================"
