#!/bin/bash
# =====================================================================
# 🚀 AutoML Regression Framework Runner - Local CSV Dataset
# =====================================================================

# Resolve the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================================"
echo "🏃 Running AutoML Pipeline with Local CSV dataset..."
echo "========================================================"

python "$PROJECT_ROOT/main.py" \
  --config configs/default.yml \
  --dataset-path data/synthetic_regression.csv \
  --target Target_Y \
  --turn 1

echo "========================================================"
echo "✅ Pipeline run completed!"
echo "========================================================"
