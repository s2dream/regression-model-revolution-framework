#!/bin/bash
# =====================================================================
# 🚀 AutoML Regression Framework Runner - Direct URL Dataset
# =====================================================================

# Resolve the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================================"
echo "🏃 Running AutoML Pipeline with remote URL dataset..."
echo "========================================================"

python "$PROJECT_ROOT/main.py" \
  --config configs/default.yml \
  --url "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv" \
  --target medv \
  --turn 1

echo "========================================================"
echo "✅ Pipeline run completed!"
echo "========================================================"
