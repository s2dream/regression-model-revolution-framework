#!/bin/bash
# =====================================================================
# 🚀 AutoML Regression Framework Runner - Web UI
# =====================================================================

# Resolve the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================================"
echo "🏃 Starting Enterprise AutoML Web Studio Server..."
echo "🌐 Open http://localhost:8501 in your browser"
echo "========================================================"

cd "$PROJECT_ROOT"
python server.py
