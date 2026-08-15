#!/bin/bash
# =====================================================================
# 🚀 AutoML Regression Framework Runner - Web UI
# =====================================================================

# Resolve the absolute path of the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================================"
echo "🏃 Starting Streamlit WebUI Studio (Headless Mode)..."
echo "========================================================"

streamlit run "$PROJECT_ROOT/app.py" --server.headless true
