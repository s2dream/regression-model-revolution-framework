import pytest
import os
import re

def test_ui_tooltips_help_docs_and_markup_integrity():
    """Verifies that all options have comprehensive help documentation and matching UI markup triggers."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Read app.js and inspect HELP_DOCS
    js_path = os.path.join(project_root, "web", "app.js")
    assert os.path.exists(js_path)
    with open(js_path, "r", encoding="utf-8") as f:
        js_code = f.read()

    assert "HELP_DOCS" in js_code
    assert "initTooltipSystem" in js_code
    assert "showTooltip" in js_code
    assert "globalTooltipPopover" in js_code

    # Check key categories in HELP_DOCS
    required_keys = [
        "data_dir", "output_dir", "data_source", "dataset_select", "target_column", "ignored_columns",
        "split_method", "test_size", "n_splits",
        "hpo_enabled", "hpo_trials",
        "shap_enabled", "shap_model", "shap_max_samples",
        "run_id", "overwrite_run",
        "XGBoost.n_estimators", "XGBoost.learning_rate", "XGBoost.max_depth",
        "CatBoost.iterations", "CatBoost.depth",
        "RandomForest.n_estimators", "RandomForest.max_depth",
        "MLP.hidden_layer_sizes", "MLP.activation", "MLP.solver",
        "TabPFN.N_ensemble_configurations",
        "TabICL.n_estimators", "TabICL.device",
        "Transformer.epochs", "Transformer.d_model", "Transformer.nhead"
    ]

    for key in required_keys:
        assert f"'{key}':" in js_code or f'"{key}":' in js_code, f"Missing documentation for key: {key}"

    # 2. Read index.html and verify static info-tip elements
    html_path = os.path.join(project_root, "web", "index.html")
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()

    static_tips = [
        "data_dir", "output_dir", "data_source", "dataset_select", "target_column", "ignored_columns",
        "split_method", "test_size", "n_splits", "hpo_enabled", "hpo_trials",
        "shap_enabled", "shap_model", "shap_max_samples", "run_id", "overwrite_run"
    ]

    for tip in static_tips:
        pattern = f'data-tip="{tip}"'
        assert pattern in html_code, f"Missing info-tip trigger in index.html for: {tip}"

    # 3. Read style.css and verify tooltip popper styling
    css_path = os.path.join(project_root, "web", "style.css")
    assert os.path.exists(css_path)
    with open(css_path, "r", encoding="utf-8") as f:
        css_code = f.read()

    assert ".info-tip" in css_code
    assert ".tooltip-popover" in css_code
    assert ".tooltip-header" in css_code
    assert ".tooltip-rec" in css_code
    assert ".tooltip-warn" in css_code
