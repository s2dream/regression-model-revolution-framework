import pytest
import os
import json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from server import app
from automl_framework.model.model_factory import ModelFactory
from automl_framework.model.model_pool import ModelPool

client = TestClient(app)


def test_active_model_selection_and_options_api():
    """Verifies that active model selection and custom hyperparameter options are correctly saved and loaded via API."""
    custom_model_config = {
        "framework": {
            "random_state": 42,
            "active_models": ["XGBoost", "CatBoost", "RandomForest", "MLP", "Transformer"]
        },
        "data": {
            "data_dir": "data",
            "output_dir": "outputs",
            "target_column": "Target_Y",
            "split": {"method": "train_test_split", "test_size": 0.25}
        },
        "models": {
            "XGBoost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 1},
            "CatBoost": {"iterations": 40, "learning_rate": 0.08, "depth": 4, "verbose": 0},
            "RandomForest": {"n_estimators": 60, "max_depth": 5, "n_jobs": 1},
            "MLP": {"hidden_layer_sizes": [64, 32], "activation": "relu", "max_iter": 100},
            "Transformer": {"epochs": 5, "d_model": 16, "nhead": 2, "num_layers": 1, "batch_size": 8},
            "TabPFN": {"N_ensemble_configurations": 4},
            "TabICL": {"n_estimators": 4, "device": "cpu", "batch_size": 2}
        },
        "hpo": {"enabled": False, "n_trials": 10},
        "shap": {"enabled": True, "model": "XGBoost", "max_samples": 50}
    }

    # 1. Save configuration via POST /api/config
    save_res = client.post("/api/config", json={"config": custom_model_config})
    assert save_res.status_code == 200
    assert save_res.json()["status"] == "success"

    # 2. Retrieve configuration via GET /api/config
    get_res = client.get("/api/config")
    assert get_res.status_code == 200
    retrieved = get_res.json()
    assert "active" in retrieved
    active_cfg = retrieved["active"]

    # 3. Assert active model selections match
    assert active_cfg["framework"]["active_models"] == ["XGBoost", "CatBoost", "RandomForest", "MLP", "Transformer"]
    
    # 4. Assert hyperparameter values match across types (int, float, list, string)
    assert active_cfg["models"]["XGBoost"]["n_estimators"] == 50
    assert active_cfg["models"]["XGBoost"]["learning_rate"] == 0.05
    assert active_cfg["models"]["CatBoost"]["iterations"] == 40
    assert active_cfg["models"]["RandomForest"]["max_depth"] == 5
    assert active_cfg["models"]["MLP"]["hidden_layer_sizes"] == [64, 32]
    assert active_cfg["models"]["Transformer"]["epochs"] == 5


def test_model_pool_initializes_selected_active_models_with_custom_options():
    """Verifies that ModelPool instantiates only active models and applies custom parameter options."""
    test_config = {
        "framework": {
            "random_state": 42,
            "active_models": ["XGBoost", "RandomForest", "MLP"]
        },
        "models": {
            "XGBoost": {"n_estimators": 10, "max_depth": 3},
            "RandomForest": {"n_estimators": 10, "max_depth": 4},
            "MLP": {"hidden_layer_sizes": [32, 16], "max_iter": 20},
            "CatBoost": {"iterations": 10} # Inactive, should not be in pool
        }
    }

    pool = ModelPool(random_state=42, config=test_config)
    available_models = pool.list_available_models()

    # Active models should be present, inactive models should not
    assert "XGBoost" in available_models
    assert "RandomForest" in available_models
    assert "MLP" in available_models
    assert "CatBoost" not in available_models

    # Verify custom hyperparameter applied to model instances
    xgb_wrapper = pool.get_model("XGBoost")
    assert xgb_wrapper.model.n_estimators == 10
    assert xgb_wrapper.model.max_depth == 3

    rf_wrapper = pool.get_model("RandomForest")
    assert rf_wrapper.model.n_estimators == 10
    assert rf_wrapper.model.max_depth == 4


def test_web_ui_has_model_pool_and_accordion_components():
    """Verifies that web static assets (index.html, app.js, style.css) contain required model option selectors and accordions."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    html_path = os.path.join(project_root, "web", "index.html")
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    assert 'id="modelSelectionGrid"' in html_content
    assert 'id="modelParamsAccordion"' in html_content

    js_path = os.path.join(project_root, "web", "app.js")
    assert os.path.exists(js_path)
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    assert 'DEFAULT_MODEL_TEMPLATES' in js_content
    assert 'renderModelPool' in js_content
    assert 'model-param-input' in js_content
    assert 'active-model' in js_content
