import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from automl_framework.model.model_pool import ModelPool
from automl_framework.model.model_factory import ModelType, ModelFactory
from automl_framework.model.hpo import run_hpo_tuning
from automl_framework.model.wrappers import ModelWrapperXGBoost, ModelWrapperTabPFN


@pytest.fixture
def hpo_dataset():
    """Generates a small regression dataset for testing tuning."""
    np.random.seed(42)
    X = pd.DataFrame({"X1": np.random.uniform(-1, 1, 20), "X2": np.random.uniform(0, 2, 20)})
    y = pd.Series(2.0 * X["X1"] + X["X2"] + np.random.normal(0, 0.05, 20))
    return X, y


@pytest.fixture
def hpo_config():
    """Generates a sample configuration dictionary with HPO enabled."""
    return {
        "framework": {"active_models": ["XGBoost", "MLP", "RandomForest", "CatBoost", "TabPFN"]},
        "hpo": {"enabled": True, "n_trials": 3},
        "models": {
            "XGBoost": {"n_estimators": 10, "max_depth": 3},
            "MLP": {"hidden_layer_sizes": [64], "max_iter": 5},
            "RandomForest": {"n_estimators": 5},
            "CatBoost": {"iterations": 5},
            "TabPFN": {"N_ensemble_configurations": 4},
        },
    }


def test_hpo_tuning_success_and_config_update(hpo_dataset, hpo_config):
    """Verifies that active models are tuned and their parameters are updated in the pool config."""
    X, y = hpo_dataset
    # Remove TabPFN for this specific test to avoid heavy imports/warning logs
    hpo_config["framework"]["active_models"] = ["XGBoost", "RandomForest"]

    pool = ModelPool(random_state=42, config=hpo_config)

    # Cache original parameters
    orig_xgb_estimators = pool.config["models"]["XGBoost"]["n_estimators"]

    run_hpo_tuning(pool, X, y)

    # Verify parameters were updated
    assert "models" in pool.config
    assert "XGBoost" in pool.config["models"]
    
    # Check that HPO changed/updated parameters based on Optuna trials
    assert pool.config["models"]["XGBoost"]["n_estimators"] != orig_xgb_estimators
    
    # Verify the wrappers in the pool are updated to use the best configurations
    xgb_wrap = pool.get_model("XGBoost")
    assert xgb_wrap.model.n_estimators == pool.config["models"]["XGBoost"]["n_estimators"]


def test_hpo_mlp_parameter_mapping(hpo_dataset, hpo_config):
    """Verifies HPO tunes MLP and maps hidden_layer_sizes to valid formats."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["MLP"]
    
    pool = ModelPool(random_state=42, config=hpo_config)
    run_hpo_tuning(pool, X, y)
    
    mlp_wrap = pool.get_model("MLP")
    # Verify hidden_layer_sizes is tuned and formatted as a tuple
    hidden_sizes = mlp_wrap.model.hidden_layer_sizes
    assert isinstance(hidden_sizes, tuple)
    assert len(hidden_sizes) > 0


def test_hpo_tabpfn_is_skipped(hpo_dataset, hpo_config):
    """Verifies TabPFN is skipped during the HPO optimization process."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["TabPFN"]
    
    pool = ModelPool(random_state=42, config=hpo_config)
    
    # Store original wrap reference
    orig_tabpfn = pool.get_model("TabPFN")
    
    run_hpo_tuning(pool, X, y)
    
    # Verify model was not replaced/modified
    assert pool.get_model("TabPFN") is orig_tabpfn


def test_hpo_tabicl_is_skipped(hpo_dataset, hpo_config):
    """Verifies TabICL is skipped during the HPO optimization process."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["TabICL"]
    hpo_config["models"]["TabICL"] = {"n_estimators": 2, "device": "cpu", "batch_size": 2}
    
    pool = ModelPool(random_state=42, config=hpo_config)
    orig_tabicl = pool.get_model("TabICL")
    
    run_hpo_tuning(pool, X, y)
    
    assert pool.get_model("TabICL") is orig_tabicl


def test_hpo_custom_and_invalid_models_skipped(hpo_dataset, hpo_config):
    """Verifies custom models and invalid model type strings are skipped gracefully."""
    X, y = hpo_dataset
    pool = ModelPool(random_state=42, config={"framework": {"active_models": []}})
    
    # Add a custom mock model
    mock_model = MagicMock()
    pool.add_custom_model("CustomEstimator", mock_model)
    
    # Set config to run HPO
    pool.config = hpo_config
    
    run_hpo_tuning(pool, X, y)
    
    # Verify custom estimator is untouched and remains in the pool
    assert pool.get_model("CustomEstimator") is not None


def test_hpo_trial_failure_handling(hpo_dataset, hpo_config, monkeypatch):
    """Verifies that if a model wrapper fails to fit during a trial, HPO handles it gracefully."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["XGBoost"]
    pool = ModelPool(random_state=42, config=hpo_config)

    # Monkeypatch the fit method of ModelWrapperXGBoost to raise an exception
    def mock_fit(self, X_t, y_t):
        raise ValueError("Simulated training crash")
        
    monkeypatch.setattr(ModelWrapperXGBoost, "fit", mock_fit)

    # Run HPO - should complete without raising errors (Optuna will get float('inf') for RMSE)
    run_hpo_tuning(pool, X, y)


def test_hpo_tuning_with_mae_metric(hpo_dataset, hpo_config):
    """Verifies that HPO runs successfully when target metric is set to MAE."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["XGBoost"]
    hpo_config["hpo"]["metric"] = "MAE"
    pool = ModelPool(random_state=42, config=hpo_config)
    orig_xgb_estimators = pool.config["models"]["XGBoost"]["n_estimators"]

    run_hpo_tuning(pool, X, y)

    # Verify parameters were updated under HPO with MAE
    assert pool.config["models"]["XGBoost"]["n_estimators"] != orig_xgb_estimators


def test_hpo_tuning_with_r2_metric(hpo_dataset, hpo_config):
    """Verifies that HPO runs successfully when target metric is set to R2 (maximize direction)."""
    X, y = hpo_dataset
    hpo_config["framework"]["active_models"] = ["XGBoost"]
    hpo_config["hpo"]["metric"] = "R2"
    pool = ModelPool(random_state=42, config=hpo_config)
    orig_xgb_estimators = pool.config["models"]["XGBoost"]["n_estimators"]

    run_hpo_tuning(pool, X, y)

    # Verify parameters were updated under HPO with R2
    assert pool.config["models"]["XGBoost"]["n_estimators"] != orig_xgb_estimators

