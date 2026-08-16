import pytest
import pandas as pd
import numpy as np
import optuna
from automl_framework.model.model_pool import ModelPool
from automl_framework.model.hpo import run_hpo_tuning, suggest_trial_param


@pytest.fixture
def synthetic_hpo_data():
    """Generates synthetic regression data for testing HPO search."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat1": np.random.uniform(-2, 2, n),
        "feat2": np.random.uniform(0, 5, n),
    })
    y = pd.Series(3.0 * X["feat1"] - 1.5 * X["feat2"] + np.random.normal(0, 0.1, n))
    return X, y


def test_suggest_trial_param_mapping():
    """Verifies that suggest_trial_param correctly routes range, choice, and fixed scalar specs."""
    study = optuna.create_study()
    trial = study.ask()

    # 1. Integer Range [min, max]
    int_val = suggest_trial_param(trial, "int_test", [10, 50])
    assert 10 <= int_val <= 50
    assert isinstance(int_val, int)

    # 2. Float Continuous Range [min, max]
    float_val = suggest_trial_param(trial, "float_test", [0.001, 0.1])
    assert 0.001 <= float_val <= 0.1
    assert isinstance(float_val, float)

    # 3. Categorical String Choices
    cat_val = suggest_trial_param(trial, "cat_test", ["relu", "tanh", "gelu"])
    assert cat_val in ["relu", "tanh", "gelu"]

    # 4. Complex Categorical Object Choices (e.g. hidden layer architectures)
    complex_val = suggest_trial_param(trial, "complex_test", [[32, 16], [64, 32]])
    assert complex_val in [[32, 16], [64, 32]]

    # 5. Fixed Scalar Value
    fixed_int = suggest_trial_param(trial, "fixed_int", -1)
    assert fixed_int == -1
    fixed_str = suggest_trial_param(trial, "fixed_str", "exact_value")
    assert fixed_str == "exact_value"


def test_dynamic_hpo_with_custom_ranges_and_fixed_values(synthetic_hpo_data):
    """Verifies HPO dynamically honors custom user ranges and keeps scalar parameters fixed."""
    X, y = synthetic_hpo_data

    custom_hpo_config = {
        "framework": {
            "random_state": 42,
            "active_models": ["XGBoost"]
        },
        "hpo": {
            "enabled": True,
            "metric": "RMSE",
            "n_trials": 3
        },
        "models": {
            "XGBoost": {
                "n_estimators": [15, 35],      # Custom Range
                "learning_rate": [0.02, 0.08],  # Custom Range
                "max_depth": [2, 4],            # Custom Range
                "n_jobs": 1                     # Fixed Scalar
            }
        }
    }

    pool = ModelPool(random_state=42, config=custom_hpo_config)
    run_hpo_tuning(pool, X, y)

    tuned_xgb = pool.config["models"]["XGBoost"]

    # Assert tuned parameters fall strictly within the user-specified bounds
    assert 15 <= tuned_xgb["n_estimators"] <= 35
    assert 0.02 <= tuned_xgb["learning_rate"] <= 0.08
    assert 2 <= tuned_xgb["max_depth"] <= 4
    # Assert fixed parameter remained unchanged
    assert tuned_xgb["n_jobs"] == 1


def test_dynamic_hpo_with_categorical_mlp_architectures(synthetic_hpo_data):
    """Verifies HPO with complex categorical layer choices on MLP."""
    X, y = synthetic_hpo_data

    mlp_hpo_config = {
        "framework": {
            "random_state": 42,
            "active_models": ["MLP"]
        },
        "hpo": {
            "enabled": True,
            "metric": "MAE",
            "n_trials": 3
        },
        "models": {
            "MLP": {
                "hidden_layer_sizes": [[16, 8], [32, 16]], # Categorical architecture choices
                "activation": ["relu", "tanh"],
                "max_iter": 20                              # Fixed scalar
            }
        }
    }

    pool = ModelPool(random_state=42, config=mlp_hpo_config)
    run_hpo_tuning(pool, X, y)

    tuned_mlp = pool.config["models"]["MLP"]
    assert tuned_mlp["hidden_layer_sizes"] in [[16, 8], [32, 16]]
    assert tuned_mlp["activation"] in ["relu", "tanh"]
    assert tuned_mlp["max_iter"] == 20
