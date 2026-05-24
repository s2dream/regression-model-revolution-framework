import numpy as np
import pandas as pd
import pytest
from automl_framework.model import ModelPool, StandardBenchmarkExecutor
from automl_framework.model.wrappers import ModelWrapperRandomForest

@pytest.fixture
def test_data():
    """Fixture to generate a small synthetic dataset for training and testing estimators."""
    np.random.seed(42)
    X = pd.DataFrame({"X1": np.random.uniform(-2, 2, 20), "X2": np.random.uniform(0, 5, 20)})
    y = pd.Series(3.0 * X["X1"] + 1.5 * X["X2"] + np.random.normal(0, 0.1, 20))
    return X, y


@pytest.fixture
def rf_config():
    """Fixture to define a central configurations YAML mapping containing only RandomForest."""
    return {
        "framework": {
            "active_models": ["RandomForest"]
        },
        "models": {
            "RandomForest": {
                "n_estimators": 10,
                "max_depth": 3
            }
        }
    }


def test_model_pool_initialization(rf_config):
    """Test ModelPool correctly loads active models from configuration."""
    pool = ModelPool(random_state=42, config=rf_config)
    
    assert "RandomForest" in pool.list_available_models()
    assert len(pool.list_available_models()) == 1
    
    # Retrieve RF and verify parameters are applied correctly
    rf_wrap = pool.get_model("RandomForest")
    assert isinstance(rf_wrap, ModelWrapperRandomForest)
    assert rf_wrap.model.n_estimators == 10
    assert rf_wrap.model.max_depth == 3


def test_model_pool_custom_model():
    """Test adding custom scikit-learn models to the ModelPool inventory."""
    from sklearn.linear_model import LinearRegression
    pool = ModelPool(random_state=42, config={"framework": {"active_models": []}})
    
    lr_model = LinearRegression()
    pool.add_custom_model("LinearRegression", lr_model)
    
    assert "LinearRegression" in pool.list_available_models()
    assert pool.get_model("LinearRegression") is not None


def test_standard_benchmark_executor(test_data, rf_config):
    """Test fitting, evaluating, and obtaining predictions through StandardBenchmarkExecutor."""
    X, y = test_data
    pool = ModelPool(random_state=42, config=rf_config)
    executor = StandardBenchmarkExecutor(pool)
    
    # Fit all models
    executor.fit_all(X, y)
    
    # Get predictions
    preds = executor.get_predictions(X)
    assert "RandomForest" in preds
    assert preds["RandomForest"].shape == (20,)
    
    # Evaluate all models
    metrics = executor.evaluate_all(X, y)
    assert "RandomForest" in metrics
    assert "RMSE" in metrics["RandomForest"]
    assert "MAE" in metrics["RandomForest"]
    assert "R2" in metrics["RandomForest"]
    
    assert isinstance(metrics["RandomForest"]["R2"], float)
    assert metrics["RandomForest"]["R2"] <= 1.0
