import numpy as np
import pandas as pd
import pytest
from automl_framework.model.model_factory import ModelType, ModelFactory
from automl_framework.model.wrappers import (
    ABCModelWrapper,
    ModelWrapperXGBoost,
    ModelWrapperMLP,
    ModelWrapperRandomForest,
    ModelWrapperCatBoost,
    ModelWrapperTransformer,
    ModelWrapperTabPFN,
)
from automl_framework.model.model_pool import ModelPool
from automl_framework.model.model_executor import StandardBenchmarkExecutor


@pytest.fixture
def synthetic_data():
    """Generates synthetic tabular regression dataset for model testing."""
    np.random.seed(42)
    n_samples = 40
    X = pd.DataFrame({
        "feat1": np.random.uniform(-3, 3, n_samples),
        "feat2": np.random.uniform(0, 10, n_samples),
        "feat3": np.random.normal(0, 1, n_samples),
    })
    y = pd.Series(2.5 * X["feat1"] - 1.2 * X["feat2"] + 0.5 * X["feat3"] + np.random.normal(0, 0.1, n_samples))
    return X, y


@pytest.fixture
def full_models_config():
    """Configuration covering all supported regression models with fast training params."""
    return {
        "framework": {
            "random_state": 42,
            "active_models": ["XGBoost", "MLP", "RandomForest", "CatBoost", "Transformer"]
        },
        "models": {
            "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
            "MLP": {"hidden_layer_sizes": [32, 16], "max_iter": 10},
            "RandomForest": {"n_estimators": 5, "max_depth": 3},
            "CatBoost": {"iterations": 5, "depth": 2, "verbose": 0},
            "Transformer": {"epochs": 3, "d_model": 16, "nhead": 2, "num_layers": 1, "batch_size": 16},
            "TabPFN": {"N_ensemble_configurations": 2},
        }
    }


def test_model_type_from_str_resolution():
    """Verifies case-insensitive and variant string resolutions to ModelType enum."""
    assert ModelType.from_str("XGBoost") == ModelType.XGBOOST
    assert ModelType.from_str("xgboost") == ModelType.XGBOOST
    assert ModelType.from_str("mlp") == ModelType.MLP
    assert ModelType.from_str("MLP") == ModelType.MLP
    assert ModelType.from_str("RandomForest") == ModelType.RANDOM_FOREST
    assert ModelType.from_str("random_forest") == ModelType.RANDOM_FOREST
    assert ModelType.from_str("random-forest") == ModelType.RANDOM_FOREST
    assert ModelType.from_str("CatBoost") == ModelType.CATBOOST
    assert ModelType.from_str("catboost") == ModelType.CATBOOST
    assert ModelType.from_str("Transformer") == ModelType.TRANSFORMER
    assert ModelType.from_str("transformer") == ModelType.TRANSFORMER
    assert ModelType.from_str("TabPFN") == ModelType.TABPFN
    assert ModelType.from_str("tabpfn") == ModelType.TABPFN

    with pytest.raises(ValueError, match="is not a valid ModelType"):
        ModelType.from_str("NonExistentModel")


@pytest.mark.parametrize("model_name,expected_wrapper_type", [
    ("XGBoost", ModelWrapperXGBoost),
    ("MLP", ModelWrapperMLP),
    ("RandomForest", ModelWrapperRandomForest),
    ("CatBoost", ModelWrapperCatBoost),
    ("Transformer", ModelWrapperTransformer),
    ("TabPFN", ModelWrapperTabPFN),
])
def test_model_factory_creates_all_wrappers(full_models_config, model_name, expected_wrapper_type):
    """Verifies that ModelFactory correctly builds all wrapper types."""
    wrapper = ModelFactory.create_model(model_name, full_models_config, random_state=42)
    assert isinstance(wrapper, expected_wrapper_type)
    assert isinstance(wrapper, ABCModelWrapper)
    assert wrapper.name == model_name


@pytest.mark.parametrize("model_name", ["XGBoost", "MLP", "RandomForest", "CatBoost", "Transformer"])
def test_all_models_fit_and_predict(synthetic_data, full_models_config, model_name):
    """Verifies fit and predict workflow for each individual model wrapper."""
    X, y = synthetic_data
    wrapper = ModelFactory.create_model(model_name, full_models_config, random_state=42)
    
    wrapper.fit(X, y)
    preds = wrapper.predict(X)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert not np.isnan(preds).any()


def test_full_model_pool_and_executor(synthetic_data, full_models_config):
    """Verifies multi-model ModelPool training and evaluation through StandardBenchmarkExecutor."""
    X, y = synthetic_data
    pool = ModelPool(random_state=42, config=full_models_config)
    
    available = pool.list_available_models()
    assert set(available) == {"XGBoost", "MLP", "RandomForest", "CatBoost", "Transformer"}
    
    executor = StandardBenchmarkExecutor(pool)
    executor.fit_all(X, y)
    
    preds = executor.get_predictions(X)
    assert len(preds) == 5
    for name in available:
        assert name in preds
        assert preds[name].shape == (len(X),)
        
    metrics = executor.evaluate_all(X, y)
    assert len(metrics) == 5
    for name in available:
        assert "RMSE" in metrics[name]
        assert "MAE" in metrics[name]
        assert "R2" in metrics[name]
        assert isinstance(metrics[name]["R2"], float)
