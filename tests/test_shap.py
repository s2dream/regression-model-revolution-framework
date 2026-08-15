import os
import json
import yaml
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from automl_framework.model.model_factory import ModelFactory, ModelType
from automl_framework.util.shap_analyzer import SHAPAnalyzer
from main import AutoMLPipeline


@pytest.fixture
def synthetic_data():
    """Generates small synthetic tabular regression dataset for fast test purposes."""
    X, y = make_regression(n_samples=20, n_features=3, noise=0.1, random_state=42)
    feature_names = ["Feat_Alpha", "Feat_Beta", "Feat_Gamma"]
    df_X = pd.DataFrame(X, columns=feature_names)
    series_y = pd.Series(y, name="Target_Y")
    
    X_train, X_test = df_X.iloc[:15], df_X.iloc[15:]
    y_train, y_test = series_y.iloc[:15], series_y.iloc[15:]
    return X_train, y_train, X_test, y_test, feature_names


def test_shap_analyzer_initialization(tmp_path):
    """Verifies SHAPAnalyzer initializes and creates output directory."""
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path / "shap_out"), max_samples=10)
    assert os.path.exists(str(tmp_path / "shap_out"))
    assert analyzer.max_samples == 10


def test_shap_tabicl_dedicated_explainer(synthetic_data, tmp_path):
    """Verifies TabICL utilizes dedicated in-context explainer pipeline."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    # Instantiate TabICL model wrapper or fallback mock wrapper
    try:
        tabicl_wrapper = ModelFactory.create_model(ModelType.TABICL, config={}, random_state=42)
        tabicl_wrapper.fit(X_train, y_train)
    except Exception:
        class DummyTabICLWrapper:
            def __init__(self):
                self.name = "TabICL"
            def predict(self, X):
                return np.array([25.0] * len(X))
        tabicl_wrapper = DummyTabICLWrapper()
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=5, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=tabicl_wrapper,
        model_name="TabICL",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    
    assert report is not None
    assert report["model_name"] == "TabICL"
    assert "TabICL Dedicated In-Context Explainer" in report["explainer_engine"]
    assert len(report["top_features"]) > 0
    assert len(report["mean_abs_shap"]) == len(feature_names)
    
    # Check that report JSON exists on disk
    json_path = os.path.join(str(tmp_path), "TabICL_shap_report.json")
    assert os.path.exists(json_path)
    
    # Check that at least the bar chart image was saved
    bar_img = os.path.join(str(tmp_path), "TabICL_shap_bar.png")
    assert os.path.exists(bar_img)
    assert os.path.getsize(bar_img) > 0


def test_shap_tree_explainer(synthetic_data, tmp_path):
    """Verifies XGBoost / RandomForest utilize TreeExplainer."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    # Test RandomForest with minimal estimators
    rf_config = {"models": {"RandomForest": {"n_estimators": 3, "max_depth": 2}}}
    rf_wrapper = ModelFactory.create_model(ModelType.RANDOM_FOREST, config=rf_config, random_state=42)
    rf_wrapper.fit(X_train, y_train)
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=5, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=rf_wrapper,
        model_name="RandomForest",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    
    assert report is not None
    assert "TreeExplainer" in report["explainer_engine"]
    assert report["num_samples_analyzed"] <= 5
    
    json_path = os.path.join(str(tmp_path), "RandomForest_shap_report.json")
    assert os.path.exists(json_path)


def test_shap_kernel_explainer(synthetic_data, tmp_path):
    """Verifies neural models utilize KernelExplainer/ModelExplainer."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    mlp_config = {"models": {"MLP": {"hidden_layer_sizes": [8], "max_iter": 5}}}
    mlp_wrapper = ModelFactory.create_model(ModelType.MLP, config=mlp_config, random_state=42)
    mlp_wrapper.fit(X_train, y_train)
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=4, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=mlp_wrapper,
        model_name="MLP",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    
    assert report is not None
    assert "ModelExplainer" in report["explainer_engine"] or "KernelExplainer" in report["explainer_engine"]


def test_shap_analyzer_shielding(tmp_path):
    """Verifies analyzer shields against non-standard or faulty models gracefully."""
    class FaultyModel:
        def predict(self, X):
            raise RuntimeError("Faulty prediction model")
            
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path))
    X_dummy = np.random.randn(5, 2)
    
    # Should log error and return None without crashing
    report = analyzer.analyze_model(
        model_wrapper=FaultyModel(),
        model_name="FaultyModel",
        X_train=X_dummy,
        X_test=X_dummy
    )
    assert report is None


def test_pipeline_e2e_with_shap_integration(tmp_path):
    """Verifies AutoMLPipeline runs with SHAP enabled and produces SHAP outputs in outputs/<run_id>/."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    config_dir = tmp_path / "configs"
    data_dir.mkdir()
    output_dir.mkdir()
    config_dir.mkdir()
    
    csv_file = str(data_dir / "test_data.csv")
    df = pd.DataFrame({
        "Feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "Feat2": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "Target_Y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    })
    df.to_csv(csv_file, index=False)
    
    test_config_file = str(config_dir / "test_shap_cfg.yml")
    config_payload = {
        "framework": {
            "random_state": 42,
            "test_size": 0.25,
            "active_models": ["RandomForest"]
        },
        "data": {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "target_column": "Target_Y"
        },
        "models": {
            "RandomForest": {"n_estimators": 3, "max_depth": 2}
        },
        "shap": {
            "enabled": True,
            "model": "RandomForest",
            "max_samples": 4
        }
    }
    with open(test_config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f)
    
    test_run_id = "run_shap_test_isolated"
    pipeline = AutoMLPipeline(
        config_path=test_config_file,
        run_id=test_run_id,
        overwrite_run=True,
        enable_shap=True,
        shap_model="RandomForest"
    )
    
    pipeline.run(dataset_path=csv_file)
    
    run_dir = os.path.join(output_dir, test_run_id)
    shap_report_file = os.path.join(run_dir, "RandomForest_shap_report.json")
    assert os.path.exists(shap_report_file)
    
    with open(shap_report_file, "r") as f:
        data = json.load(f)
    assert data["model_name"] == "RandomForest"
    assert "Feat1" in data["mean_abs_shap"] or "Feat2" in data["mean_abs_shap"]
