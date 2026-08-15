import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from automl_framework.model.model_factory import ModelFactory, ModelType
from automl_framework.util.shap_analyzer import SHAPAnalyzer
from main import AutoMLPipeline


@pytest.fixture
def synthetic_data():
    """Generates synthetic tabular regression dataset for test purposes."""
    X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=42)
    feature_names = ["Feat_Alpha", "Feat_Beta", "Feat_Gamma", "Feat_Delta"]
    df_X = pd.DataFrame(X, columns=feature_names)
    series_y = pd.Series(y, name="Target_Y")
    
    X_train, X_test = df_X.iloc[:40], df_X.iloc[40:]
    y_train, y_test = series_y.iloc[:40], series_y.iloc[40:]
    return X_train, y_train, X_test, y_test, feature_names


def test_shap_analyzer_initialization(tmp_path):
    """Verifies SHAPAnalyzer initializes and creates output directory."""
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path / "shap_out"), max_samples=50)
    assert os.path.exists(str(tmp_path / "shap_out"))
    assert analyzer.max_samples == 50


def test_shap_tabicl_dedicated_explainer(synthetic_data, tmp_path):
    """Verifies TabICL utilizes dedicated in-context explainer pipeline."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    # Instantiate TabICL model wrapper
    tabicl_wrapper = ModelFactory.create_model(ModelType.TABICL, config={}, random_state=42)
    tabicl_wrapper.fit(X_train, y_train)
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=20, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=tabicl_wrapper,
        model_name="TabICL",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        turn=101
    )
    
    assert report is not None
    assert report["model_name"] == "TabICL"
    assert "TabICL Dedicated In-Context Explainer" in report["explainer_engine"]
    assert len(report["top_features"]) > 0
    assert len(report["mean_abs_shap"]) == len(feature_names)
    
    # Check that report JSON exists on disk
    json_path = os.path.join(str(tmp_path), "turn_101_TabICL_shap_report.json")
    assert os.path.exists(json_path)
    
    # Check that at least the bar chart image was saved
    bar_img = os.path.join(str(tmp_path), "turn_101_TabICL_shap_bar.png")
    assert os.path.exists(bar_img)
    assert os.path.getsize(bar_img) > 0


def test_shap_tree_explainer(synthetic_data, tmp_path):
    """Verifies XGBoost / RandomForest utilize TreeExplainer."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    # Test RandomForest
    rf_wrapper = ModelFactory.create_model(ModelType.RANDOM_FOREST, config={}, random_state=42)
    rf_wrapper.fit(X_train, y_train)
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=20, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=rf_wrapper,
        model_name="RandomForest",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        turn=102
    )
    
    assert report is not None
    assert "TreeExplainer" in report["explainer_engine"]
    assert report["num_samples_analyzed"] <= 20
    
    json_path = os.path.join(str(tmp_path), "turn_102_RandomForest_shap_report.json")
    assert os.path.exists(json_path)


def test_shap_kernel_explainer(synthetic_data, tmp_path):
    """Verifies neural models utilize KernelExplainer/ModelExplainer."""
    X_train, y_train, X_test, y_test, feature_names = synthetic_data
    
    mlp_wrapper = ModelFactory.create_model(ModelType.MLP, config={}, random_state=42)
    mlp_wrapper.fit(X_train, y_train)
    
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path), max_samples=15, random_state=42)
    report = analyzer.analyze_model(
        model_wrapper=mlp_wrapper,
        model_name="MLP",
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        turn=103
    )
    
    assert report is not None
    assert "ModelExplainer" in report["explainer_engine"] or "KernelExplainer" in report["explainer_engine"]


def test_shap_analyzer_shielding(tmp_path):
    """Verifies analyzer shields against non-standard or faulty models gracefully."""
    class FaultyModel:
        def predict(self, X):
            raise RuntimeError("Faulty prediction model")
            
    analyzer = SHAPAnalyzer(output_dir=str(tmp_path))
    X_dummy = np.random.randn(10, 3)
    
    # Should log error and return None without crashing
    report = analyzer.analyze_model(
        model_wrapper=FaultyModel(),
        model_name="FaultyModel",
        X_train=X_dummy,
        X_test=X_dummy,
        turn=999
    )
    assert report is None


def test_pipeline_e2e_with_shap_integration(tmp_path):
    """Verifies AutoMLPipeline runs with SHAP enabled and produces SHAP outputs."""
    csv_file = str(tmp_path / "test_data.csv")
    df = pd.DataFrame({
        "Feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "Feat2": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "Target_Y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    })
    df.to_csv(csv_file, index=False)
    
    pipeline = AutoMLPipeline(
        turn=777,
        enable_shap=True,
        shap_model="RandomForest"
    )
    pipeline.output_dir = str(tmp_path / "outputs")
    pipeline.shap_analyzer.output_dir = str(tmp_path / "outputs")
    os.makedirs(pipeline.output_dir, exist_ok=True)
    
    pipeline.run(dataset_path=csv_file)
    
    shap_report_file = os.path.join(pipeline.output_dir, "turn_777_RandomForest_shap_report.json")
    assert os.path.exists(shap_report_file)
    
    with open(shap_report_file, "r") as f:
        data = json.load(f)
    assert data["model_name"] == "RandomForest"
    assert "Feat1" in data["mean_abs_shap"] or "Feat2" in data["mean_abs_shap"]
