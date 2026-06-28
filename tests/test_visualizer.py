import os
import json
import numpy as np
import pytest
from automl_framework.util import Visualizer

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to create a temporary directory for plotting outputs."""
    return str(tmp_path / "outputs")


@pytest.fixture
def dummy_results():
    """Fixture supplying sample values and metrics dictionaries for visualization testing."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 32.0, 38.0])
    
    metrics = {
        "MLP": {"RMSE": 1.5, "MAE": 1.2, "R2": 0.95},
        "RandomForest": {"RMSE": 2.2, "MAE": 1.8, "R2": 0.90}
    }
    return y_true, y_pred, metrics


def test_visualizer_plot_actual_vs_predicted(temp_output_dir, dummy_results):
    """Test plot_actual_vs_predicted creates the correct PNG file in output directory."""
    y_true, y_pred, _ = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.plot_actual_vs_predicted(y_true, y_pred, model_name="MLP", turn=1)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "MLP" in filepath
    assert "turn_1" in filepath


def test_visualizer_plot_residuals(temp_output_dir, dummy_results):
    """Test plot_residuals creates the correct PNG file in output directory."""
    y_true, y_pred, _ = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.plot_residuals(y_true, y_pred, model_name="MLP", turn=1)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "MLP" in filepath
    assert "residuals" in filepath


def test_visualizer_plot_model_comparison(temp_output_dir, dummy_results):
    """Test plot_model_comparison creates horizontal comparison charts properly."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.plot_model_comparison(metrics, metric_name="R2", turn=1)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "comparison_r2" in filepath


def test_visualizer_save_json_report(temp_output_dir, dummy_results):
    """Test save_json_report generates a correct JSON structure detailing performance champions."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.save_json_report(metrics, turn=1)
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".json")
    
    # Read the JSON structure and assert values
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["turn"] == 1
    assert "MLP" in data["metrics"]
    assert data["best_model"] == "MLP"  # MLP R2 (0.95) > RandomForest R2 (0.90)


def test_visualizer_shap_explainability(temp_output_dir):
    """Test plot_shap_explainability runs and outputs correct PNG files."""
    import pandas as pd
    
    # Generate dummy training and test datasets
    X_train = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0], "feat2": [0.5, 1.5, 2.5, 3.5, 4.5]})
    X_test = pd.DataFrame({"feat1": [1.5, 2.5], "feat2": [0.75, 1.75]})
    
    # Use a custom class instead of MagicMock to prevent shap from misidentifying it as a linear model
    class DummyModel:
        def predict(self, X):
            return np.array([1.5] * len(X))
            
    dummy_model_wrap = DummyModel()
    
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    # Run explainability plotting with a small number of samples
    paths = visualizer.plot_shap_explainability(
        model_wrap=dummy_model_wrap,
        X_train=X_train,
        X_test=X_test,
        model_name="CustomEstimator",
        turn=1,
        max_samples=3
    )
    
    # Verify that paths are returned and the files exist
    assert "summary_plot" in paths
    assert "bar_plot" in paths
    assert os.path.exists(paths["summary_plot"])
    assert os.path.exists(paths["bar_plot"])


