import os
import json
import numpy as np
import pandas as pd
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
    
    filepath = visualizer.plot_actual_vs_predicted(y_true, y_pred, model_name="MLP")
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "MLP" in filepath
    assert "actual_vs_pred" in filepath


def test_visualizer_plot_residuals(temp_output_dir, dummy_results):
    """Test plot_residuals creates the correct PNG file in output directory."""
    y_true, y_pred, _ = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.plot_residuals(y_true, y_pred, model_name="MLP")
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "MLP" in filepath
    assert "residuals" in filepath


def test_visualizer_plot_model_comparison(temp_output_dir, dummy_results):
    """Test plot_model_comparison creates horizontal comparison charts properly."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.plot_model_comparison(metrics, metric_name="R2")
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "model_comparison_r2" in filepath


def test_visualizer_save_json_report(temp_output_dir, dummy_results):
    """Test save_json_report dumps a structured JSON file."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.save_json_report(metrics, run_id="run_test_123")
    
    assert os.path.exists(filepath)
    assert filepath.endswith("report.json")
    
    # Read the JSON structure and assert values
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["run_id"] == "run_test_123"
    assert "MLP" in data["metrics"]
    assert data["best_model"] == "MLP"  # MLP R2 (0.95) > RandomForest R2 (0.90)


def test_visualizer_save_html_report(temp_output_dir, dummy_results):
    """Test save_html_report generates an interactive standalone HTML report."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.save_html_report(
        metrics, 
        run_id="run_test_123", 
        metadata={"target_column": "Target_Y", "split_method": "train_test_split", "train_samples": 100, "test_samples": 25}
    )
    
    assert os.path.exists(filepath)
    assert filepath.endswith("report.html")
    
    with open(filepath, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    assert "<!DOCTYPE html>" in html_content
    assert "MLP" in html_content
    assert "RandomForest" in html_content
    assert "Champion" in html_content
    assert "Target_Y" in html_content
    assert "leaderboardTable" in html_content


def test_visualizer_save_markdown_summary(temp_output_dir, dummy_results):
    """Test save_markdown_summary generates a shareable GFM summary report."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.save_markdown_summary(
        metrics, 
        run_id="run_test_123", 
        metadata={"target_column": "Target_Y", "split_method": "train_test_split", "train_samples": 100, "test_samples": 25}
    )
    
    assert os.path.exists(filepath)
    assert filepath.endswith("summary.md")
    
    with open(filepath, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    assert "# 🚀 AutoML Regression Benchmark Summary" in md_content
    assert "🏆 Champion Model" in md_content
    assert "MLP" in md_content
    assert "RandomForest" in md_content
    assert "Target_Y" in md_content


def test_visualizer_shap_explainability(temp_output_dir):
    """Test plot_shap_explainability runs and outputs correct PNG files."""
    X_train = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0], "feat2": [0.5, 1.5, 2.5, 3.5, 4.5]})
    X_test = pd.DataFrame({"feat1": [1.5, 2.5], "feat2": [0.75, 1.75]})
    
    class DummyModel:
        def predict(self, X):
            return np.array([1.5] * len(X))
            
    dummy_model_wrap = DummyModel()
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    paths = visualizer.plot_shap_explainability(
        model_wrap=dummy_model_wrap,
        X_train=X_train,
        X_test=X_test,
        model_name="CustomEstimator",
        max_samples=3
    )
    
    assert "summary_plot" in paths
    assert "bar_plot" in paths
    assert os.path.exists(paths["summary_plot"])
    assert os.path.exists(paths["bar_plot"])


def test_visualizer_plot_learning_curve(temp_output_dir):
    """Test plot_learning_curve outputs a correct PNG file."""
    visualizer = Visualizer(output_dir=temp_output_dir)
    loss_history = [10.0, 8.5, 6.2, 4.1, 2.5, 1.2]
    
    filepath = visualizer.plot_learning_curve(loss_history, model_name="MLP")
    
    assert filepath != ""
    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "learning_curve" in filepath


def test_visualizer_save_markdown_report(temp_output_dir):
    """Test save_markdown_report outputs a correct Markdown report file."""
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    metrics = {
        "MLP": {"RMSE": 1.25, "MAE": 1.10, "R2": 0.95},
        "RandomForest": {"RMSE": 2.20, "MAE": 1.80, "R2": 0.90}
    }
    
    dataset_info = {
        "target_column": "target",
        "num_features": 8,
        "train_size": 800,
        "test_size": 200,
        "split_method": "KFold"
    }
    
    filepath = visualizer.save_markdown_report(
        metrics=metrics,
        run_id="run_test_123",
        dataset_info=dataset_info,
        shap_reports={"MLP": {"summary_plot": "mlp_summary.png", "bar_plot": "mlp_bar.png"}},
        learning_curves={"MLP": "mlp_loss.png"}
    )
    
    assert filepath != ""
    assert os.path.exists(filepath)
    assert filepath.endswith(".md")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "AutoML Tabular Regression Benchmark Report" in content
    assert "Model Leaderboard" in content
    assert "🥇 Champion" in content
    assert "MLP" in content
    assert "Target Column" in content
