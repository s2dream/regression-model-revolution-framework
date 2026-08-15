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


def test_visualizer_save_html_report(temp_output_dir, dummy_results):
    """Test save_html_report generates an interactive standalone HTML report."""
    _, _, metrics = dummy_results
    visualizer = Visualizer(output_dir=temp_output_dir)
    
    filepath = visualizer.save_html_report(
        metrics, 
        turn=1, 
        metadata={"target_column": "Target_Y", "split_method": "train_test_split", "train_samples": 100, "test_samples": 25}
    )
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".html")
    assert "turn_1_report.html" in filepath
    
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
        turn=1, 
        metadata={"target_column": "Target_Y", "split_method": "train_test_split", "train_samples": 100, "test_samples": 25}
    )
    
    assert os.path.exists(filepath)
    assert filepath.endswith(".md")
    assert "turn_1_summary.md" in filepath
    
    with open(filepath, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    assert "# 🚀 AutoML Regression Benchmark Summary" in md_content
    assert "🏆 Champion Model" in md_content
    assert "MLP" in md_content
    assert "RandomForest" in md_content
    assert "Target_Y" in md_content

