import os
import json
import yaml
import pytest
import numpy as np
import pandas as pd
from main import AutoMLPipeline


@pytest.fixture
def e2e_environment(tmp_path):
    """Sets up a complete isolated directory structure and synthetic dataset for E2E testing."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    config_dir = tmp_path / "configs"
    
    data_dir.mkdir()
    output_dir.mkdir()
    config_dir.mkdir()
    
    # Create synthetic dataset CSV
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "Feature_A": np.random.uniform(10, 50, n),
        "Feature_B": np.random.normal(0, 2, n),
        "Category_C": np.random.choice(["Type1", "Type2", "Type3"], n),
        "Target_Price": np.random.uniform(100, 500, n),
    })
    csv_path = data_dir / "sample_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    # Create configuration YAML
    config_payload = {
        "framework": {
            "random_state": 42,
            "test_size": 0.25,
            "active_models": ["RandomForest", "XGBoost"]
        },
        "data": {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "target_column": "Target_Price",
            "feature_columns": None,
            "ignored_columns": None,
            "split": {
                "method": "train_test_split",
                "test_size": 0.25,
                "shuffle": True
            }
        },
        "models": {
            "RandomForest": {"n_estimators": 5, "max_depth": 3},
            "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1}
        }
    }
    
    config_path = config_dir / "test_config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f)
        
    return {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "config_path": str(config_path)
    }


def test_automl_pipeline_e2e_execution(e2e_environment):
    """Verifies that AutoMLPipeline executes end-to-end and creates all expected output artifacts."""
    env = e2e_environment
    turn_idx = 1
    
    pipeline = AutoMLPipeline(
        config_path=env["config_path"],
        turn=turn_idx
    )
    
    # Run full pipeline
    pipeline.run(dataset_path=env["csv_path"])
    
    # Verify split shapes
    assert pipeline.X_train is not None
    assert pipeline.y_train is not None
    assert pipeline.X_test is not None
    assert pipeline.y_test is not None
    assert len(pipeline.X_train) == 37  # 50 * 0.75 rounded
    assert len(pipeline.X_test) == 13   # 50 * 0.25 rounded
    
    # Verify evaluation metrics
    assert pipeline.metrics is not None
    assert "RandomForest" in pipeline.metrics
    assert "XGBoost" in pipeline.metrics
    for model_name in ["RandomForest", "XGBoost"]:
        assert "R2" in pipeline.metrics[model_name]
        assert "RMSE" in pipeline.metrics[model_name]
        assert "MAE" in pipeline.metrics[model_name]
        
    # Verify all generated file artifacts in output_dir
    out_dir = env["output_dir"]
    
    json_report = os.path.join(out_dir, f"turn_{turn_idx}_report.json")
    html_report = os.path.join(out_dir, f"turn_{turn_idx}_report.html")
    md_summary = os.path.join(out_dir, f"turn_{turn_idx}_summary.md")
    r2_chart = os.path.join(out_dir, f"turn_{turn_idx}_model_comparison_r2.png")
    rmse_chart = os.path.join(out_dir, f"turn_{turn_idx}_model_comparison_rmse.png")
    
    assert os.path.exists(json_report), "JSON report was not generated"
    assert os.path.exists(html_report), "HTML report was not generated"
    assert os.path.exists(md_summary), "Markdown summary was not generated"
    assert os.path.exists(r2_chart), "R2 comparison chart was not generated"
    assert os.path.exists(rmse_chart), "RMSE comparison chart was not generated"
    
    # Check JSON report contents
    with open(json_report, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        assert report_data["turn"] == turn_idx
        assert "best_model" in report_data
        assert report_data["best_model"] in ["RandomForest", "XGBoost"]


def test_automl_pipeline_parameter_overrides(e2e_environment):
    """Verifies that CLI / init parameter overrides take precedence over YAML config."""
    env = e2e_environment
    
    pipeline = AutoMLPipeline(
        config_path=env["config_path"],
        turn=2,
        target="Target_Price",
        test_size=0.4
    )
    
    assert pipeline.test_size == 0.4
    assert pipeline.turn == 2
    assert pipeline.target_column == "Target_Price"
