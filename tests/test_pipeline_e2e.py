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
    """Verifies that AutoMLPipeline executes end-to-end and creates all expected output artifacts in outputs/<run_id>/."""
    env = e2e_environment
    test_run_id = "run_20260816_test_e2e"
    
    pipeline = AutoMLPipeline(
        config_path=env["config_path"],
        run_id=test_run_id
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
        
    # Verify all generated file artifacts in output_dir (outputs/<run_id>/)
    out_dir = os.path.join(env["output_dir"], test_run_id)
    assert os.path.exists(out_dir), f"Run ID directory '{out_dir}' was not created"
    
    json_report = os.path.join(out_dir, "report.json")
    html_report = os.path.join(out_dir, "report.html")
    md_summary = os.path.join(out_dir, "summary.md")
    r2_chart = os.path.join(out_dir, "model_comparison_r2.png")
    rmse_chart = os.path.join(out_dir, "model_comparison_rmse.png")
    
    assert os.path.exists(json_report), "JSON report was not generated"
    assert os.path.exists(html_report), "HTML report was not generated"
    assert os.path.exists(md_summary), "Markdown summary was not generated"
    assert os.path.exists(r2_chart), "R2 comparison chart was not generated"
    assert os.path.exists(rmse_chart), "RMSE comparison chart was not generated"
    
    # Check JSON report contents
    with open(json_report, "r", encoding="utf-8") as f:
        report_data = json.load(f)
        assert report_data["run_id"] == test_run_id
        assert "best_model" in report_data
        assert report_data["best_model"] in ["RandomForest", "XGBoost"]


def test_automl_pipeline_parameter_overrides(e2e_environment):
    """Verifies that CLI / init parameter overrides take precedence over YAML config."""
    env = e2e_environment
    test_run_id = "run_custom_override"
    
    pipeline = AutoMLPipeline(
        config_path=env["config_path"],
        run_id=test_run_id,
        target="Target_Price",
        test_size=0.4
    )
    
    assert pipeline.test_size == 0.4
    assert pipeline.run_id == test_run_id
    assert pipeline.target_column == "Target_Price"


def test_automl_pipeline_run_id_collision_handling(e2e_environment):
    """Verifies that duplicate run_id raises FileExistsError unless overwrite_run is True."""
    env = e2e_environment
    test_run_id = "run_collision_check"
    
    # First execution creates the directory
    pipeline1 = AutoMLPipeline(
        config_path=env["config_path"],
        run_id=test_run_id
    )
    assert os.path.exists(pipeline1.output_dir)
    
    # Second execution with same run_id without overwrite_run should raise FileExistsError
    with pytest.raises(FileExistsError) as exc_info:
        AutoMLPipeline(
            config_path=env["config_path"],
            run_id=test_run_id,
            overwrite_run=False
        )
    assert "already exists" in str(exc_info.value)
    
    # Third execution with overwrite_run=True should succeed
    pipeline3 = AutoMLPipeline(
        config_path=env["config_path"],
        run_id=test_run_id,
        overwrite_run=True
    )
    assert pipeline3.run_id == test_run_id
