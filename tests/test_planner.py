import os
import json
import pandas as pd
import pytest
from automl_framework.planner import DataPlanner

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Fixture to generate a directory with multiple dummy dataset files."""
    d = tmp_path / "data_dir"
    d.mkdir()
    
    # 1. Create a CSV file with an explicit target column "target_y"
    df_csv = pd.DataFrame({
        "feature_a": [1.0, 2.0],
        "feature_b": [3.0, 4.0],
        "target_y": [0.5, 1.5]
    })
    df_csv.to_csv(d / "dataset_1.csv", index=False)
    
    # 2. Create a Parquet file with a substring match "label_col"
    df_pq = pd.DataFrame({
        "x1": [10, 20],
        "label_col": [0, 1]
    })
    df_pq.to_parquet(d / "dataset_2.parquet")
    
    # 3. Create a JSONL file with fallback target (last column)
    jsonl_path = d / "dataset_3.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"col_first": 1, "col_last": 99}) + "\n")
        
    # 4. Create an unsupported file to ensure it's ignored
    with open(d / "ignored.pdf", "w") as f:
        f.write("dummy pdf content")
        
    return d


def test_scan_directory(dummy_data_dir):
    """Verify DataPlanner scans supported files, parses columns, and infers targets."""
    planner = DataPlanner()
    jobs = planner.scan_directory(str(dummy_data_dir))
    
    # We expect exactly 3 jobs sorted by filename
    assert len(jobs) == 3
    
    # Check dataset_1.csv
    job1 = jobs[0]
    assert job1["file_name"] == "dataset_1.csv"
    assert "target_y" in job1["columns"]
    assert job1["target_column"] == "target_y"
    
    # Check dataset_2.parquet
    job2 = jobs[1]
    assert job2["file_name"] == "dataset_2.parquet"
    assert "label_col" in job2["columns"]
    assert job2["target_column"] == "label_col"
    
    # Check dataset_3.jsonl
    job3 = jobs[2]
    assert job3["file_name"] == "dataset_3.jsonl"
    assert job3["columns"] == ["col_first", "col_last"]
    assert job3["target_column"] == "col_last" # fallback to last column


def test_infer_target_column():
    """Verify target column estimation heuristics (exact, partial, and fallback)."""
    planner = DataPlanner()
    
    # 1. Exact match priority
    assert planner.infer_target_column(["col_x", "Target_Y", "other"]) == "Target_Y"
    assert planner.infer_target_column(["Label", "col_x"]) == "Label"
    assert planner.infer_target_column(["col_x", "y"]) == "y"
    
    # 2. Substring match fallback
    assert planner.infer_target_column(["col_x", "my_target_variable", "other"]) == "my_target_variable"
    assert planner.infer_target_column(["class_type", "col_x"]) == "class_type"
    
    # 3. Last column fallback
    assert planner.infer_target_column(["col_a", "col_b", "col_c"]) == "col_c"
    
    # 4. Empty column list fallback
    assert planner.infer_target_column([]) == "Target_Y"


def test_scan_invalid_paths(tmp_path):
    """Verify scan_directory raises correct exceptions on invalid directory paths."""
    planner = DataPlanner()
    
    # Non-existent path
    with pytest.raises(FileNotFoundError):
        planner.scan_directory("non_existent_folder_path_xyz")
        
    # File path instead of a directory
    temp_file = tmp_path / "dummy_file.csv"
    temp_file.write_text("a,b,c")
    with pytest.raises(ValueError):
        planner.scan_directory(str(temp_file))
