import os
import json
import pytest
import pandas as pd
from PIL import Image
from app import (
    load_config,
    save_config,
    is_valid_image,
    get_dataset_columns,
    preview_dataset_sample,
)


@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture supplying temporary directory for file I/O tests."""
    return tmp_path


def test_load_and_save_config(temp_workspace):
    """Verifies save_config and load_config maintain data integrity."""
    config_file = str(temp_workspace / "subfolder" / "test_config.yml")
    sample_data = {
        "framework": {"random_state": 99, "active_models": ["XGBoost"]},
        "models": {"XGBoost": {"n_estimators": 50}}
    }
    
    save_config(sample_data, path=config_file)
    assert os.path.exists(config_file)
    
    loaded = load_config(path=config_file)
    assert loaded == sample_data
    
    # Test non-existent file handling
    empty = load_config(path=str(temp_workspace / "non_existent.yml"))
    assert empty == {}


def test_get_dataset_columns_csv_and_tsv(temp_workspace):
    """Verifies column extraction across CSV, TSV, and Parquet formats."""
    csv_file = str(temp_workspace / "data.csv")
    tsv_file = str(temp_workspace / "data.tsv")
    parquet_file = str(temp_workspace / "data.parquet")
    
    df = pd.DataFrame({"colA": [1, 2], "colB": [3.0, 4.0], "colC": ["x", "y"]})
    df.to_csv(csv_file, index=False)
    df.to_csv(tsv_file, sep="\t", index=False)
    df.to_parquet(parquet_file, index=False)
    
    assert get_dataset_columns(csv_file) == ["colA", "colB", "colC"]
    assert get_dataset_columns(tsv_file) == ["colA", "colB", "colC"]
    assert get_dataset_columns(parquet_file) == ["colA", "colB", "colC"]
    
    # Test non-existent file
    assert get_dataset_columns(str(temp_workspace / "invalid.csv")) == []


def test_preview_dataset_sample(temp_workspace):
    """Verifies dataset preview returns a DataFrame with limited rows."""
    csv_file = str(temp_workspace / "sample.csv")
    df = pd.DataFrame({"A": range(20), "B": range(20, 40)})
    df.to_csv(csv_file, index=False)
    
    preview = preview_dataset_sample(csv_file, nrows=5)
    assert preview is not None
    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 5
    assert list(preview.columns) == ["A", "B"]
    
    # Test non-existent file
    assert preview_dataset_sample(str(temp_workspace / "missing.csv")) is None


def test_is_valid_image(temp_workspace):
    """Verifies image validation helper correctly detects valid, corrupted, and empty files."""
    valid_img_path = str(temp_workspace / "valid.png")
    img = Image.new("RGB", (30, 30), color="blue")
    img.save(valid_img_path)
    
    assert is_valid_image(valid_img_path) is True
    
    # Empty file
    empty_file = str(temp_workspace / "empty.png")
    with open(empty_file, "w") as f:
        pass
    assert is_valid_image(empty_file) is False
    
    # Corrupted file
    corrupt_file = str(temp_workspace / "corrupt.png")
    with open(corrupt_file, "wb") as f:
        f.write(b"not an image byte sequence")
    assert is_valid_image(corrupt_file) is False
    
    # Non-existent file
    assert is_valid_image(str(temp_workspace / "missing.png")) is False
