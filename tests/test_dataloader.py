import os
import pandas as pd
import numpy as np
import pytest
from automl_framework.dataloader import (
    DataLoaderHelper,
    LocalFileDataLoader,
    StandardDataPreprocessor,
    TrainTestSplitter
)

@pytest.fixture
def dummy_csv_path(tmp_path):
    """Fixture to generate a temporary CSV file with categorical, numerical, and missing values."""
    df = pd.DataFrame({
        "Feature_Num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "Feature_Cat": ["A", "B", "A", np.nan, "B"],
        "Target_Y": [10, 20, 30, 40, 50]
    })
    filepath = tmp_path / "dummy_dataset.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_local_file_data_loader(dummy_csv_path):
    """Test LocalFileDataLoader for successfully reading local files and separating features/target."""
    loader = LocalFileDataLoader(filepath=dummy_csv_path, target_column="Target_Y")
    X, y = loader.load_data()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (5, 2)
    assert y.shape == (5,)
    assert list(X.columns) == ["Feature_Num", "Feature_Cat"]
    assert list(y.values) == [10, 20, 30, 40, 50]


def test_standard_data_preprocessor():
    """Test StandardDataPreprocessor for correct missing value imputation and categorical dummy encoding."""
    df_raw = pd.DataFrame({
        "Num_Col": [1.0, 2.0, np.nan, 4.0],
        "Cat_Col": ["X", "Y", "X", np.nan]
    })
    
    preprocessor = StandardDataPreprocessor()
    df_proc = preprocessor.preprocess(df_raw)
    
    # Assert missing values are imputed
    # Median of [1.0, 2.0, 4.0] is 2.0
    assert df_proc["Num_Col"].isnull().sum() == 0
    assert df_proc["Num_Col"].iloc[2] == 2.0
    
    # Imputed mode of Cat_Col is "X"
    # Categorical column Cat_Col should be dummy encoded (drop_first=True, leaving Cat_Col_Y)
    # The columns should be Num_Col and Cat_Col_Y
    assert "Cat_Col_Y" in df_proc.columns
    assert "Cat_Col_X" not in df_proc.columns
    assert df_proc["Cat_Col_Y"].isnull().sum() == 0


def test_train_test_splitter():
    """Test TrainTestSplitter for correct dataset partitions and reproducible seed behavior."""
    X = pd.DataFrame({"Feature": np.arange(10)})
    y = pd.Series(np.arange(10))
    
    splitter = TrainTestSplitter()
    
    # Test split without validation
    splits = splitter.split(X, y, test_size=0.2, random_state=42)
    assert "X_train" in splits
    assert "X_test" in splits
    assert "X_val" not in splits
    assert splits["X_train"].shape == (8, 1)
    assert splits["X_test"].shape == (2, 1)
    
    # Test split with validation
    splits_val = splitter.split(X, y, test_size=0.2, val_size=0.2, random_state=42)
    assert "X_train" in splits_val
    assert "X_test" in splits_val
    assert "X_val" in splits_val
    assert splits_val["X_train"].shape == (6, 1)
    assert splits_val["X_val"].shape == (2, 1)
    assert splits_val["X_test"].shape == (2, 1)


def test_data_loader_facade(dummy_csv_path):
    """Test DataLoader facade backward compatibility matches the original interface expectations."""
    facade = DataLoaderHelper()
    X, y = facade.load_dataset(dummy_csv_path, target_column="Target_Y")
    
    assert X.shape == (5, 2)
    assert y.shape == (5,)
    
    X_proc = facade.preprocess_data(X)
    assert X_proc.isnull().sum().sum() == 0
    
    splits = facade.split_data(X_proc, y, test_size=0.2, random_state=42)
    assert "X_train" in splits
    assert "X_test" in splits

def test_data_loader_helper_fetch_and_prepare(dummy_csv_path):
    """Test fetch_dataset and prepare_data methods on DataLoaderHelper."""
    facade = DataLoaderHelper()
    
    # Test fetch_dataset with local path
    resolved_path = facade.fetch_dataset(dataset_path=dummy_csv_path)
    assert resolved_path == dummy_csv_path
    
    # Test prepare_data direct facade invocation
    X_train, y_train, X_test, y_test = facade.prepare_data(
        dataset_file=dummy_csv_path,
        target_column="Target_Y",
        test_size=0.2,
        random_state=42
    )
    
    assert X_train.shape[1] == 2
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1

