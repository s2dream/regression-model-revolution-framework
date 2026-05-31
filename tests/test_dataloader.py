import os
import pandas as pd
import numpy as np
import pytest
from automl_framework.dataloader import (
    DataLoaderHelper,
    LocalFileDataLoader,
    StandardDataPreprocessor,
    NoOpDataPreprocessor,
    TrainTestSplitter,
    KFoldSplitter,
    TimeSeriesSplitter
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


def test_local_file_data_loader_custom_features(dummy_csv_path):
    """Test LocalFileDataLoader for correctly picking custom feature columns and raising errors on missing ones."""
    # Test valid selection
    loader = LocalFileDataLoader(filepath=dummy_csv_path, target_column="Target_Y", feature_columns=["Feature_Num"])
    X, y = loader.load_data()
    assert X.shape == (5, 1)
    assert list(X.columns) == ["Feature_Num"]
    
    # Test invalid selection (raises ValueError)
    loader_invalid = LocalFileDataLoader(filepath=dummy_csv_path, target_column="Target_Y", feature_columns=["Non_Existent"])
    with pytest.raises(ValueError, match="Specified feature columns .* not found"):
        loader_invalid.load_data()


def test_kfold_splitter():
    """Test KFoldSplitter partitions and compatibility."""
    X = pd.DataFrame({"Feature": np.arange(10)})
    y = pd.Series(np.arange(10))
    
    splitter = KFoldSplitter()
    splits = splitter.split(X, y, n_splits=5, shuffle=True, random_state=42)
    
    assert "X_train" in splits
    assert "X_test" in splits
    assert splits["X_train"].shape == (8, 1)
    assert splits["X_test"].shape == (2, 1)


def test_timeseries_splitter():
    """Test TimeSeriesSplitter sequential behavior."""
    X = pd.DataFrame({"Feature": np.arange(10)})
    y = pd.Series(np.arange(10))
    
    splitter = TimeSeriesSplitter()
    splits = splitter.split(X, y, n_splits=5)
    
    assert "X_train" in splits
    assert "X_test" in splits
    # For 10 samples and 5 splits: TimeSeriesSplit(n_splits=5) last split has
    # train size = 9, test size = 1 (standard TimeSeriesSplit index splits)
    assert splits["X_train"].shape == (9, 1)
    assert splits["X_test"].shape == (1, 1)


def test_dataloader_helper_dynamic_split(dummy_csv_path):
    """Test DataLoaderHelper successfully resolves configuration for custom features and splitting strategies."""
    # Custom config for KFold & specific feature column
    config = {
        "data": {
            "feature_columns": ["Feature_Num"],
            "split": {
                "method": "kfold",
                "n_splits": 5,
                "shuffle": True
            }
        }
    }
    
    helper = DataLoaderHelper(config=config)
    X_train, y_train, X_test, y_test = helper.prepare_data(
        dataset_file=dummy_csv_path,
        target_column="Target_Y",
        random_state=42
    )
    
    # Feature columns should only be "Feature_Num" (1 column)
    assert X_train.shape[1] == 1
    assert "Feature_Num" in X_train.columns
    assert "Feature_Cat" not in X_train.columns
    
    # Check KFold splitter resolution
    assert isinstance(helper.splitter, KFoldSplitter)
    assert X_train.shape[0] == 4  # 4 out of 5 samples for training in first fold
    assert X_test.shape[0] == 1   # 1 out of 5 samples for test


def test_data_preprocessor_noop():
    """Test NoOpDataPreprocessor to ensure it does not modify features in any way."""
    df_raw = pd.DataFrame({
        "Num_Col": [1.0, np.nan, 3.0],
        "Cat_Col": ["A", "B", np.nan]
    })
    preprocessor = NoOpDataPreprocessor()
    df_proc = preprocessor.preprocess(df_raw)
    
    # Assert data is completely identical (including NaNs)
    pd.testing.assert_frame_equal(df_raw, df_proc)


def test_resolve_parameters_cli_overrides():
    """Test AutoMLPipeline correctly prioritizes constructor overrides over config profile settings."""
    from main import AutoMLPipeline
    
    # Instantiate with explicit overrides representing CLI inputs
    pipeline = AutoMLPipeline(
        config_path="configs/default.yml",
        turn=5,
        target="OVERRIDE_TARGET",
        test_size=0.45
    )
    
    # Constructor/CLI arguments must override YAML configs
    assert pipeline.target_column == "OVERRIDE_TARGET"
    assert pipeline.test_size == 0.45
    assert pipeline.turn == 5


def test_load_config_fallback(tmp_path):
    """Test load_config in AutoMLPipeline safely handles missing or corrupted configuration files."""
    from main import AutoMLPipeline
    
    # 1. Test missing file fallback
    resolved_missing = AutoMLPipeline._load_config("configs/non_existent_file.yml")
    assert resolved_missing == {}
    
    # 2. Test corrupted/invalid YAML file fallback
    corrupt_file = tmp_path / "corrupt_config.yml"
    with open(corrupt_file, "w") as f:
        f.write("framework:\n  random_state: [unbalanced brackets")
        
    resolved_corrupted = AutoMLPipeline._load_config(str(corrupt_file))
    assert resolved_corrupted == {}


@pytest.fixture
def dummy_jsonl_path(tmp_path):
    """Fixture to generate a temporary JSONL file with dynamically appearing keys."""
    # Row 1 has Feature_Num, Target_Y
    # Row 2 has Feature_Num, Feature_Cat, Target_Y (Feature_Cat is a new key)
    # Row 3 has Feature_Num, Target_Y (Feature_Cat is missing)
    import json
    lines = [
        {"Feature_Num": 1.0, "Target_Y": 10},
        {"Feature_Num": 2.0, "Feature_Cat": "B", "Target_Y": 20},
        {"Feature_Num": 3.0, "Target_Y": 30}
    ]
    filepath = tmp_path / "dummy_dataset.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return str(filepath)


def test_local_file_data_loader_jsonl(dummy_jsonl_path):
    """Test LocalFileDataLoader for successfully reading JSONL files with dynamic columns."""
    loader = LocalFileDataLoader(filepath=dummy_jsonl_path, target_column="Target_Y")
    X, y = loader.load_data()
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert set(X.columns) == {"Feature_Num", "Feature_Cat"}
    
    # Assert missing values (NaN) are handled correctly for row index 0 and 2
    assert pd.isna(X.loc[0, "Feature_Cat"])
    assert X.loc[1, "Feature_Cat"] == "B"
    assert pd.isna(X.loc[2, "Feature_Cat"])





