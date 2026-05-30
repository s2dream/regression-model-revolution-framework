import os
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

from automl_framework.dataloader.loaders import LocalFileDataLoader, KaggleDataLoader, URLDataLoader
from automl_framework.dataloader.preprocessors import StandardDataPreprocessor
from automl_framework.dataloader.splitters import TrainTestSplitter, KFoldSplitter, TimeSeriesSplitter

class DataLoaderHelper:
    """
    DataLoader facade class. Maintains backwards compatibility with the original DataLoader interface,
    delegating logic under-the-hood to modular strategy classes.
    """
    def __init__(self, data_dir: str = "data", config: Optional[Dict[str, Any]] = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.feature_columns = self.config.get("data", {}).get("feature_columns", None)
        self.preprocessor = StandardDataPreprocessor()
        self.splitter = self._resolve_splitter()

    def _resolve_splitter(self):
        split_config = self.config.get("data", {}).get("split", {})
        method = split_config.get("method", "train_test_split").lower()
        if method == "kfold":
            return KFoldSplitter()
        elif method == "timeseries":
            return TimeSeriesSplitter()
        else:
            return TrainTestSplitter()

    def download_from_kaggle(self, dataset_name: str) -> str:
        """
        Downloads a dataset from Kaggle using KaggleDataLoader.
        """
        kaggler_dir = os.path.join(self.data_dir, "kaggle_dataset")
        loader = KaggleDataLoader(dataset_name=dataset_name, target_column="", data_dir=kaggler_dir)
        return loader.load_data()

    def download_from_url(self, url: str, filename: str) -> str:
        """
        Downloads a dataset from URL using URLDataLoader.
        """
        loader = URLDataLoader(url=url, filename=filename, target_column="", data_dir=self.data_dir)
        return loader.load_data()

    def load_dataset(self, filepath: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads dataset using LocalFileDataLoader.
        """
        loader = LocalFileDataLoader(filepath=filepath, target_column=target_column, feature_columns=self.feature_columns, data_dir=self.data_dir)
        return loader.load_data()

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses raw features using StandardDataPreprocessor.
        """
        return self.preprocessor.preprocess(X)

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: Optional[float] = None, random_state: int = 42) -> Dict[str, Any]:
        """
        Splits data using TrainTestSplitter.
        """
        return self.splitter.split(X, y, test_size=test_size, val_size=val_size, random_state=random_state)

    def load_and_preprocess_data(self, dataset_file: str, target_column: str, test_size: Optional[float] = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load, preprocess, and split dataset into train and test sets.
        """
        X, y = self.load_dataset(dataset_file, target_column=target_column)
        X_processed = self.preprocess_data(X)
        
        # Split Data (read dynamically from config split block)
        split_config = self.config.get("data", {}).get("split", {})
        split_kwargs = {
            "test_size": test_size if test_size is not None else split_config.get("test_size", 0.2),
            "val_size": split_config.get("val_size", None),
            "n_splits": split_config.get("n_splits", 5),
            "shuffle": split_config.get("shuffle", True),
            "random_state": random_state
        }
        
        splits = self.splitter.split(X_processed, y, **split_kwargs)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_test, y_test = splits["X_test"], splits["y_test"]
        
        logger.info("Data shape summary:")
        logger.info(f"  - Features dimension: {X_processed.shape[1]}")
        logger.info(f"  - Training samples: {X_train.shape[0]}")
        logger.info(f"  - Testing samples:  {X_test.shape[0]}")
        
        return X_train, y_train, X_test, y_test

    def fetch_dataset(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: Optional[str] = None,
        url: Optional[str] = None
    ) -> str:
        """
        Kaggle, URL, 혹은 로컬 경로로부터 데이터셋을 안전하게 다운로드하거나 
        유효성 검사를 거쳐 최종 데이터셋 파일의 절대 경로를 반환합니다.
        """
        dataset_file = dataset_path
        
        if kaggle_dataset:
            download_dir = self.download_from_kaggle(kaggle_dataset)
            csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
            if csv_files:
                dataset_file = os.path.join(download_dir, csv_files[0])
            else:
                raise FileNotFoundError(f"Kaggle에서 다운로드한 폴더 내에 CSV 파일이 존재하지 않습니다: {download_dir}")
                
        elif url:
            # URL을 통해 downloaded_data.csv로 다운로드
            dataset_file = self.download_from_url(url, "downloaded_data.csv")

        # 최종 경로 검증
        if not dataset_file:
            raise ValueError(
                "사용 가능한 데이터셋 정보가 주어지지 않았습니다. "
                "local path (--dataset-path), Kaggle dataset (--kaggle-dataset), 또는 UCI URL (--url) 중 하나를 지정해야 합니다."
            )
            
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"지정한 데이터셋 경로가 로컬에 존재하지 않습니다: {dataset_file}")
            
        return dataset_file
