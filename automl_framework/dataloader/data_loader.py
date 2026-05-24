import os
import pandas as pd
from typing import Tuple, Optional, Dict, Any

from automl_framework.dataloader.loaders import LocalFileDataLoader, KaggleDataLoader, URLDataLoader
from automl_framework.dataloader.preprocessors import StandardDataPreprocessor
from automl_framework.dataloader.splitters import TrainTestSplitter

class DataLoader:
    """
    DataLoader facade class. Maintains backwards compatibility with the original DataLoader interface,
    delegating logic under-the-hood to modular strategy classes.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.preprocessor = StandardDataPreprocessor()
        self.splitter = TrainTestSplitter()

    def download_from_kaggle(self, dataset_name: str) -> str:
        """
        Downloads a dataset from Kaggle using KaggleDataLoader.
        """
        loader = KaggleDataLoader(dataset_name=dataset_name, target_column="", data_dir=self.data_dir)
        return loader.download_data()

    def download_from_url(self, url: str, filename: str) -> str:
        """
        Downloads a dataset from URL using URLDataLoader.
        """
        loader = URLDataLoader(url=url, filename=filename, target_column="", data_dir=self.data_dir)
        return loader.download_data()

    def load_dataset(self, filepath: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads dataset using LocalFileDataLoader.
        """
        loader = LocalFileDataLoader(filepath=filepath, target_column=target_column, data_dir=self.data_dir)
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
