import os
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ABCDataLoader(ABC):
    """
    Abstract Base Class for DataLoader strategies.
    Responsible for fetching/loading raw Features (X) and Target (y) from a data source.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads and returns raw Features (X) and Target (y).
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y)
        """
        pass


class LocalFileDataLoader(ABCDataLoader):
    """
    Concrete DataLoader strategy to load CSV/TSV/Parquet files directly from the local file system.
    """
    def __init__(self, filepath: str, target_column: str, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)
        self.filepath = filepath
        self.target_column = target_column

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Loading data from {self.filepath}...")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        if self.filepath.endswith('.csv'):
            df = pd.read_csv(self.filepath)
        elif self.filepath.endswith('.tsv') or self.filepath.endswith('.txt'):
            df = pd.read_csv(self.filepath, sep='\t')
        elif self.filepath.endswith('.parquet'):
            df = pd.read_parquet(self.filepath)
        else:
            raise ValueError(f"Unsupported file format for {self.filepath}")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset columns: {list(df.columns)}")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y


class XFielDataLoader(LocalFileDataLoader):
    def __init__(self, filepath: str, target_column:str, data_dir:str="data"):
        super().__init__(filepath=filepath, target_column=target_column, data_dir=data_dir)

    def load_data(self):
        loaded_X, loaded_y = super().load_data()
        # do something for loaded_X and loaded_y
        return loaded_X, loaded_y


class KaggleDataLoader(ABCDataLoader):
    """
    Concrete DataLoader strategy that downloads a dataset from Kaggle using the Kaggle API,
    and then loads it.
    """
    def __init__(self, dataset_name: str, target_column: str, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)
        self.dataset_name = dataset_name
        self.target_column = target_column

    def download_data(self) -> str:
        logger.info(f"Attempting to download dataset '{self.dataset_name}' from Kaggle...")
        try:
            import kaggle
            kaggle.api.authenticate()
            target_path = os.path.join(self.data_dir, self.dataset_name.replace("/", "_"))
            os.makedirs(target_path, exist_ok=True)            
            kaggle.api.dataset_download_files(self.dataset_name, path=target_path, unzip=True)
            logger.info(f"Successfully downloaded and extracted to {target_path}")
            return target_path
        except ImportError:
            logger.error("'kaggle' package is not installed. Please install it using 'pip install kaggle'.")
            raise
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}", exc_info=True)
            logger.error("Make sure kaggle.json is configured in ~/.kaggle/ or equivalent location.")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        download_dir = self.download_data()
        csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in downloaded Kaggle files at {download_dir}")
        
        filepath = os.path.join(download_dir, csv_files[0])
        local_loader = LocalFileDataLoader(filepath=filepath, target_column=self.target_column, data_dir=self.data_dir)
        return local_loader.load_data()


class URLDataLoader(ABCDataLoader):
    """
    Concrete DataLoader strategy that downloads a dataset from a direct URL (e.g., UCI repository),
    and then loads it.
    """
    def __init__(self, url: str, filename: str, target_column: str, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)
        self.url = url
        self.filename = filename
        self.target_column = target_column

    def download_data(self) -> str:
        import urllib.request
        target_path = os.path.join(self.data_dir, self.filename)
        if os.path.exists(target_path):
            logger.info(f"File already exists at {target_path}. Skipping download.")
            return target_path
            
        logger.info(f"Downloading from {self.url} to {target_path}...")
        try:
            urllib.request.urlretrieve(self.url, target_path)
            logger.info("Download complete.")
            return target_path
        except Exception as e:
            logger.error(f"Failed to download from URL: {e}", exc_info=True)
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        filepath = self.download_data()
        local_loader = LocalFileDataLoader(filepath=filepath, target_column=self.target_column, data_dir=self.data_dir)
        return local_loader.load_data()
