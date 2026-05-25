import os
from typing import Tuple
import pandas as pd
from automl_framework.dataloader.base import ABCDataLoader

class LocalFileDataLoader(ABCDataLoader):
    """
    Concrete DataLoader strategy to load CSV/TSV/Parquet files directly from the local file system.
    """
    def __init__(self, filepath: str, target_column: str, data_dir: str = "data"):
        super().__init__(data_dir=data_dir)
        self.filepath = filepath
        self.target_column = target_column

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"[LocalFileDataLoader] Loading data from {self.filepath}...")
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
        print(f"[KaggleDataLoader] Attempting to download dataset '{self.dataset_name}' from Kaggle...")
        try:
            import kaggle
            kaggle.api.authenticate()
            target_path = os.path.join(self.data_dir, self.dataset_name.replace("/", "_"))
            os.makedirs(target_path, exist_ok=True)            
            kaggle.api.dataset_download_files(self.dataset_name, path=target_path, unzip=True)
            print(f"[KaggleDataLoader] Successfully downloaded and extracted to {target_path}")
            return target_path
        except ImportError:
            print("[KaggleDataLoader] ERROR: 'kaggle' package is not installed. Please install it using 'pip install kaggle'.")
            raise
        except Exception as e:
            print(f"[KaggleDataLoader] ERROR: Failed to download from Kaggle: {e}")
            print("[KaggleDataLoader] Make sure kaggle.json is configured in ~/.kaggle/ or equivalent location.")
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
            print(f"[URLDataLoader] File already exists at {target_path}. Skipping download.")
            return target_path
            
        print(f"[URLDataLoader] Downloading from {self.url} to {target_path}...")
        try:
            urllib.request.urlretrieve(self.url, target_path)
            print("[URLDataLoader] Download complete.")
            return target_path
        except Exception as e:
            print(f"[URLDataLoader] ERROR: Failed to download from URL: {e}")
            raise

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        filepath = self.download_data()
        local_loader = LocalFileDataLoader(filepath=filepath, target_column=self.target_column, data_dir=self.data_dir)
        return local_loader.load_data()
