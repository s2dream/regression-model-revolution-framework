import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ABCDataLoader(ABC):
    """
    Abstract Base Class for DataLoader strategies.
    Responsible for fetching/loading raw Features (X) and Target (y) from a data source.
    """
    def __init__(self, data_dir: str = "data", feature_columns: Optional[List[str]] = None, ignored_columns: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.feature_columns = feature_columns
        self.ignored_columns = ignored_columns
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
    def __init__(self, filepath: str, target_column: str, data_dir: str = "data", feature_columns: Optional[List[str]] = None, ignored_columns: Optional[List[str]] = None):
        super().__init__(data_dir=data_dir, feature_columns=feature_columns, ignored_columns=ignored_columns)
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
        elif self.filepath.endswith('.jsonl'):
            import json
            records = []
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            raise ValueError(f"Unsupported file format for {self.filepath}")

        if self.ignored_columns:
            cols_to_drop = [col for col in self.ignored_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset columns: {list(df.columns)}")

        y = df[self.target_column]

        if self.feature_columns:
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Specified feature columns {missing_cols} not found in dataset columns: {list(df.columns)}")
            X = df[self.feature_columns]
        else:
            X = df.drop(columns=[self.target_column])
            
        return X, y


# class XFilelDataLoader(LocalFileDataLoader):
#     def __init__(self, filepath: str, target_column:str, data_dir:str="data", feature_columns: Optional[List[str]] = None):
#         super().__init__(filepath=filepath, target_column=target_column, data_dir=data_dir, feature_columns=feature_columns)

#     def load_data(self):
#         loaded_X, loaded_y = super().load_data()
#         # do something for loaded_X and loaded_y
#         return loaded_X, loaded_y


