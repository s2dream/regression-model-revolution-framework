import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd

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


class ABCDataPreprocessor(ABC):
    """
    Abstract Base Class for Data Preprocessor strategies.
    Responsible for handling missing values, encoding categorical variables, etc.
    """
    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data preprocessing on raw features.
        
        Args:
            X (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        pass


class ABCDataSplitter(ABC):
    """
    Abstract Base Class for Dataset Splitter strategies.
    Responsible for partitioning datasets into train, validation (optional), and test splits.
    """
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Splits Features (X) and Target (y) into train/validation/test datasets.
        
        Args:
            X (pd.DataFrame): Preprocessed features
            y (pd.Series): Target variable
            **kwargs: Configuration settings (e.g. test_size, random_state)
            
        Returns:
            Dict[str, Any]: Dictionary containing data splits (e.g., X_train, y_train, etc.)
        """
        pass
