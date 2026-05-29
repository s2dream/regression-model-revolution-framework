from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

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


class TrainTestSplitter(ABCDataSplitter):
    """
    Concrete Dataset Splitter strategy.
    Splits features and target into train, validation (optional), and test splits.
    """
    def split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: Optional[float] = None, random_state: int = 42, **kwargs) -> Dict[str, Any]:
        logger.info(f"Splitting data (test_size={test_size}, val_size={val_size}, random_state={random_state})...")
        
        if val_size:
            # Adjust test_size to account for the first split
            first_split_size = test_size + val_size
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=first_split_size, random_state=random_state
            )
            
            # Split the temporary set into validation and test
            val_relative_size = val_size / first_split_size
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_relative_size), random_state=random_state
            )
            
            return {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test
            }


class KFoldSplitter(ABCDataSplitter):
    """
    Concrete Dataset Splitter strategy for KFold Cross Validation.
    Supports single fold extraction for direct training compatibility.
    """
    def split(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42, **kwargs) -> Dict[str, Any]:
        logger.info(f"Splitting data using KFold (n_splits={n_splits}, shuffle={shuffle}, random_state={random_state})...")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        # Extract the first fold to comply with train/test signature requirements
        train_idx, test_idx = next(kf.split(X, y))
        return {
            "X_train": X.iloc[train_idx],
            "y_train": y.iloc[train_idx],
            "X_test": X.iloc[test_idx],
            "y_test": y.iloc[test_idx]
        }


class TimeSeriesSplitter(ABCDataSplitter):
    """
    Concrete Dataset Splitter strategy for TimeSeriesSplit.
    Ideal for sequential or time-series regression datasets.
    """
    def split(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, **kwargs) -> Dict[str, Any]:
        logger.info(f"Splitting data using TimeSeriesSplit (n_splits={n_splits})...")
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Extract the last fold (the most recent historical train and subsequent test split)
        folds = list(tscv.split(X, y))
        train_idx, test_idx = folds[-1]
        return {
            "X_train": X.iloc[train_idx],
            "y_train": y.iloc[train_idx],
            "X_test": X.iloc[test_idx],
            "y_test": y.iloc[test_idx]
        }

