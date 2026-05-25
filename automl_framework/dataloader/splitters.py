from typing import Dict, Any, Optional
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from automl_framework.dataloader.base import ABCDataSplitter

logger = logging.getLogger(__name__)

class TrainTestSplitter(ABCDataSplitter):
    """
    Concrete Dataset Splitter strategy.
    Splits features and target into train, validation (optional), and test splits.
    """
    def split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: Optional[float] = None, random_state: int = 42) -> Dict[str, Any]:
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
