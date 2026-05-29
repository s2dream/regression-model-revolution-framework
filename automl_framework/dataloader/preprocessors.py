from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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


class StandardDataPreprocessor(ABCDataPreprocessor):
    """
    Concrete Data Preprocessor strategy.
    Performs basic preprocessing such as handling missing values (median for numerical,
    mode for categorical) and one-hot encoding for categorical features.
    """
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing raw features...")
        X_processed = X.copy()
        
        # Handle numerical columns: simple median imputation
        num_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if X_processed[col].isnull().any():
                logger.debug(f"Imputing missing values in numerical column '{col}' with median.")
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                
        # Handle categorical columns: simple frequency imputation & dummy encoding
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if X_processed[col].isnull().any():
                logger.debug(f"Imputing missing values in categorical column '{col}' with mode.")
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
        
        if len(cat_cols) > 0:
            logger.info(f"One-hot encoding categorical features: {list(cat_cols)}")
            X_processed = pd.get_dummies(X_processed, columns=cat_cols, drop_first=True)
            
        return X_processed


class NoOpDataPreprocessor(ABCDataPreprocessor):
    """
    Concrete Data Preprocessor strategy that returns the features as-is without any modifications.
    """
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
