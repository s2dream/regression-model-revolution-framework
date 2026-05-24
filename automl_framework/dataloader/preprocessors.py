import pandas as pd
from automl_framework.dataloader.base import ABCDataPreprocessor

class StandardDataPreprocessor(ABCDataPreprocessor):
    """
    Concrete Data Preprocessor strategy.
    Performs basic preprocessing such as handling missing values (median for numerical,
    mode for categorical) and one-hot encoding for categorical features.
    """
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        
        # Handle numerical columns: simple median imputation
        num_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                
        # Handle categorical columns: simple frequency imputation & dummy encoding
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
        
        if len(cat_cols) > 0:
            X_processed = pd.get_dummies(X_processed, columns=cat_cols, drop_first=True)
            
        return X_processed


class NoOpDataPreprocessor(ABCDataPreprocessor):
    """
    Concrete Data Preprocessor strategy that returns the features as-is without any modifications.
    """
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
