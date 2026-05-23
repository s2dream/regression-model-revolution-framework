from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np

class ABCModelWrapper(ABC):
    """
    Abstract base class defining the unified interface for model wrappers in the pool.
    """
    def __init__(self, name: str, model_instance: Any):
        self.name = name
        self.model = model_instance

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ABCModelWrapper':
        """
        Fits the underlying model on the provided data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            ABCModelWrapper: The fitted wrapper instance
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts values using the underlying model on the provided features.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted values
        """
        pass


class ModelWrapper(ABCModelWrapper):
    """
    Standard generic wrapper for models conforming to Scikit-Learn style fit/predict interface.
    Serves as a fallback for custom or generic models.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapper':
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)


class ModelWrapperXGBoost(ABCModelWrapper):
    """
    Dedicated wrapper for XGBoost Regressor.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperXGBoost':
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)


class ModelWrapperMLP(ABCModelWrapper):
    """
    Dedicated wrapper for Multi-Layer Perceptron (MLP) Regressor.
    Automatically handles feature scaling via StandardScaler to improve neural network training and convergence.
    """
    def __init__(self, name: str, model_instance: Any):
        super().__init__(name, model_instance)
        # Import dynamically to avoid scikit-learn hard requirement at module-level
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperMLP':
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y
        
        # MLPs are highly sensitive to feature scaling; automatically fit and transform features
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict(X_scaled)


class ModelWrapperTabPFN(ABCModelWrapper):
    """
    Dedicated wrapper for TabPFN Regressor.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperTabPFN':
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        y_arr = y.to_numpy(dtype=np.float32) if isinstance(y, pd.Series) else np.array(y, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        return self.model.predict(X_arr)


class ModelWrapperRandomForest(ABCModelWrapper):
    """
    Dedicated wrapper for RandomForest Regressor baseline.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperRandomForest':
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)


class ModelWrapperCatBoost(ABCModelWrapper):
    """
    Dedicated wrapper for CatBoost Regressor.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperCatBoost':
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)
