import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from automl_framework.model.models import ModelPool

class ABCModelExecutor(ABC):
    """
    ABCModelExecutor defines the interface for running training, evaluation, 
    and prediction pipelines on a ModelPool container.
    """
    @abstractmethod
    def fit_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    @abstractmethod
    def evaluate_all(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        pass

    @abstractmethod
    def get_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        pass


class StandardBenchmarkExecutor(ABCModelExecutor):
    """
    StandardBenchmarkExecutor implements the standard benchmarking execution strategy.
    It fits models individually, scores them, and generates target prediction lists.
    Includes robust exception shielding against model-specific linkage/runtime crashes.
    """
    def __init__(self, pool: ModelPool):
        self.pool = pool

    def fit_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits all models in the pool on the training data."""
        for name, model_wrap in self.pool.models.items():
            print(f"[BenchmarkExecutor] Training model: {name}...")
            try:
                model_wrap.fit(X_train, y_train)
                print(f"[BenchmarkExecutor] Finished training: {name}")
            except Exception as e:
                print(f"[BenchmarkExecutor] ERROR training {name}: {e}")

    def evaluate_all(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluates all trained models in the pool on the given dataset."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        results = {}
        for name, model_wrap in self.pool.models.items():
            try:
                preds = model_wrap.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                mae = mean_absolute_error(y_val, preds)
                r2 = r2_score(y_val, preds)
                
                results[name] = {
                    "RMSE": float(rmse),
                    "MAE": float(mae),
                    "R2": float(r2)
                }
            except Exception as e:
                print(f"[BenchmarkExecutor] ERROR evaluating {name}: {e}")
                
        return results

    def get_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Gets predictions from all trained models in the pool."""
        predictions = {}
        for name, model_wrap in self.pool.models.items():
            try:
                predictions[name] = model_wrap.predict(X)
            except Exception as e:
                print(f"[BenchmarkExecutor] ERROR getting predictions for {name}: {e}")
        return predictions
