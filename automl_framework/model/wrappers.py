from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ABCModelWrapper(ABC):
    """
    Abstract base class defining the unified interface for model wrappers in the pool.
    """
    def __init__(self, name: str, model_instance: Any):
        self.name = name
        self.model = model_instance

    def get_loss_history(self) -> list:
        """
        Returns the training loss history if available.
        """
        return []

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
        self.model.fit(X_arr, y_arr, eval_set=[(X_arr, y_arr)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)

    def get_loss_history(self) -> list:
        try:
            evals = self.model.evals_result()
            metric_key = list(evals['validation_0'].keys())[0]
            return list(evals['validation_0'][metric_key])
        except Exception:
            return []


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

    def get_loss_history(self) -> list:
        try:
            return list(self.model.loss_curve_)
        except Exception:
            return []

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

    def get_loss_history(self) -> list:
        try:
            evals = self.model.get_evals_result()
            metric_key = list(evals['learn'].keys())[0]
            return list(evals['learn'][metric_key])
        except Exception:
            return []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)


class ModelWrapperTransformer(ABCModelWrapper):
    """
    Dedicated wrapper for TransformerBasedRegression model in PyTorch.
    Supports standard fit/predict workflow, automatic sequence reshaping,
    and handles training loops with support for probabilistic outputs.
    """
    def __init__(
        self,
        name: str,
        model_instance: Any,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: bool = True
    ):
        super().__init__(name, model_instance)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_history = []
        
        # Optimizer Setup
        import torch.optim as optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def _prepare_input(self, X: Any):
        import torch
        # Convert X to numpy
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.array(X)
            
        X_arr = X_arr.astype(np.float32)
        
        # If 2D (batch_size, num_features), reshape to 3D (batch_size, num_features, 1)
        # to treat features as sequence steps of dimension 1.
        if len(X_arr.shape) == 2:
            if X_arr.shape[1] == self.model.encoder.input_projection.in_features:
                X_arr = np.expand_dims(X_arr, axis=1)  # (batch, 1, num_features)
            else:
                X_arr = np.expand_dims(X_arr, axis=-1) # (batch, num_features, 1)
                
        return torch.tensor(X_arr)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperTransformer':
        import torch
        import torch.nn.functional as F
        
        X_tensor = self._prepare_input(X)
        
        if hasattr(y, "to_numpy"):
            y_arr = y.to_numpy()
        else:
            y_arr = np.array(y)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(-1)
        
        self.model.train()
        dataset_size = X_tensor.size(0)
        self.loss_history = []
        
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, dataset_size, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                self.optimizer.zero_grad()
                
                if self.model.predict_distribution:
                    mean, variance = self.model(batch_x)
                    # Gaussian NLL loss
                    nll_loss = 0.5 * torch.log(variance) + 0.5 * ((batch_y - mean) ** 2) / (variance + 1e-8)
                    loss = nll_loss.mean()
                else:
                    pred = self.model(batch_x)
                    loss = F.mse_loss(pred, batch_y)
                    
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            self.loss_history.append(epoch_loss / max(1, num_batches))
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/num_batches:.6f}")
                
        return self

    def get_loss_history(self) -> list:
        return self.loss_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self.model.eval()
        X_tensor = self._prepare_input(X)
        
        with torch.no_grad():
            if self.model.predict_distribution:
                mean, _ = self.model(X_tensor)
                return mean.squeeze(-1).numpy()
            else:
                pred = self.model(X_tensor)
                return pred.squeeze(-1).numpy()


class ModelWrapperTabICL(ABCModelWrapper):
    """
    Dedicated wrapper for TabICL (Tabular In-Context Learning) Regressor.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapperTabICL':
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        y_arr = y.to_numpy(dtype=np.float32) if isinstance(y, pd.Series) else np.array(y, dtype=np.float32)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=np.float32).reshape(-1)

