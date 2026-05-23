import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from automl_framework.core.wrappers import (
    ABCModelWrapper,
    ModelWrapper,
    ModelWrapperXGBoost,
    ModelWrapperMLP,
    ModelWrapperTabPFN,
    ModelWrapperRandomForest,
    ModelWrapperCatBoost,
)


class ModelPool:
    """
    ModelPool manages a suite of regression models (e.g., XGBoost, MLP, TabPFN, and standard Baselines).
    Supports initialization, fitting, validation, and retrieval of individual models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, ABCModelWrapper] = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initializes the suite of default models in the pool with reasonable hyperparameters."""
        
        # 1. XGBoost Regressor
        try:
            from xgboost import XGBRegressor
            xgb_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.models["XGBoost"] = ModelWrapperXGBoost("XGBoost", xgb_model)
            print("[ModelPool] XGBoost initialized successfully.")
        except ImportError:
            print("[ModelPool] WARNING: 'xgboost' is not installed. XGBoost will be unavailable.")

        # 2. Multi-Layer Perceptron (MLP)
        try:
            from sklearn.neural_network import MLPRegressor
            mlp_model = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=self.random_state
            )
            self.models["MLP"] = ModelWrapperMLP("MLP", mlp_model)
            print("[ModelPool] MLP Regressor initialized successfully.")
        except ImportError:
            print("[ModelPool] WARNING: 'scikit-learn' is required for MLP.")

        # 3. TabPFN Regressor (Prior-Data Fitted Network)
        # Note: TabPFN is highly optimized for small-to-medium tabular datasets.
        try:
            # We try importing from tabpfn. In newer versions, it might be TabPFNRegressor
            from tabpfn import TabPFNRegressor
            tabpfn_model = TabPFNRegressor(random_state=self.random_state)
            self.models["TabPFN"] = ModelWrapperTabPFN("TabPFN", tabpfn_model)
            print("[ModelPool] TabPFN Regressor initialized successfully.")
        except ImportError:
            try:
                # Fallback or alternative import path if any
                from tabpfn.scripts.transformer_prediction_interface import TabPFNRegressor
                tabpfn_model = TabPFNRegressor(random_state=self.random_state)
                self.models["TabPFN"] = ModelWrapperTabPFN("TabPFN", tabpfn_model)
                print("[ModelPool] TabPFN Regressor initialized successfully (fallback import).")
            except ImportError:
                print("[ModelPool] WARNING: 'tabpfn' is not installed. TabPFN will be unavailable.")

        # 4. Standard baseline: Random Forest (from scikit-learn)
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            self.models["RandomForest"] = ModelWrapperRandomForest("RandomForest", rf_model)
            print("[ModelPool] RandomForest Baseline initialized successfully.")
        except ImportError:
            pass

        # 5. CatBoost Regressor
        try:
            from catboost import CatBoostRegressor
            cat_model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=self.random_state,
                verbose=0
            )
            self.models["CatBoost"] = ModelWrapperCatBoost("CatBoost", cat_model)
            print("[ModelPool] CatBoost initialized successfully.")
        except ImportError:
            print("[ModelPool] WARNING: 'catboost' is not installed. CatBoost will be unavailable.")

    def add_custom_model(self, name: str, model_instance: Any):
        """Allows adding any custom estimator that conforms to the fit/predict interface."""
        if isinstance(model_instance, ABCModelWrapper):
            self.models[name] = model_instance
        else:
            self.models[name] = ModelWrapper(name, model_instance)
        print(f"[ModelPool] Custom model '{name}' added successfully.")

    def fit_all(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits all models in the pool on the training data."""
        for name, model_wrap in self.models.items():
            print(f"[ModelPool] Training model: {name}...")
            try:
                model_wrap.fit(X_train, y_train)
                print(f"[ModelPool] Finished training: {name}")
            except Exception as e:
                print(f"[ModelPool] ERROR training {name}: {e}")

    def evaluate_all(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluates all trained models in the pool on the given dataset.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping model names to metrics (RMSE, MAE, R2)
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        results = {}
        for name, model_wrap in self.models.items():
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
                print(f"[ModelPool] ERROR evaluating {name}: {e}")
                
        return results

    def get_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Gets predictions from all trained models in the pool."""
        predictions = {}
        for name, model_wrap in self.models.items():
            try:
                predictions[name] = model_wrap.predict(X)
            except Exception as e:
                print(f"[ModelPool] ERROR getting predictions for {name}: {e}")
        return predictions

    def get_model(self, name: str) -> Optional[ABCModelWrapper]:
        """Retrieves a specific wrapped model from the pool."""
        return self.models.get(name)

    def list_available_models(self) -> List[str]:
        """Returns the list of active models in the pool."""
        return list(self.models.keys())
