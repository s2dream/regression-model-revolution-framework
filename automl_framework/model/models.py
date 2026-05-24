import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from abc import ABC, abstractmethod
from automl_framework.model.wrappers import (
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
    Acts purely as an inventory repository/container of estimators, bound dynamically by configurations.
    """
    def __init__(self, random_state: int = 42, config: Optional[Dict[str, Any]] = None):
        self.random_state = random_state
        self.config = config or {}
        self.models: Dict[str, ABCModelWrapper] = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initializes the suite of default models dynamically from YAML configurations."""
        active_models = self.config.get("framework", {}).get("active_models", [
            "XGBoost", "MLP", "TabPFN", "RandomForest", "CatBoost"
        ])
        
        # 1. XGBoost Regressor
        if "XGBoost" in active_models:
            try:
                from xgboost import XGBRegressor
                xgb_params = self.config.get("models", {}).get("XGBoost", {}).copy()
                if "random_state" not in xgb_params:
                    xgb_params["random_state"] = self.random_state
                xgb_model = XGBRegressor(**xgb_params)
                self.models["XGBoost"] = ModelWrapperXGBoost("XGBoost", xgb_model)
                print("[ModelPool] XGBoost initialized successfully from config.")
            except Exception as e:
                print(f"[ModelPool] WARNING: XGBoost could not be initialized. XGBoost will be unavailable. Error: {e}")

        # 2. Multi-Layer Perceptron (MLP)
        if "MLP" in active_models:
            try:
                from sklearn.neural_network import MLPRegressor
                mlp_params = self.config.get("models", {}).get("MLP", {}).copy()
                
                # Cast list hidden layers config from YAML into python tuple safely
                if "hidden_layer_sizes" in mlp_params:
                    mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
                if "random_state" not in mlp_params:
                    mlp_params["random_state"] = self.random_state
                
                # Safe merge with core defaults
                default_mlp_params = {
                    "hidden_layer_sizes": (128, 64),
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 500,
                    "random_state": self.random_state
                }
                merged_mlp_params = {**default_mlp_params, **mlp_params}
                
                mlp_model = MLPRegressor(**merged_mlp_params)
                self.models["MLP"] = ModelWrapperMLP("MLP", mlp_model)
                print("[ModelPool] MLP Regressor initialized successfully from config.")
            except Exception as e:
                print(f"[ModelPool] WARNING: MLP could not be initialized. Error: {e}")

        # 3. TabPFN Regressor (Prior-Data Fitted Network)
        if "TabPFN" in active_models:
            try:
                from tabpfn import TabPFNRegressor
                tabpfn_params = self.config.get("models", {}).get("TabPFN", {}).copy()
                if "random_state" not in tabpfn_params:
                    tabpfn_params["random_state"] = self.random_state
                tabpfn_model = TabPFNRegressor(**tabpfn_params)
                self.models["TabPFN"] = ModelWrapperTabPFN("TabPFN", tabpfn_model)
                print("[ModelPool] TabPFN Regressor initialized successfully from config.")
            except Exception as e:
                try:
                    from tabpfn.scripts.transformer_prediction_interface import TabPFNRegressor
                    tabpfn_params = self.config.get("models", {}).get("TabPFN", {}).copy()
                    if "random_state" not in tabpfn_params:
                        tabpfn_params["random_state"] = self.random_state
                    tabpfn_model = TabPFNRegressor(**tabpfn_params)
                    self.models["TabPFN"] = ModelWrapperTabPFN("TabPFN", tabpfn_model)
                    print("[ModelPool] TabPFN Regressor initialized successfully (fallback import).")
                except Exception as fallback_e:
                    print(f"[ModelPool] WARNING: TabPFN could not be initialized. TabPFN will be unavailable. Error: {fallback_e}")

        # 4. Standard baseline: Random Forest (from scikit-learn)
        if "RandomForest" in active_models:
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf_params = self.config.get("models", {}).get("RandomForest", {}).copy()
                if "random_state" not in rf_params:
                    rf_params["random_state"] = self.random_state
                rf_model = RandomForestRegressor(**rf_params)
                self.models["RandomForest"] = ModelWrapperRandomForest("RandomForest", rf_model)
                print("[ModelPool] RandomForest Baseline initialized successfully from config.")
            except Exception as e:
                print(f"[ModelPool] WARNING: RandomForest could not be initialized. Error: {e}")

        # 5. CatBoost Regressor
        if "CatBoost" in active_models:
            try:
                from catboost import CatBoostRegressor
                cat_params = self.config.get("models", {}).get("CatBoost", {}).copy()
                if "random_seed" not in cat_params:
                    cat_params["random_seed"] = self.random_state
                cat_model = CatBoostRegressor(**cat_params)
                self.models["CatBoost"] = ModelWrapperCatBoost("CatBoost", cat_model)
                print("[ModelPool] CatBoost initialized successfully from config.")
            except Exception as e:
                print(f"[ModelPool] WARNING: CatBoost could not be initialized. CatBoost will be unavailable. Error: {e}")

    def add_custom_model(self, name: str, model_instance: Any):
        """Allows adding any custom estimator that conforms to the fit/predict interface."""
        if isinstance(model_instance, ABCModelWrapper):
            self.models[name] = model_instance
        else:
            self.models[name] = ModelWrapper(name, model_instance)
        print(f"[ModelPool] Custom model '{name}' added successfully.")

    def get_model(self, name: str) -> Optional[ABCModelWrapper]:
        """Retrieves a specific wrapped model from the pool."""
        return self.models.get(name)

    def list_available_models(self) -> List[str]:
        """Returns the list of active models in the pool."""
        return list(self.models.keys())


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
