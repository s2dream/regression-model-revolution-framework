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
        
        # Registry of model builders mapping active names to their factory functions
        self._model_builders = {
            "XGBoost": self._build_xgboost,
            "MLP": self._build_mlp,
            "TabPFN": self._build_tabpfn,
            "RandomForest": self._build_random_forest,
            "CatBoost": self._build_catboost
        }
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initializes the suite of default models dynamically from YAML configurations."""
        active_models = self.config.get("framework", {}).get("active_models", [
            "XGBoost", "MLP", "TabPFN", "RandomForest", "CatBoost"
        ])
        
        for model_name in active_models:
            if model_name in self._model_builders:
                builder = self._model_builders[model_name]
                try:
                    wrapped_model = builder()
                    self.models[model_name] = wrapped_model
                    print(f"[ModelPool] {model_name} initialized successfully from config.")
                except Exception as e:
                    print(f"[ModelPool] WARNING: {model_name} could not be initialized. {model_name} will be unavailable. Error: {e}")
            else:
                print(f"[ModelPool] WARNING: Model '{model_name}' is not registered in the default builders.")

    def _build_xgboost(self) -> ABCModelWrapper:
        from xgboost import XGBRegressor
        xgb_params = self.config.get("models", {}).get("XGBoost", {}).copy()
        if "random_state" not in xgb_params:
            xgb_params["random_state"] = self.random_state
        xgb_model = XGBRegressor(**xgb_params)
        return ModelWrapperXGBoost("XGBoost", xgb_model)

    def _build_mlp(self) -> ABCModelWrapper:
        from sklearn.neural_network import MLPRegressor
        mlp_params = self.config.get("models", {}).get("MLP", {}).copy()
        
        if "hidden_layer_sizes" in mlp_params:
            mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
        if "random_state" not in mlp_params:
            mlp_params["random_state"] = self.random_state
        
        default_mlp_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "random_state": self.random_state
        }
        merged_mlp_params = {**default_mlp_params, **mlp_params}
        mlp_model = MLPRegressor(**merged_mlp_params)
        return ModelWrapperMLP("MLP", mlp_model)

    def _build_tabpfn(self) -> ABCModelWrapper:
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise Exception("import tabpfn package is required. Please run pip install tabpfn")
        
        tabpfn_params = self.config.get("models", {}).get("TabPFN", {}).copy()
        if "random_state" not in tabpfn_params:
            tabpfn_params["random_state"] = self.random_state
        
        # Modern TabPFN uses 'n_estimators'
        if "N_ensemble_configurations" in tabpfn_params and "n_estimators" not in tabpfn_params:
            tabpfn_params["n_estimators"] = tabpfn_params.pop("N_ensemble_configurations")
        
        tabpfn_model = TabPFNRegressor(**tabpfn_params)
        return ModelWrapperTabPFN("TabPFN", tabpfn_model)

    def _build_random_forest(self) -> ABCModelWrapper:
        from sklearn.ensemble import RandomForestRegressor
        rf_params = self.config.get("models", {}).get("RandomForest", {}).copy()
        if "random_state" not in rf_params:
            rf_params["random_state"] = self.random_state
        rf_model = RandomForestRegressor(**rf_params)
        return ModelWrapperRandomForest("RandomForest", rf_model)

    def _build_catboost(self) -> ABCModelWrapper:
        from catboost import CatBoostRegressor
        cat_params = self.config.get("models", {}).get("CatBoost", {}).copy()
        if "random_seed" not in cat_params:
            cat_params["random_seed"] = self.random_state
        cat_model = CatBoostRegressor(**cat_params)
        return ModelWrapperCatBoost("CatBoost", cat_model)

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



