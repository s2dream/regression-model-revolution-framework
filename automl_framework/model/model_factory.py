from enum import Enum
import logging
from typing import Dict, Any, Optional

from automl_framework.model.wrappers import (
    ABCModelWrapper,
    ModelWrapperXGBoost,
    ModelWrapperMLP,
    ModelWrapperTabPFN,
    ModelWrapperRandomForest,
    ModelWrapperCatBoost,
    ModelWrapperTransformer,
)

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    XGBOOST = "XGBoost"
    MLP = "MLP"
    TABPFN = "TabPFN"
    RANDOM_FOREST = "RandomForest"
    CATBOOST = "CatBoost"
    TRANSFORMER = "Transformer"

    @classmethod
    def from_str(cls, value: str) -> "ModelType":
        """Resolves a string representation to a ModelType enum member."""
        # Exact match with value (e.g., "XGBoost")
        for member in cls:
            if member.value == value:
                return member
        # Case-insensitive match on name or value (e.g., "xgboost" or "randomforest")
        normalized_value = value.replace("_", "").replace("-", "").lower()
        for member in cls:
            if (
                member.name.lower() == normalized_value
                or member.value.lower() == normalized_value
            ):
                return member
        raise ValueError(f"'{value}' is not a valid ModelType")


class ModelFactory:
    """
    Factory pattern class for creating model wrappers.
    Encapsulates building logic for different regression models.
    """

    @staticmethod
    def create_model(
        model_type: Any, config: Dict[str, Any], random_state: int
    ) -> ABCModelWrapper:
        """
        Creates and returns a wrapper instance for the requested model type.
        """
        # Resolve to ModelType enum if a string is provided
        if not isinstance(model_type, ModelType):
            model_type = ModelType.from_str(str(model_type))

        if model_type == ModelType.XGBOOST:
            return ModelFactory._build_xgboost(config, random_state)
        elif model_type == ModelType.MLP:
            return ModelFactory._build_mlp(config, random_state)
        elif model_type == ModelType.TABPFN:
            return ModelFactory._build_tabpfn(config, random_state)
        elif model_type == ModelType.RANDOM_FOREST:
            return ModelFactory._build_random_forest(config, random_state)
        elif model_type == ModelType.CATBOOST:
            return ModelFactory._build_catboost(config, random_state)
        elif model_type == ModelType.TRANSFORMER:
            return ModelFactory._build_transformer(config, random_state)
        else:
            raise ValueError(f"Unsupported ModelType: {model_type}")

    @staticmethod
    def _build_xgboost(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        from xgboost import XGBRegressor

        xgb_params = config.get("models", {}).get(ModelType.XGBOOST.value, {}).copy()
        if "random_state" not in xgb_params:
            xgb_params["random_state"] = random_state
        xgb_model = XGBRegressor(**xgb_params)
        return ModelWrapperXGBoost(ModelType.XGBOOST.value, xgb_model)

    @staticmethod
    def _build_mlp(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        from sklearn.neural_network import MLPRegressor

        mlp_params = config.get("models", {}).get(ModelType.MLP.value, {}).copy()

        if "hidden_layer_sizes" in mlp_params:
            mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
        if "random_state" not in mlp_params:
            mlp_params["random_state"] = random_state

        default_mlp_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "random_state": random_state,
        }
        merged_mlp_params = {**default_mlp_params, **mlp_params}
        mlp_model = MLPRegressor(**merged_mlp_params)
        return ModelWrapperMLP(ModelType.MLP.value, mlp_model)

    @staticmethod
    def _build_tabpfn(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise Exception("import tabpfn package is required. Please run pip install tabpfn")

        tabpfn_params = config.get("models", {}).get(ModelType.TABPFN.value, {}).copy()
        if "random_state" not in tabpfn_params:
            tabpfn_params["random_state"] = random_state

        # Modern TabPFN uses 'n_estimators'
        if "N_ensemble_configurations" in tabpfn_params and "n_estimators" not in tabpfn_params:
            tabpfn_params["n_estimators"] = tabpfn_params.pop("N_ensemble_configurations")

        tabpfn_model = TabPFNRegressor(**tabpfn_params)
        return ModelWrapperTabPFN(ModelType.TABPFN.value, tabpfn_model)

    @staticmethod
    def _build_random_forest(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        from sklearn.ensemble import RandomForestRegressor

        rf_params = config.get("models", {}).get(ModelType.RANDOM_FOREST.value, {}).copy()
        if "random_state" not in rf_params:
            rf_params["random_state"] = random_state
        rf_model = RandomForestRegressor(**rf_params)
        return ModelWrapperRandomForest(ModelType.RANDOM_FOREST.value, rf_model)

    @staticmethod
    def _build_catboost(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        from catboost import CatBoostRegressor

        cat_params = config.get("models", {}).get(ModelType.CATBOOST.value, {}).copy()
        if "random_seed" not in cat_params:
            cat_params["random_seed"] = random_state
        cat_model = CatBoostRegressor(**cat_params)
        return ModelWrapperCatBoost(ModelType.CATBOOST.value, cat_model)

    @staticmethod
    def _build_transformer(config: Dict[str, Any], random_state: int) -> ABCModelWrapper:
        import torch
        from automl_framework.model.architecture.transformer_encoder import (
            TransformerBasedRegression,
        )

        tf_params = config.get("models", {}).get(ModelType.TRANSFORMER.value, {}).copy()

        # Extract wrapper specific arguments
        epochs = tf_params.pop("epochs", 50)  # fast default for training neural nets
        lr = tf_params.pop("lr", 0.001)
        batch_size = tf_params.pop("batch_size", 32)
        verbose = tf_params.pop("verbose", False)

        # Get dataset shape info or default
        input_dim = tf_params.pop("input_dim", 1)
        d_model = tf_params.pop("d_model", 32)
        nhead = tf_params.pop("nhead", 2)
        num_layers = tf_params.pop("num_layers", 1)
        dim_feedforward = tf_params.pop("dim_feedforward", 64)
        dropout = tf_params.pop("dropout", 0.1)
        pooling = tf_params.pop("pooling", "mean")
        predict_distribution = tf_params.pop("predict_distribution", False)

        torch.manual_seed(random_state)
        raw_model = TransformerBasedRegression(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            predict_distribution=predict_distribution,
        )

        return ModelWrapperTransformer(
            ModelType.TRANSFORMER.value,
            raw_model,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
        )
