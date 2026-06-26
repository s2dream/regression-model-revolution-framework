import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
import logging
from automl_framework.model.wrappers import ABCModelWrapper, ModelWrapper
from automl_framework.model.model_factory import ModelFactory, ModelType

logger = logging.getLogger(__name__)


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
            "XGBoost",
            "MLP",
            "TabPFN",
            "RandomForest",
            "CatBoost",
        ])

        for model_name in active_models:
            try:
                # Convert string to ModelType enum
                model_type = ModelType.from_str(model_name)
                wrapped_model = ModelFactory.create_model(model_type, self.config, self.random_state)
                # Keep the model key as the string value of the enum for backward compatibility
                self.models[model_type.value] = wrapped_model
                logger.info(f"Model '{model_name}' initialized successfully from config.")
            except ValueError as ve:
                logger.warning(f"Model '{model_name}' is not registered or supported: {ve}")
            except Exception as e:
                logger.warning(
                    f"Model '{model_name}' could not be initialized. {model_name} will be unavailable. Error: {e}",
                    exc_info=True,
                )

    def add_custom_model(self, name: Any, model_instance: Any):
        """Allows adding any custom estimator that conforms to the fit/predict interface."""
        key = name.value if isinstance(name, ModelType) else str(name)
        if isinstance(model_instance, ABCModelWrapper):
            self.models[key] = model_instance
        else:
            self.models[key] = ModelWrapper(key, model_instance)
        logger.info(f"Custom model '{key}' added successfully.")

    def get_model(self, name: Any) -> Optional[ABCModelWrapper]:
        """Retrieves a specific wrapped model from the pool."""
        key = name.value if isinstance(name, ModelType) else str(name)
        return self.models.get(key)

    def list_available_models(self) -> List[str]:
        """Returns the list of active models in the pool."""
        return list(self.models.keys())
