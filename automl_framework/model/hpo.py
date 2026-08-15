import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from automl_framework.model.model_pool import ModelPool
from automl_framework.model.model_factory import ModelFactory, ModelType

logger = logging.getLogger(__name__)


def run_hpo_tuning(pool: ModelPool, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Runs hyperparameter optimization using Optuna on each active model wrapper in the pool.
    Updates the pool config and the pool models dict with the best tuned models.
    """
    import optuna

    logger.info("🎯 Optuna Hyperparameter Optimization (HPO) started...")

    # Internal split for validation to prevent leakage of the test dataset
    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train, test_size=0.2, random_state=pool.random_state
    )

    hpo_config = pool.config.get("hpo", {})
    n_trials = hpo_config.get("n_trials", 10)

    # Enable Optuna log visibility so the user can see HPO progress in real-time
    optuna.logging.set_verbosity(optuna.logging.INFO)

    for name in list(pool.models.keys()):
        try:
            model_type = ModelType.from_str(name)
        except ValueError:
            logger.info(f"Skipping HPO for custom/non-standard model: '{name}'")
            continue

        if model_type in (ModelType.TABPFN, ModelType.TABICL):
            logger.info(f"{name} is a pre-trained foundation model and does not require HPO. Skipping...")
            continue

        logger.info(f"Tuning hyperparameters for model: {name} (trials={n_trials})...")

        def objective(trial):
            # Define search space for each model type
            suggested_params = {}
            if model_type == ModelType.XGBOOST:
                suggested_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                }
            elif model_type == ModelType.MLP:
                arch_choice = trial.suggest_categorical(
                    "hidden_layer_sizes", ["64", "128", "128,64", "64,32"]
                )
                suggested_params = {
                    "hidden_layer_sizes": [int(x) for x in arch_choice.split(",")],
                    "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                    "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                }
            elif model_type == ModelType.RANDOM_FOREST:
                suggested_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                }
            elif model_type == ModelType.CATBOOST:
                suggested_params = {
                    "iterations": trial.suggest_int("iterations", 50, 250),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 3, 8),
                    "verbose": 0,
                }
            elif model_type == ModelType.TRANSFORMER:
                suggested_params = {
                    "epochs": trial.suggest_int("epochs", 10, 40),
                    "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                }

            # Create a mock config dictionary containing trial parameters
            trial_config = pool.config.copy()
            if "models" not in trial_config:
                trial_config["models"] = {}
            # Update configuration with suggested params
            if name not in trial_config["models"]:
                trial_config["models"][name] = {}
            trial_config["models"][name] = {
                **trial_config["models"][name],
                **suggested_params,
            }

            try:
                # Build wrapper with suggested parameters
                model_wrap = ModelFactory.create_model(
                    model_type, trial_config, pool.random_state
                )
                model_wrap.fit(X_t, y_t)
                preds = model_wrap.predict(X_v)
                rmse = np.sqrt(mean_squared_error(y_v, preds))
                return rmse
            except Exception as e:
                logger.warning(f"Trial failed for {name} with parameters {suggested_params}: {e}")
                return float("inf")

        try:
            # Create study and run optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params.copy()
            if model_type == ModelType.MLP and "hidden_layer_sizes" in best_params:
                best_params["hidden_layer_sizes"] = [
                    int(x) for x in best_params["hidden_layer_sizes"].split(",")
                ]

            logger.info(
                f"🎉 HPO complete for {name}. Best parameters: {best_params} (Best Validation RMSE: {study.best_value:.4f})"
            )

            # Save best parameters back to pool config
            if "models" not in pool.config:
                pool.config["models"] = {}
            if name not in pool.config["models"]:
                pool.config["models"][name] = {}
            pool.config["models"][name].update(best_params)

            # Recreate model wrapper using the best parameters and replace it in the pool
            best_model_wrap = ModelFactory.create_model(model_type, pool.config, pool.random_state)
            pool.models[name] = best_model_wrap

        except Exception as e:
            logger.error(f"Failed to optimize hyperparameters for {name}: {e}", exc_info=True)
