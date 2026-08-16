import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from automl_framework.model.model_pool import ModelPool
from automl_framework.model.model_factory import ModelFactory, ModelType

logger = logging.getLogger(__name__)


# =====================================================================
# 🧠 SMART DEFAULT SEARCH SPACES
# =====================================================================
SMART_DEFAULT_SPACES: Dict[ModelType, Dict[str, Any]] = {
    ModelType.XGBOOST: {
        "n_estimators": [50, 300],
        "learning_rate": [0.01, 0.2],
        "max_depth": [3, 9],
    },
    ModelType.CATBOOST: {
        "iterations": [50, 300],
        "learning_rate": [0.01, 0.2],
        "depth": [3, 8],
        "verbose": 0,
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": [50, 300],
        "max_depth": [3, 15],
    },
    ModelType.MLP: {
        "hidden_layer_sizes": [[64], [128], [128, 64], [64, 32], [128, 64, 32]],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-1],
    },
    ModelType.TRANSFORMER: {
        "epochs": [10, 40],
        "lr": [1e-4, 5e-3],
        "dropout": [0.0, 0.3],
    },
}


def suggest_trial_param(trial: Any, param_name: str, spec: Any) -> Any:
    """
    Dynamically maps a parameter specification (range, choice list, or fixed scalar)
    to the appropriate Optuna trial suggestion method.
    """
    # 1. Handle Ranges & Categorical Choice Lists
    if isinstance(spec, (list, tuple)):
        # 1-a. Interval Range: [min, max] where both are numbers
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            min_val, max_val = spec[0], spec[1]
            if min_val > max_val:
                min_val, max_val = max_val, min_val

            # Pure integer range
            if isinstance(min_val, int) and isinstance(max_val, int):
                return trial.suggest_int(param_name, min_val, max_val)
            
            # Continuous float range (apply log scale if spanning orders of magnitude)
            min_f, max_f = float(min_val), float(max_val)
            log_scale = bool(min_f > 0 and (max_f / min_f) >= 10.0)
            return trial.suggest_float(param_name, min_f, max_f, log=log_scale)

        # 1-b. Categorical Choices: list of choices (strings, numbers, or complex objects like [128, 64])
        elif len(spec) > 0:
            # If choices contain lists/dicts, serialize to string for Optuna categorical compatibility
            serialized_choices = []
            is_complex = any(isinstance(x, (list, dict)) for x in spec)
            for item in spec:
                if isinstance(item, (list, dict)):
                    serialized_choices.append(json.dumps(item))
                else:
                    serialized_choices.append(str(item) if not isinstance(item, (int, float, str, bool)) else item)

            chosen = trial.suggest_categorical(param_name, serialized_choices)
            if is_complex and isinstance(chosen, str):
                try:
                    return json.loads(chosen)
                except Exception:
                    return chosen
            return chosen

    # 2. Fixed Scalar Value (User explicitly fixed this parameter, do not tune)
    return spec


def run_hpo_tuning(pool: ModelPool, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Runs dynamic hyperparameter optimization using Optuna on each active model in the pool.
    Honors user-defined search ranges [min, max], choice lists, and fixed scalar values.
    Updates the pool config and the active models with the best-tuned wrappers.
    """
    import optuna

    logger.info("🎯 Optuna Dynamic Hyperparameter Optimization (HPO) started...")

    # Internal split for validation to prevent leakage of test data
    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train, test_size=0.2, random_state=pool.random_state
    )

    hpo_config = pool.config.get("hpo", {})
    n_trials = hpo_config.get("n_trials", 10)
    metric = str(hpo_config.get("metric", "RMSE")).upper()
    logger.info(f"🎯 HPO optimization target metric: {metric} (trials={n_trials})")

    # Enable Optuna log visibility
    optuna.logging.set_verbosity(optuna.logging.INFO)

    user_models_config = pool.config.get("models", {})

    for name in list(pool.models.keys()):
        try:
            model_type = ModelType.from_str(name)
        except ValueError:
            logger.info(f"Skipping HPO for custom/non-standard model: '{name}'")
            continue

        if model_type in (ModelType.TABPFN, ModelType.TABICL):
            logger.info(f"ℹ️ {name} is a pre-trained tabular foundation model and does not require HPO. Skipping...")
            continue

        logger.info(f"🔍 Tuning dynamic search space for model: {name} (trials={n_trials})...")

        # Merge user specifications with smart defaults
        user_spec = user_models_config.get(name, {})
        default_spec = SMART_DEFAULT_SPACES.get(model_type, {})
        
        merged_specs = {}
        for pk, pv in default_spec.items():
            merged_specs[pk] = pv

        for pk, pv in user_spec.items():
            if isinstance(pv, (list, tuple)):
                # User explicitly defined a custom search range [min, max] or choice list
                merged_specs[pk] = pv
            elif pk not in default_spec:
                # Fixed parameter (e.g. n_jobs, verbose)
                merged_specs[pk] = pv
            else:
                if not isinstance(default_spec[pk], (list, tuple)):
                    merged_specs[pk] = pv

        def objective(trial):
            suggested_params = {}
            for param_name, spec in merged_specs.items():
                suggested_params[param_name] = suggest_trial_param(trial, param_name, spec)

            # Build mock configuration with trial parameters
            trial_config = pool.config.copy()
            if "models" not in trial_config:
                trial_config["models"] = {}
            trial_config["models"][name] = {
                **trial_config["models"].get(name, {}),
                **suggested_params,
            }

            try:
                model_wrap = ModelFactory.create_model(
                    model_type, trial_config, pool.random_state
                )
                model_wrap.fit(X_t, y_t)
                preds = model_wrap.predict(X_v)

                if metric == "RMSE":
                    score = np.sqrt(mean_squared_error(y_v, preds))
                elif metric == "MAE":
                    score = mean_absolute_error(y_v, preds)
                elif metric == "R2":
                    score = r2_score(y_v, preds)
                else:
                    score = np.sqrt(mean_squared_error(y_v, preds))
                return score
            except Exception as e:
                logger.warning(f"Trial failed for {name} with parameters {suggested_params}: {e}")
                return float("-inf") if metric == "R2" else float("inf")

        try:
            direction = "maximize" if metric == "R2" else "minimize"
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials)

            best_params = {}
            # Extract winning parameters (including fixed parameters and parsed objects)
            for param_name, spec in merged_specs.items():
                if param_name in study.best_params:
                    raw_val = study.best_params[param_name]
                    if isinstance(raw_val, str) and (raw_val.startswith("[") or raw_val.startswith("{")):
                        try:
                            best_params[param_name] = json.loads(raw_val)
                        except Exception:
                            best_params[param_name] = raw_val
                    else:
                        best_params[param_name] = raw_val
                else:
                    best_params[param_name] = spec

            logger.info(
                f"🎉 HPO complete for {name}. Best parameters: {best_params} (Best Validation {metric}: {study.best_value:.4f})"
            )

            # Update pool config with optimal concrete parameters
            if "models" not in pool.config:
                pool.config["models"] = {}
            if name not in pool.config["models"]:
                pool.config["models"][name] = {}
            pool.config["models"][name].update(best_params)

            # Recreate model wrapper using best parameters and replace in pool
            best_model_wrap = ModelFactory.create_model(model_type, pool.config, pool.random_state)
            pool.models[name] = best_model_wrap

        except Exception as e:
            logger.error(f"Failed to optimize hyperparameters for {name}: {e}", exc_info=True)
