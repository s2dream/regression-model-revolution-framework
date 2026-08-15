"""
SHAP (SHapley Additive exPlanations) Analyzer Module.
Provides model explainability, feature attribution plots, and dedicated in-context
explainer pipelines for tabular regressors including TabICL, tree ensembles, and neural models.
"""

import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger("automl_framework.shap_analyzer")


class SHAPAnalyzer:
    """
    Automated SHAP feature attribution and model interpretability analyzer.
    Features:
    - TreeExplainer for tree ensembles (XGBoost, CatBoost, RandomForest)
    - Dedicated In-Context Explainer for TabICL
    - KernelExplainer / ModelExplainer for neural and black-box models
    - Premium dark-theme visualization rendering
    - Standalone structured JSON report generation
    - Exception shielding against package missing or computation errors
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        max_samples: int = 100,
        random_state: int = 42
    ):
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.random_state = random_state
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_dark_theme(self):
        """Configure matplotlib aesthetic styling for dark slate UI consistency."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': '#0d1117',
            'axes.facecolor': '#161b22',
            'axes.edgecolor': '#30363d',
            'axes.labelcolor': '#c9d1d9',
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
            'grid.color': '#21262d',
            'text.color': '#f0f6fc',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        })

    def analyze_model(
        self,
        model_wrapper: Any,
        model_name: str,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        turn: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Execute SHAP feature attribution analysis for a given trained model wrapper.
        
        Args:
            model_wrapper: Concrete ModelWrapper instance (e.g. ModelWrapperXGBoost, ModelWrapperTabICL)
            model_name: String name identifier (e.g. 'TabICL', 'XGBoost', 'CatBoost')
            X_train: Training split features (used as background dataset)
            X_test: Testing split features (used for explanation evaluation)
            feature_names: Optional list of column names
            turn: Current execution turn index

        Returns:
            Dictionary containing SHAP report metadata or None if analysis failed/skipped.
        """
        try:
            import shap
        except ImportError:
            logger.warning("[SHAP] 'shap' package is not installed. Skipping SHAP analysis. (Run: pip install shap)")
            return None

        logger.info(f"🔍 [SHAP] Initiating SHAP analysis for model: '{model_name}' (Turn {turn})...")

        # 1. Resolve feature names and data formats
        if isinstance(X_test, pd.DataFrame):
            resolved_feature_names = list(X_test.columns)
            X_test_arr = X_test.to_numpy(dtype=np.float32)
        else:
            X_test_arr = np.asarray(X_test, dtype=np.float32)
            resolved_feature_names = feature_names or [f"Feature_{i}" for i in range(X_test_arr.shape[1])]

        if isinstance(X_train, pd.DataFrame):
            X_train_arr = X_train.to_numpy(dtype=np.float32)
        else:
            X_train_arr = np.asarray(X_train, dtype=np.float32)

        # 2. Subsample evaluation test set if larger than max_samples
        n_test = len(X_test_arr)
        if n_test > self.max_samples:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_test, size=self.max_samples, replace=False)
            X_eval = X_test_arr[sample_indices]
        else:
            X_eval = X_test_arr

        # 3. Determine Explainer engine and compute SHAP values
        raw_model = getattr(model_wrapper, "model", model_wrapper)
        lower_name = model_name.lower()
        explainer_engine = "Unknown"
        shap_values_matrix = None

        try:
            # Case A: TabICL Dedicated In-Context Explainer
            if "tabicl" in lower_name:
                explainer_engine = "TabICL Dedicated In-Context Explainer"
                logger.info(f"⚡ [SHAP] Applying dedicated In-Context Explainer pipeline for TabICL...")
                
                # Sample background data for in-context prompting
                bg_size = min(30, len(X_train_arr))
                np.random.seed(self.random_state)
                bg_indices = np.random.choice(len(X_train_arr), size=bg_size, replace=False)
                bg_sample = X_train_arr[bg_indices]
                
                def predict_fn(x):
                    if isinstance(X_test, pd.DataFrame):
                        x_df = pd.DataFrame(x, columns=resolved_feature_names)
                        return model_wrapper.predict(x_df)
                    return model_wrapper.predict(x)

                explainer = shap.Explainer(predict_fn, bg_sample)
                shap_explanation = explainer(X_eval)
                if hasattr(shap_explanation, "values"):
                    shap_values_matrix = shap_explanation.values
                else:
                    shap_values_matrix = np.asarray(shap_explanation)

            # Case B: Tree-based models (XGBoost, CatBoost, RandomForest)
            elif any(k in lower_name for k in ["xgboost", "catboost", "randomforest", "random_forest", "tree"]):
                explainer_engine = "TreeExplainer (Exact Tree SHAP)"
                logger.info(f"🌲 [SHAP] Utilizing TreeExplainer for tree ensemble '{model_name}'...")
                
                try:
                    explainer = shap.TreeExplainer(raw_model)
                    shap_values_obj = explainer.shap_values(X_eval)
                    shap_values_matrix = np.asarray(shap_values_obj)
                except Exception as tree_err:
                    logger.warning(f"[SHAP] TreeExplainer fallback triggered: {tree_err}")
                    def predict_fn(x):
                        return model_wrapper.predict(x)
                    explainer = shap.Explainer(predict_fn, X_train_arr[:30])
                    shap_explanation = explainer(X_eval)
                    shap_values_matrix = shap_explanation.values if hasattr(shap_explanation, "values") else np.asarray(shap_explanation)

            # Case C: General Neural / Foundation / Black-box models (MLP, Transformer, TabPFN)
            else:
                explainer_engine = "KernelExplainer / ModelExplainer"
                logger.info(f"🧠 [SHAP] Utilizing ModelExplainer for '{model_name}'...")
                bg_size = min(25, len(X_train_arr))
                bg_sample = X_train_arr[:bg_size]
                
                def predict_fn(x):
                    return model_wrapper.predict(x)
                    
                explainer = shap.Explainer(predict_fn, bg_sample)
                shap_explanation = explainer(X_eval)
                shap_values_matrix = shap_explanation.values if hasattr(shap_explanation, "values") else np.asarray(shap_explanation)

        except Exception as e:
            logger.error(f"[SHAP] Error during SHAP value computation for '{model_name}': {e}", exc_info=True)
            return None

        if shap_values_matrix is None or len(shap_values_matrix) == 0:
            logger.warning(f"[SHAP] SHAP values empty for '{model_name}'.")
            return None

        # Ensure 2D matrix shape (num_samples, num_features)
        if len(shap_values_matrix.shape) > 2:
            shap_values_matrix = shap_values_matrix.squeeze()

        # 4. Calculate Feature Importance Metrics
        mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
        feature_importance_dict = {
            resolved_feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(resolved_feature_names))
        }
        # Sort descending by importance
        sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
        top_features = list(sorted_importance.keys())[:min(5, len(sorted_importance))]

        # 5. Generate Visualizations (Summary, Bar, Waterfall)
        self._setup_dark_theme()
        generated_artifacts = []

        # Chart 1: Feature Importance Horizontal Bar Plot
        bar_plot_path = os.path.join(self.output_dir, f"turn_{turn}_{model_name}_shap_bar.png")
        try:
            fig, ax = plt.subplots(figsize=(10, max(5, len(resolved_feature_names) * 0.35)), dpi=150)
            y_pos = np.arange(len(sorted_importance))
            features_rev = list(reversed(list(sorted_importance.keys())))
            values_rev = list(reversed(list(sorted_importance.values())))
            
            bars = ax.barh(y_pos, values_rev, color='#6366f1', edgecolor='#818cf8', alpha=0.85, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_rev, fontsize=10, fontweight='medium')
            ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output)", fontsize=11, labelpad=8)
            ax.set_title(f"SHAP Feature Importance — {model_name} (Turn {turn})\nEngine: {explainer_engine}", 
                         fontsize=13, fontweight='bold', pad=12, color='#f0f6fc')
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='#30363d')
            
            # Label bar values
            max_val = max(values_rev) if values_rev and max(values_rev) > 0 else 1.0
            for bar in bars:
                width = bar.get_width()
                ax.text(width + max_val * 0.015, bar.get_y() + bar.get_height() / 2,
                        f"{width:.4f}", va='center', ha='left', color='#e6edf3', fontsize=9)

            plt.tight_layout()
            plt.savefig(bar_plot_path, facecolor='#0d1117', bbox_inches='tight')
            plt.close(fig)
            generated_artifacts.append(bar_plot_path)
            logger.info(f"  - Saved SHAP Bar Chart: {bar_plot_path}")
        except Exception as e:
            logger.warning(f"[SHAP] Failed to render SHAP bar plot: {e}")

        # Chart 2: SHAP Beeswarm / Summary Scatter Plot
        summary_plot_path = os.path.join(self.output_dir, f"turn_{turn}_{model_name}_shap_summary.png")
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, len(resolved_feature_names) * 0.4)), dpi=150)
            shap.summary_plot(
                shap_values_matrix,
                X_eval,
                feature_names=resolved_feature_names,
                show=False,
                plot_type="dot",
                color_bar=True
            )
            plt.title(f"SHAP Summary (Beeswarm) — {model_name} (Turn {turn})\nEngine: {explainer_engine}", 
                      fontsize=12, fontweight='bold', color='#f0f6fc', pad=15)
            plt.tight_layout()
            plt.savefig(summary_plot_path, facecolor='#0d1117', bbox_inches='tight')
            plt.close()
            generated_artifacts.append(summary_plot_path)
            logger.info(f"  - Saved SHAP Summary Plot: {summary_plot_path}")
        except Exception as e:
            logger.warning(f"[SHAP] Failed to render SHAP summary plot: {e}")

        # 6. Build and Save Standalone JSON Report
        report_data = {
            "turn": turn,
            "model_name": model_name,
            "explainer_engine": explainer_engine,
            "num_samples_analyzed": len(X_eval),
            "num_features": len(resolved_feature_names),
            "top_features": top_features,
            "mean_abs_shap": sorted_importance,
            "artifacts": [os.path.basename(p) for p in generated_artifacts],
            "timestamp": datetime.datetime.now().isoformat()
        }

        json_report_path = os.path.join(self.output_dir, f"turn_{turn}_{model_name}_shap_report.json")
        try:
            with open(json_report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  - Saved SHAP JSON Report: {json_report_path}")
            report_data["report_file"] = json_report_path
        except Exception as e:
            logger.warning(f"[SHAP] Failed to write SHAP JSON report: {e}")

        logger.info(f"✅ [SHAP] Analysis for '{model_name}' completed successfully.")
        return report_data
