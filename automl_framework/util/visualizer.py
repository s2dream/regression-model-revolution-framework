import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualizer is responsible for plotting model evaluation metrics, actual vs. predicted values,
    residual analyses, and model comparisons. Saves outputs as elegant images (.png) and metadata (.json).
    """
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Apply premium aesthetic configurations for matplotlib/seaborn
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'figure.dpi': 150,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.titlesize': 15,
            'figure.facecolor': '#0d1117',  # Elegant dark slate background
            'axes.facecolor': '#161b22',    # Lighter container background
            'grid.color': '#30363d',        # Soft gridlines
            'text.color': '#c9d1d9',        # Crisp white/grey text
            'axes.labelcolor': '#8b949e',   # Muted grey labels
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
        })
        # Harmonious modern color palette
        self.palette = ["#58a6ff", "#ff7b72", "#aff5b4", "#d2a8ff", "#e3b341"]

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """
        Creates an elegant Scatter Plot of Actual vs. Predicted values with an identity line.
        
        Returns:
            str: Path to the saved visualization
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Scatter plot with elegant translucency
        ax.scatter(y_true, y_pred, alpha=0.6, color=self.palette[0], edgecolors='none', s=25, label='Predictions')
        
        # Identity line (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color=self.palette[1], linestyle='--', lw=2, label='Perfect Fit')
        
        ax.set_title(f"{model_name}: Actual vs Predicted (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_actual_vs_pred.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Actual vs Pred plot to {filepath}")
        return filepath

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """
        Plots Residuals vs Predicted values to diagnose variance behavior (heteroscedasticity).
        
        Returns:
            str: Path to the saved visualization
        """
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(7, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.6, color=self.palette[2], edgecolors='none', s=25)
        ax.axhline(0, color=self.palette[1], linestyle='--', lw=2)
        
        ax.set_title(f"{model_name}: Residual Plot (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_residuals.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Residual plot to {filepath}")
        return filepath

    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], metric_name: str = "RMSE", turn: int = 1) -> str:
        """
        Compares multiple models in a neat horizontal bar chart for a specified metric (e.g. RMSE, R2).
        
        Returns:
            str: Path to the saved visualization
        """
        models = list(metrics.keys())
        values = [m_data[metric_name] for m_data in metrics.values()]
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Sort in ascending/descending depending on the metric
        reverse_sort = True if metric_name in ["R2"] else False
        sorted_pairs = sorted(zip(values, models), reverse=reverse_sort)
        sorted_values, sorted_models = zip(*sorted_pairs)
        
        bars = ax.barh(sorted_models, sorted_values, color=self.palette[3], height=0.5, edgecolor='#30363d')
        
        # Add labels to the ends of the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (max(sorted_values) * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}", 
                va='center', 
                ha='left', 
                color='#c9d1d9',
                fontsize=9
            )
            
        ax.set_title(f"Model Comparison: {metric_name} (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Models")
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_model_comparison_{metric_name.lower()}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Model Comparison plot to {filepath}")
        return filepath

    def save_json_report(self, metrics: Dict[str, Dict[str, float]], turn: int = 1, shap_reports: Dict[str, Dict[str, str]] = None, learning_curves: Dict[str, str] = None) -> str:
        """
        Saves the turn's execution and performance metrics in an structured JSON report.
        
        Returns:
            str: Path to the saved report
        """
        report_data = {
            "turn": turn,
            "metrics": metrics,
            "best_model": max(metrics.keys(), key=lambda k: metrics[k]["R2"]) if metrics else None
        }
        if shap_reports:
            report_data["shap_reports"] = shap_reports
        if learning_curves:
            report_data["learning_curves"] = learning_curves
            
        filename = f"turn_{turn}_report.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved JSON report to {filepath}")
        return filepath

    def plot_shap_explainability(self, model_wrap, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str, turn: int = 1, max_samples: int = 100) -> Dict[str, str]:
        """
        Computes SHAP values and saves Beeswarm and Bar Plots for the given model wrapper.
        Includes safety shielding and fallback strategies.
        
        Returns:
            Dict[str, str]: Dictionary containing file paths to generated plots
        """
        try:
            import shap
        except ImportError:
            logger.warning("⚠️ 'shap' library is not installed. Skipping SHAP analysis.")
            return {}

        logger.info(f"🧠 Computing SHAP values for model: {model_name}...")
        
        # Prepare datasets: downsample for performance
        if len(X_train) > max_samples:
            X_background = shap.sample(X_train, max_samples, random_state=42)
        else:
            X_background = X_train

        X_explain = X_test
        if len(X_test) > max_samples:
            X_explain = shap.sample(X_test, max_samples, random_state=42)

        explainer = None
        shap_values = None

        # Build Explainer based on model class/type
        try:
            from automl_framework.model.model_factory import ModelType
            try:
                model_type = ModelType.from_str(model_name)
            except ValueError:
                model_type = None

            # TreeExplainer is extremely fast and works directly on tree ensembles
            if model_type in [ModelType.XGBOOST, ModelType.CATBOOST, ModelType.RANDOM_FOREST]:
                try:
                    explainer = shap.TreeExplainer(model_wrap.model)
                    shap_values = explainer(X_explain)
                except Exception as e:
                    logger.debug(f"Failed to initialize TreeExplainer for {model_name}: {e}. Falling back to default Explainer.")
                    explainer = None

            # Fallback model-agnostic Permutation/Kernel Explainer
            if explainer is None:
                explainer = shap.Explainer(model_wrap.predict, X_background)
                shap_values = explainer(X_explain)

        except Exception as e:
            logger.error(f"Failed to initialize explainer or calculate SHAP values for {model_name}: {e}", exc_info=True)
            return {}

        paths = {}

        # 1. Beeswarm / Summary Plot
        try:
            plt.figure(figsize=(8, 5))
            
            # Support both shap.Explanation and numpy arrays
            try:
                if hasattr(shap_values, "values"):
                    shap.plots.beeswarm(shap_values, show=False)
                else:
                    shap.summary_plot(shap_values, X_explain, show=False)
            except Exception:
                shap.summary_plot(shap_values, X_explain, show=False)

            fig = plt.gcf()
            fig.patch.set_facecolor('#0d1117')
            ax = plt.gca()
            ax.set_facecolor('#161b22')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.tick_params(colors='#8b949e')
            plt.title(f"{model_name}: SHAP Summary (Turn {turn})", color='#ffffff', pad=15)
            plt.tight_layout()

            summary_filename = f"turn_{turn}_{model_name}_shap_summary.png"
            summary_path = os.path.join(self.output_dir, summary_filename)
            plt.savefig(summary_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
            plt.close()
            paths["summary_plot"] = summary_path
            logger.info(f"Saved SHAP Beeswarm plot to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP Beeswarm plot for {model_name}: {e}", exc_info=True)
            plt.close()

        # 2. Bar Plot (Feature Importance)
        try:
            plt.figure(figsize=(8, 5))
            
            try:
                if hasattr(shap_values, "values"):
                    shap.plots.bar(shap_values, show=False)
                else:
                    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
            except Exception:
                shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)

            fig = plt.gcf()
            fig.patch.set_facecolor('#0d1117')
            ax = plt.gca()
            ax.set_facecolor('#161b22')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.tick_params(colors='#8b949e')
            plt.title(f"{model_name}: Feature Importance (Turn {turn})", color='#ffffff', pad=15)
            plt.tight_layout()

            bar_filename = f"turn_{turn}_{model_name}_shap_bar.png"
            bar_path = os.path.join(self.output_dir, bar_filename)
            plt.savefig(bar_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
            plt.close()
            paths["bar_plot"] = bar_path
            logger.info(f"Saved SHAP Bar plot to {bar_path}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP Bar plot for {model_name}: {e}", exc_info=True)
            plt.close()

        return paths

    def plot_learning_curve(self, loss_history: list, model_name: str, turn: int = 1) -> str:
        """
        Plots the training loss curve for iterative models.
        
        Returns:
            str: Path to the saved visualization
        """
        if not loss_history:
            return ""
            
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(loss_history) + 1), loss_history, color=self.palette[0], lw=2, label="Train Loss")
        
        ax.set_title(f"{model_name}: Learning Curve (Turn {turn})", color='#ffffff', pad=15)
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel("Loss / Error")
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        
        plt.tight_layout()
        
        filename = f"turn_{turn}_{model_name}_learning_curve.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
        plt.close()
        logger.info(f"Saved Learning Curve plot to {filepath}")
        return filepath

