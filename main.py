#!/usr/bin/env python
"""
Main entry point for the AutoML Framework.
Executes the full pipeline: Data Loading -> Preprocessing -> Model Training -> Evaluation -> Premium Visualization.
Now powered dynamically by central config.yml configurations!
"""

import os
import sys
import argparse
import pandas as pd
import yaml
import logging

# Standard absolute imports from the newly package-structured automl_framework
from automl_framework import DataLoaderHelper, ModelPool, StandardBenchmarkExecutor, Visualizer, setup_logger

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AutoML Framework for Advanced Tabular Regression")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config.yml file.")    
    parser.add_argument("--target", type=str, default=None, help="Name of the target variable/column. Overrides YAML config.")
    parser.add_argument("--test-size", type=float, default=None, help="Proportion of the dataset to use for testing. Overrides YAML config.")
    parser.add_argument("--turn", type=int, default=1, help="Current execution turn index (used for naming reports and outputs).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to local dataset CSV. Overrides YAML config.")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Optional Kaggle dataset name to download (e.g. 'user/dataset-name')")
    parser.add_argument("--url", type=str, default=None, help="Optional direct download URL (e.g. UCI dataset)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration profile from config.yml safely."""
    config = {}
    if os.path.exists(config_path):
        print(f"[Main] Loading configuration profile from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            print("[Main] Configuration profile loaded successfully.")
        except Exception as e:
            print(f"[Main] WARNING: Could not parse YAML file. Falling back to default settings. Error: {e}")
    else:
        print(f"[Main] WARNING: Config file '{config_path}' not found. Falling back to default settings.")
    return config


def resolve_parameters(args: argparse.Namespace, config: dict) -> dict:
    """Resolve parameters by combining CLI options and configuration file defaults."""
    random_state = config.get("framework", {}).get("random_state", 42)
    
    test_size = args.test_size
    if test_size is None:
        test_size = config.get("framework", {}).get("test_size", 0.2)
        
    target_column = args.target
    if target_column is None:
        target_column = config.get("data", {}).get("target_column", "Target_Y")

    data_dir = config.get("data", {}).get("data_dir", "data")
    output_dir = config.get("data", {}).get("output_dir", "outputs")
    
    return {
        "random_state": random_state,
        "test_size": test_size,
        "target_column": target_column,
        "data_dir": data_dir,
        "output_dir": output_dir
    }


def train_and_evaluate(executor: StandardBenchmarkExecutor, pool: ModelPool, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train all models in the pool and evaluate them on testing split."""
    logger = logging.getLogger("automl_framework.main")
    logger.info("🤖 Model Pool Training initiated...")
    logger.info(f"Available models to train: {pool.list_available_models()}")
    executor.fit_all(X_train, y_train)
    
    logger.info("📊 Evaluating trained models on testing split...")
    metrics = executor.evaluate_all(X_test, y_test)
    
    # Print results summary in standard output
    logger.info("\n" + "-" * 50)
    logger.info(f"{'Model Name':<18} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    logger.info("-" * 50)
    for model_name, score_dict in metrics.items():
        logger.info(f"{model_name:<18} | {score_dict['RMSE']:<10.4f} | {score_dict['MAE']:<10.4f} | {score_dict['R2']:<10.4f}")
    logger.info("-" * 50 + "\n")
    
    if not metrics:
        logger.error("ERROR: No models were successfully trained and evaluated.")
        sys.exit(1)
        
    return metrics


def generate_visualizations_and_report(visualizer: Visualizer, executor: StandardBenchmarkExecutor, metrics: dict, X_test: pd.DataFrame, y_test: pd.Series, turn: int):
    """Generate premium charts, residual plots, and a structured JSON report."""
    logger = logging.getLogger("automl_framework.main")
    logger.info("🎨 Generating premium charts & structured reports...")
    
    # Plot comparisons of R2 score
    visualizer.plot_model_comparison(metrics, metric_name="R2", turn=turn)
    # Plot comparisons of RMSE score
    visualizer.plot_model_comparison(metrics, metric_name="RMSE", turn=turn)
    
    predictions = executor.get_predictions(X_test)
    
    # Generate specific plots for each model
    for model_name, y_pred in predictions.items():
        visualizer.plot_actual_vs_predicted(y_test, y_pred, model_name=model_name, turn=turn)
        visualizer.plot_residuals(y_test, y_pred, model_name=model_name, turn=turn)
        
    # Save structured JSON execution report
    report_path = visualizer.save_json_report(metrics, turn=turn)
    
    best_model = max(metrics.keys(), key=lambda k: metrics[k]["R2"])
    best_r2 = metrics[best_model]["R2"]
    
    logger.info("\n🏆 Execution Summary:")
    logger.info(f"  - Best Model: {best_model} with R2 Score of {best_r2:.4f}")
    logger.info(f"  - Report saved: {report_path}")
    logger.info(f"  - Visualization outputs saved in: '{visualizer.output_dir}'")
    logger.info("=" * 60)


def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    # Initialize logger
    setup_logger(turn=args.turn, config=config)
    logger = logging.getLogger("automl_framework.main")
    
    logger.info("=" * 60)
    logger.info(f"🚀 AutoML Regression Framework - Turn {args.turn}")
    logger.info("=" * 60)
    
    params = resolve_parameters(args, config)
    
    # Initialize framework components dynamically from configurations
    dataloader_helper = DataLoaderHelper(data_dir=params["data_dir"])
    pool = ModelPool(random_state=params["random_state"], config=config)
    executor = StandardBenchmarkExecutor(pool)
    visualizer = Visualizer(output_dir=params["output_dir"])
    
    # fetch dataset
    try:
        dataset_file = dataloader_helper.fetch_dataset(
            dataset_path=args.dataset_path,
            kaggle_dataset=args.kaggle_dataset,
            url=args.url
        )
    except Exception as e:
        logger.critical(f"CRITICAL ERROR fetching dataset: {e}", exc_info=True)
        sys.exit(1)
    
    # preprocessing datset
    try:
        X_train, y_train, X_test, y_test = dataloader_helper.prepare_data(
            dataset_file, params["target_column"], params["test_size"], params["random_state"]
        )
    except Exception as e:
        logger.critical(f"CRITICAL ERROR loading/processing dataset: {e}", exc_info=True)
        sys.exit(1)
    
    # train and evaluate
    metrics = train_and_evaluate(executor, pool, X_train, y_train, X_test, y_test)

    # visualize and report
    generate_visualizations_and_report(visualizer, executor, metrics, X_test, y_test, args.turn)

if __name__ == "__main__":
    main()
