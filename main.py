#!/usr/bin/env python
"""
Main entry point for the AutoML Framework.
Now refactored into a high-level AutoMLPipeline class for robust state-management, 
effortless library imports, and one-click execution orchestration!
"""

import os
import sys
import argparse
import pandas as pd
import yaml
import logging
from typing import Tuple, Optional, Dict, Any

# Standard absolute imports from the newly package-structured automl_framework
from automl_framework import DataLoaderHelper, ModelPool, StandardBenchmarkExecutor, Visualizer, setup_logger

class AutoMLPipeline:
    """
    High-level orchestrator for the AutoML tabular regression pipeline.
    Encapsulates all step states and coordinates ingestion, preprocessing, training, 
    evaluating, and premium chart reporting.
    """
    def __init__(
        self, 
        config_path: str = "configs/default.yml", 
        turn: int = 1,
        target: Optional[str] = None, 
        test_size: Optional[float] = None
    ):
        self.config_path = config_path
        self.turn = turn
        self.config = self._load_config(config_path)
        
        # Resolve random state, test size, target column, data directory, and output directory
        self.random_state = self.config.get("framework", {}).get("random_state", 42)
        
        self.test_size = test_size
        if self.test_size is None:
            self.test_size = self.config.get("framework", {}).get("test_size", 0.2)
            
        self.target_column = target
        if self.target_column is None:
            self.target_column = self.config.get("data", {}).get("target_column", "Target_Y")

        self.data_dir = self.config.get("data", {}).get("data_dir", "data")
        self.output_dir = self.config.get("data", {}).get("output_dir", "outputs")

        # Initialize core components
        self.dataloader_helper = DataLoaderHelper(data_dir=self.data_dir, config=self.config)
        self.pool = ModelPool(random_state=self.random_state, config=self.config)
        self.executor = StandardBenchmarkExecutor(self.pool)
        self.visualizer = Visualizer(output_dir=self.output_dir)

        # Pipeline step states
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.metrics: Optional[Dict[str, Dict[str, float]]] = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration profile safely with robust exception shield fallbacks."""
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

    def prepare_data(
        self, 
        dataset_path: Optional[str] = None, 
        kaggle_dataset: Optional[str] = None, 
        url: Optional[str] = None
    ):
        """Step 1: Download, load, preprocess, and partition dataset."""
        logger = logging.getLogger("automl_framework.main")
        
        # Fetch dataset absolute filepath
        dataset_file = self.dataloader_helper.fetch_dataset(
            dataset_path=dataset_path,
            kaggle_dataset=kaggle_dataset,
            url=url
        )
        
        # Load, preprocess, and partition train/test splits
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataloader_helper.load_and_preprocess_data(
            dataset_file=dataset_file,
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def train_and_evaluate(self) -> dict:
        """Step 2: Train all active models inside ModelPool and evaluate test splits."""
        logger = logging.getLogger("automl_framework.main")
        logger.info("🤖 Model Pool Training initiated...")
        logger.info(f"Available models to train: {self.pool.list_available_models()}")
        self.executor.fit_all(self.X_train, self.y_train)
        
        logger.info("📊 Evaluating trained models on testing split...")
        self.metrics = self.executor.evaluate_all(self.X_test, self.y_test)
        
        # Print results summary in standard output
        logger.info("\n" + "-" * 50)
        logger.info(f"{'Model Name':<18} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
        logger.info("-" * 50)
        for model_name, score_dict in self.metrics.items():
            logger.info(f"{model_name:<18} | {score_dict['RMSE']:<10.4f} | {score_dict['MAE']:<10.4f} | {score_dict['R2']:<10.4f}")
        logger.info("-" * 50 + "\n")
        
        if not self.metrics:
            logger.error("ERROR: No models were successfully trained and evaluated.")
            sys.exit(1)
            
        return self.metrics

    def generate_reports(self):
        """Step 3: Generate horizontal bar comparisons, diagnostic residual plots, and champion report."""
        logger = logging.getLogger("automl_framework.main")
        logger.info("🎨 Generating premium charts & structured reports...")
        
        # Plot comparisons of R2 & RMSE score
        self.visualizer.plot_model_comparison(self.metrics, metric_name="R2", turn=self.turn)
        self.visualizer.plot_model_comparison(self.metrics, metric_name="RMSE", turn=self.turn)
        
        predictions = self.executor.get_predictions(self.X_test)
        
        # Generate specific plots for each model
        for model_name, y_pred in predictions.items():
            self.visualizer.plot_actual_vs_predicted(self.y_test, y_pred, model_name=model_name, turn=self.turn)
            self.visualizer.plot_residuals(self.y_test, y_pred, model_name=model_name, turn=self.turn)
            
        # Save structured JSON execution report
        report_path = self.visualizer.save_json_report(self.metrics, turn=self.turn)
        
        best_model = max(self.metrics.keys(), key=lambda k: self.metrics[k]["R2"])
        best_r2 = self.metrics[best_model]["R2"]
        
        logger.info("\n🏆 Execution Summary:")
        logger.info(f"  - Best Model: {best_model} with R2 Score of {best_r2:.4f}")
        logger.info(f"  - Report saved: {report_path}")
        logger.info(f"  - Visualization outputs saved in: '{self.visualizer.output_dir}'")
        logger.info("=" * 60)

    def run(
        self, 
        dataset_path: Optional[str] = None, 
        kaggle_dataset: Optional[str] = None, 
        url: Optional[str] = None
    ):
        """One-click pipeline runner executing all AutoML phases sequentially."""
        self.prepare_data(dataset_path, kaggle_dataset, url)
        self.train_and_evaluate()
        self.generate_reports()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AutoML Framework for Advanced Tabular Regression")
    parser.add_argument("--config", type=str, default="configs/default.yml", help="Path to config profile YAML.")    
    parser.add_argument("--target", type=str, default=None, help="Name of the target variable/column. Overrides YAML config.")
    parser.add_argument("--test-size", type=float, default=None, help="Proportion of the dataset to use for testing. Overrides YAML config.")
    parser.add_argument("--turn", type=int, default=1, help="Current execution turn index (used for naming reports and outputs).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to local dataset CSV. Overrides YAML config.")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Optional Kaggle dataset name to download (e.g. 'user/dataset-name')")
    parser.add_argument("--url", type=str, default=None, help="Optional direct download URL (e.g. UCI dataset)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Instantiate the AutoML pipeline class (CLI args prioritize over configs)
    pipeline = AutoMLPipeline(
        config_path=args.config,
        turn=args.turn,
        target=args.target,
        test_size=args.test_size
    )
    
    # Initialize logger
    setup_logger(turn=args.turn, config=pipeline.config)
    logger = logging.getLogger("automl_framework.main")
    
    logger.info("=" * 60)
    logger.info(f"🚀 AutoML Regression Framework - Turn {args.turn}")
    logger.info("=" * 60)
    
    # Execute full pipeline
    try:
        pipeline.run(
            dataset_path=args.dataset_path,
            kaggle_dataset=args.kaggle_dataset,
            url=args.url
        )
    except Exception as e:
        logger.critical(f"CRITICAL ERROR running AutoML Pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
