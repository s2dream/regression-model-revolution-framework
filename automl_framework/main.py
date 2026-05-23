#!/usr/bin/env python
"""
Main entry point for the AutoML Framework.
Executes the full pipeline: Data Loading -> Preprocessing -> Model Training -> Evaluation -> Premium Visualization.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add the parent directory of core to path so imports work cleanly if run from root or inside automl_framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.models import ModelPool
from core.visualizer import Visualizer

def create_synthetic_dataset(filepath: str):
    """
    Creates a synthetic regression dataset if no external dataset is provided or available.
    Ensures the user can run 'python main.py' immediately and successfully!
    """
    print(f"[Main] Creating a synthetic regression dataset at {filepath}...")
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)
    x3 = np.random.choice(['low', 'medium', 'high'], size=n_samples)
    
    # Non-linear regression equation with some noise
    y = 3.5 * (x1 ** 2) + 1.2 * x2 + np.where(x3 == 'high', 5.0, np.where(x3 == 'medium', 2.0, 0.0)) + np.random.normal(0, 1.5, n_samples)
    
    df = pd.DataFrame({
        'Feature_X1': x1,
        'Feature_X2': x2,
        'Category_X3': x3,
        'Target_Y': y
    })
    
    # Make sure parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print("[Main] Synthetic dataset created successfully.")

def main():
    parser = argparse.ArgumentParser(description="AutoML Framework for Advanced Tabular Regression")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to local dataset CSV. If None, a synthetic dataset is generated.")
    parser.add_argument("--target", type=str, default="Target_Y", help="Name of the target variable/column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to use for testing (0.0 to 1.0).")
    parser.add_argument("--turn", type=int, default=1, help="Current execution turn index (used for naming reports and outputs).")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Optional Kaggle dataset name to download (e.g. 'user/dataset-name')")
    parser.add_argument("--url", type=str, default=None, help="Optional direct download URL (e.g. UCI dataset)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🚀 AutoML Regression Framework - Turn {args.turn}")
    print("=" * 60)
    
    # Initialize framework components
    loader = DataLoader(data_dir="data")
    pool = ModelPool(random_state=42)
    visualizer = Visualizer(output_dir="outputs")
    
    dataset_file = args.dataset_path
    
    # Step 1: Data Fetching and Loading
    if args.kaggle_dataset:
        try:
            download_dir = loader.download_from_kaggle(args.kaggle_dataset)
            # Try to find CSV files in the download directory
            csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
            if csv_files:
                dataset_file = os.path.join(download_dir, csv_files[0])
            else:
                print(f"[Main] No CSV files found in downloaded Kaggle files at {download_dir}")
                sys.exit(1)
        except Exception as e:
            print(f"[Main] Kaggle download failed. Falling back. Error: {e}")
            
    elif args.url:
        try:
            dataset_file = loader.download_from_url(args.url, "downloaded_data.csv")
        except Exception as e:
            print(f"[Main] URL download failed. Falling back. Error: {e}")

    # Fallback to synthetic data if no filepath was resolved or exists
    if not dataset_file or not os.path.exists(dataset_file):
        dataset_file = os.path.join("data", "synthetic_regression.csv")
        if not os.path.exists(dataset_file):
            create_synthetic_dataset(dataset_file)
            args.target = "Target_Y" # Adjust target name to match synthetic
            
    # Load and Preprocess Data
    try:
        X, y = loader.load_dataset(dataset_file, target_column=args.target)
        X_processed = loader.preprocess_data(X)
        
        # Split Data
        splits = loader.split_data(X_processed, y, test_size=args.test_size, random_state=42)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_test, y_test = splits["X_test"], splits["y_test"]
        
        print(f"\n[Main] Data shape summary:")
        print(f"  - Features dimension: {X_processed.shape[1]}")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Testing samples:  {X_test.shape[0]}\n")
        
    except Exception as e:
        print(f"[Main] CRITICAL ERROR loading/processing dataset: {e}")
        sys.exit(1)
        
    # Step 2: Model Training
    print("🤖 Model Pool Training initiated...")
    print(f"Available models to train: {pool.list_available_models()}")
    pool.fit_all(X_train, y_train)
    
    # Step 3: Evaluation
    print("\n📊 Evaluating trained models on testing split...")
    metrics = pool.evaluate_all(X_test, y_test)
    
    # Print results summary in standard output
    print("\n" + "-" * 50)
    print(f"{'Model Name':<18} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 50)
    for model_name, score_dict in metrics.items():
        print(f"{model_name:<18} | {score_dict['RMSE']:<10.4f} | {score_dict['MAE']:<10.4f} | {score_dict['R2']:<10.4f}")
    print("-" * 50 + "\n")
    
    if not metrics:
        print("[Main] ERROR: No models were successfully trained and evaluated.")
        sys.exit(1)
        
    # Step 4: Premium Visualization and Report Generation
    print("🎨 Generating premium charts & structured reports...")
    
    # Plot comparisons of R2 score
    comparison_chart_path = visualizer.plot_model_comparison(metrics, metric_name="R2", turn=args.turn)
    # Plot comparisons of RMSE score
    visualizer.plot_model_comparison(metrics, metric_name="RMSE", turn=args.turn)
    
    predictions = pool.get_predictions(X_test)
    
    # Generate specific plots for each model
    for model_name, y_pred in predictions.items():
        visualizer.plot_actual_vs_predicted(y_test, y_pred, model_name=model_name, turn=args.turn)
        visualizer.plot_residuals(y_test, y_pred, model_name=model_name, turn=args.turn)
        
    # Save structured JSON execution report
    report_path = visualizer.save_json_report(metrics, turn=args.turn)
    
    best_model = max(metrics.keys(), key=lambda k: metrics[k]["R2"])
    best_r2 = metrics[best_model]["R2"]
    
    print("\n🏆 Execution Summary:")
    print(f"  - Best Model: {best_model} with R2 Score of {best_r2:.4f}")
    print(f"  - Report saved: {report_path}")
    print(f"  - Visualization outputs saved in: '{visualizer.output_dir}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
