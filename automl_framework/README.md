# 🚀 Antigravity AutoML Regression Framework - Subpackage

This directory contains a premium, highly modular **AutoML Regression Framework** tailored for high-performance tabular learning. It integrates classical baselines, neural estimators, and cutting-edge tabular architectures like **TabPFN** and **XGBoost** with an automated, high-fidelity visualization suite.

---

## 📂 Internal Package Architecture

The framework is structured into highly cohesive, dedicated domain subpackages:

```
automl_framework/
│
├── __init__.py          # Facade layer exposing high-level classes
├── README.md            # Subpackage documentation
│
├── dataloader/          # Data Domain (Facade & strategy submodules)
│   ├── __init__.py
│   ├── base.py          # Abstract base classes for loaders/preprocessors/splitters
│   ├── loaders.py       # Modular loaders (Local, Kaggle, URL)
│   ├── preprocessors.py # Standard data imputation & encoding
│   ├── splitters.py     # Dataset split strategies (train/val/test)
│   └── data_loader_helper.py # DataLoaderHelper facade class and data pipeline orchestrator
│
├── model/               # Model Domain (Inventory & execution strategies)
│   ├── __init__.py
│   ├── model_pool.py    # ModelPool data container for bound model inventory
│   ├── model_executor.py# Execution strategies (ABCModelExecutor & StandardBenchmarkExecutor)
│   ├── wrappers.py      # Standardized wrappers for Scikit-Learn, XGBoost, TabPFN, CatBoost, Transformer
│   └── architecture/    # Neural network model architectures
│       └── transformer_encoder.py # PyTorch Transformer-based sequence regression (TransformerBasedRegression)
│
└── util/                # Utility & Analytics Domain (Premium dark theme visualizer)
    ├── __init__.py
    └── visualizer.py    # Elegant matplotlib/seaborn visualization and JSON reporting
```

---

## ✨ Features & Domains

### 1. Ingestion (`dataloader/`)
- **Facade Strategy Pattern**: `DataLoaderHelper` delegates specialized loading, preprocessing, and splitting tasks to modular strategy subcomponents, and exposes a unified high-level `prepare_data` pipeline orchestrator method.
- **Kaggle API Integration (`loaders.py`)**: Fetch datasets from Kaggle directly by passing a dataset ID.
- **Direct HTTP Downloading (`loaders.py`)**: Supports direct downloads from URLs (such as the UCI Machine Learning Repository or customized datasets).
- **Graceful Preprocessing (`preprocessors.py`)**: Handles automated median imputation for numeric features, mode imputation for categorical features, and automatic dummy/one-hot encoding.
- **Flexible Splitting (`splitters.py`)**: Supports standard train/test splitting as well as 3-way train/validation/test partitioning.

### 2. Core Learners & Executors (`model/`)
- **ModelPool Container (`model_pool.py`)**: Acts purely as a robust, configuration-driven **inventory repository** for wrapped models.
- **Benchmark Executors (`model_executor.py`)**: Decouples active fitting and prediction algorithms from inventory data using a strategy pattern. Extends `ABCModelExecutor` for highly scalable workflows.
- **Universal Adapters (`wrappers.py`)**: Wraps `XGBoost`, `TabPFN`, `CatBoost`, `Multi-Layer Perceptron (MLP)`, standard baselines, and `TransformerBasedRegression` into uniform, exception-shielded wrappers.
- **Transformer-Based Regression (`architecture/transformer_encoder.py`)**: Implements standard sequence-based deep learning regression using PyTorch, with adjustable pooling strategies (`mean`, `max`, `last`), sequence padding masking, and support for both scalar regression and probabilistic mean/variance distribution estimation.

### 3. Analytics & Visuals (`util/visualizer.py`)
Generates production-grade, dark-themed visualizations:
- **Actual vs. Predicted Plot**: Diagonal identity line chart mapping model alignment and prediction variance (`turn_{turn}_{model_name}_actual_vs_pred.png`).
- **Residual Analysis Plot**: Residual error scatter plot supporting diagnoses of heteroscedasticity (`turn_{turn}_{model_name}_residuals.png`).
- **Cross-Model Benchmarking**: High-fidelity horizontal bar charts directly comparing $R^2$, $RMSE$, and $MAE$ values (`turn_{turn}_model_comparison_{metric}.png`).
- **Turn Reports**: Standardized `.json` documents capturing all metrics per turn and highlighting the best-performing champion model (`turn_{turn}_report.json`).

---

## 🛠️ Usage Guidelines

This directory is structured as a **modular Python package** using a Facade pattern. If you wish to execute the complete AutoML benchmarking pipeline, please run the orchestrator **`main.py`** located in the **project root directory**:

```bash
# Go to the project root
cd /Users/jeonghoon/github/regression-model-revolution-framework

# Run the framework by specifying a local dataset CSV path and target column
python main.py --dataset-path data/synthetic_regression.csv --target Target_Y
```

For custom programatic usage within your own python scripts:
```python
# Simple top-level Facade import!
# Simple top-level Facade import!
from automl_framework import DataLoaderHelper, ModelPool, StandardBenchmarkExecutor, Visualizer

# 1. Load, preprocess, and split Data in one unified step!
dataloader_helper = DataLoaderHelper(data_dir="data")
X_train, y_train, X_test, y_test = dataloader_helper.prepare_data(
    "data/your_dataset.csv", target_column="target_column_name", test_size=0.2, random_state=42
)

# 2. Setup Decoupled ML Pipeline
pool = ModelPool(random_state=42)
executor = StandardBenchmarkExecutor(pool)
executor.fit_all(X_train, y_train)

# 3. Analyze & Plot
visualizer = Visualizer(output_dir="outputs")
metrics = executor.evaluate_all(X_test, y_test)
visualizer.save_json_report(metrics, turn=1)
```
