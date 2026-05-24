# 🚀 Antigravity AutoML Regression Framework - Subpackage

This directory contains a premium, highly modular **AutoML Regression Framework** tailored for high-performance tabular learning. It integrates classical baselines, neural estimators, and cutting-edge tabular architectures like **TabPFN** and **XGBoost** with an automated, high-fidelity visualization suite.

---

## 📂 Internal Package Architecture

The framework is structured into highly cohesive, dedicated domain subpackages:

```
automl_framework/
│
├── __init__.py          # Facade layer exposing the high-level classes
├── README.md            # Subpackage documentation
│
├── dataloader/          # Data Domain (Kaggle/URL loaders & imputed splits)
│   ├── __init__.py
│   └── data_loader.py   
│
├── model/               # Model Domain (Pure model inventory & execution strategies)
│   ├── __init__.py
│   ├── models.py        # ModelPool data container & StandardBenchmarkExecutor strategy
│   └── wrappers.py      # Standardized wrappers for scikit-learn, XGBoost, and TabPFN
│
└── util/                # Utility & Analytics Domain (Premium dark theme visualizer)
    ├── __init__.py
    └── visualizer.py    
```

---

## ✨ Features & Domains

### 1. Ingestion (`dataloader/data_loader.py`)
- **Kaggle API Integration**: Fetch datasets from Kaggle directly by passing a dataset ID.
- **Direct HTTP Downloading**: Supports direct downloads from URLs (such as the UCI Machine Learning Repository or customized datasets).
- **Graceful Preprocessing**: Handles automated median imputation for numeric features, mode imputation for categorical features, and automatic dummy/one-hot encoding.
- **Flexible Splitting**: Supports standard train/test splitting as well as 3-way train/validation/test partitioning.

### 2. Core Learners & Executors (`model/models.py`, `model/wrappers.py`)
- **ModelPool Container**: Acts purely as a robust, configuration-driven **inventory repository** for wrapped models.
- **Benchmark Executors**: Decouples active fitting and prediction algorithms from inventory data using a strategy pattern. Extends `ABCModelExecutor` for highly scalable workflows.
- **Universal Adapters**: Wraps `XGBoost`, `TabPFN`, `Multi-Layer Perceptron (MLP)`, and standard baselines into uniform, exception-shielded wrappers.

### 3. Analytics & Visuals (`util/visualizer.py`)
Generates production-grade, dark-themed visualizations:
- **Actual vs. Predicted Plot**: Diagonal identity line chart mapping model alignment and prediction variance.
- **Residual Analysis Plot**: Residual error scatter plot supporting diagnoses of heteroscedasticity.
- **Cross-Model Benchmarking**: High-fidelity horizontal bar charts directly comparing $R^2$, $RMSE$, and $MAE$ values.
- **Turn Reports**: Standardized `.json` documents capturing all metrics per turn and highlighting the best-performing model.

---

## 🛠️ Usage Guidelines

This directory is structured as a **modular Python package** using a Facade pattern. If you wish to execute the complete AutoML benchmarking pipeline, please run the orchestrator **`main.py`** located in the **project root directory**:

```bash
# Go to the project root
cd /Users/jeonghoon/github/regression-model-revolution-framework

# Run the framework immediately with a synthetic dataset
python main.py
```

For custom programatic usage within your own python scripts:
```python
# Simple top-level Facade import!
from automl_framework import DataLoader, ModelPool, StandardBenchmarkExecutor, Visualizer

# 1. Load Data
loader = DataLoader(data_dir="data")
X, y = loader.load_dataset("data/synthetic_regression.csv", target_column="Target_Y")

# 2. Setup Decoupled ML Pipeline
pool = ModelPool(random_state=42)
executor = StandardBenchmarkExecutor(pool)
executor.fit_all(X, y)

# 3. Analyze & Plot
visualizer = Visualizer(output_dir="outputs")
metrics = executor.evaluate_all(X, y)
visualizer.save_json_report(metrics, turn=1)
```
