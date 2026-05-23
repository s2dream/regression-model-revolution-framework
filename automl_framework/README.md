# 🚀 Antigravity AutoML Regression Framework

This directory contains a premium, highly modular **AutoML Regression Framework** tailored for high-performance tabular learning. It integrates classical baselines, neural estimators, and cutting-edge tabular architectures like **TabPFN** and **XGBoost** with an automated, high-fidelity visualization suite.

---

## 📂 Project Architecture

```
automl_framework/
│
├── main.py              # Main execution entry point orchestrating the whole pipeline
├── README.md            # Documentation, setup guide, and instructions
│
├── core/
│   ├── __init__.py      # Package initialization & key class exports
│   ├── data_loader.py   # Kaggle dataset integration, UCI downloading, and train-val-test splitting
│   ├── models.py        # Model pool defining TabPFN, XGBoost, MLP, and RandomForest
│   └── visualizer.py    # High-dpi, elegant dark-slate themed plots & JSON reports
│
├── data/                # Data storage directory for downloaded/generated datasets
└── outputs/             # Visualization outputs (.png) and structured execution reports (.json)
```

---

## ✨ Features & Requirements

### 1. Robust Data Pipeline (`core/data_loader.py`)
- **Kaggle API Integration**: Easily fetch datasets from Kaggle directly by passing a dataset ID (e.g. `user/dataset`).
- **Direct HTTP Downloading**: Supports direct downloads from URLs (such as the UCI Machine Learning Repository or customized datasets).
- **Graceful Preprocessing**: Handles automated median imputation for numeric features, mode imputation for categorical features, and automatic dummy/one-hot encoding.
- **Flexible Splitting**: Supports standard train/test splitting as well as 3-way train/validation/test partitioning.

### 2. Premium Visualization Suite (`core/visualizer.py`)
Generates production-grade, dark-themed visualizations:
- **Actual vs. Predicted Plot**: Diagonal identity line chart mapping model alignment and prediction variance.
- **Residual Analysis Plot**: Residual error scatter plot supporting diagnoses of heteroscedasticity.
- **Cross-Model Benchmarking**: High-fidelity horizontal bar charts directly comparing $R^2$, $RMSE$, and $MAE$ values.
- **Turn Reports**: Standardized `.json` documents capturing all metrics per turn and highlighting the best-performing model.

### 3. Comprehensive Model Pool (`core/models.py`)
Combines robust traditional architectures with state-of-the-art tabular models:
- **TabPFN Regressor**: Prior-Data Fitted Network optimized for extremely fast, hyperparameter-free tabular prediction.
- **XGBoost Regressor**: Extreme Gradient Boosting framework for highly efficient tabular representation.
- **Multi-Layer Perceptron (MLP)**: Deep learning baseline leveraging PyTorch/scikit-learn.
- **Random Forest Baseline**: Classical ensemble tree estimator.

---

## 🛠️ Getting Started

### Prerequisites

Install the framework dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
*(Optional) If you want to use the TabPFN and Kaggle API integrations:*
```bash
pip install tabpfn kaggle
```

### Running the Pipeline

To run the framework immediately with a **synthetic dataset** generated out-of-the-box:
```bash
python main.py
```

To run with a custom local CSV dataset:
```bash
python main.py --dataset-path path/to/dataset.csv --target name_of_target_column
```

To automatically pull a dataset from Kaggle and execute the benchmark:
```bash
python main.py --kaggle-dataset "sobhanmoosavi/us-accidents" --target "Severity"
```

To download from a URL and execute:
```bash
python main.py --url "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data" --target "MEDV"
```
