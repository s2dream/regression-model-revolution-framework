# 🚀 Antigravity Regression Model Revolution Framework

Welcome to the **Antigravity Regression Model Revolution Framework**, a premium, production-grade **AutoML Regression pipeline** built in pure Python. It integrates state-of-the-art tabular algorithms (including **TabPFN** and **XGBoost**) with an automated, ultra-sleek dark-slate visualization suite to make tabular model benchmarking fast, gorgeous, and effortless.

---

## 🎨 Visual Preview & Design Philosophy
This framework is built with a **Premium Dark Slate** aesthetic. Every generated plot uses carefully chosen high-contrast palettes, elegant micro-animations concept, and professional layouts designed to fit seamlessly into scientific papers, executive presentations, or advanced dashboards.

---

## 📂 Project Structure

A clean, modular design separating the orchestration layer from the internal package domains:

```
regression-model-revolution-framework/
│
├── main.py                         # 🌟 Central CLI pipeline orchestrator (Facade imports)
├── config.yml                      # ⚙️ Central YAML Configuration (Zero-code switches)
├── ARCHITECTURE.md                 # System modules & execution flow specs
├── REQ_SPEC.md                     # Requirements and functional specifications
├── requirements.txt                # Core packages required
│
├── automl_framework/               # 📦 The AutoML engine core package
│   ├── __init__.py                 # Facade layer exposing main APIs
│   ├── README.md                   # Subpackage documentation
│   │
│   ├── dataloader/                 # 📥 Data Ingestion domain subpackage (Facade & strategies)
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base classes for loaders/preprocessors/splitters
│   │   ├── loaders.py              # Modular loaders (Local, Kaggle, URL)
│   │   ├── preprocessors.py        # Imputation & encoding preprocessors
│   │   ├── splitters.py            # Dataset split strategy
│   │   └── data_loader_helper.py   # DataLoaderHelper Facade and pipeline orchestrator
│   │
│   ├── model/                      # 🤖 Machine Learning core subpackage
│   │   ├── __init__.py
│   │   ├── model_pool.py           # ModelPool inventory repository
│   │   ├── model_executor.py       # Benchmark executors (StandardBenchmarkExecutor)
│   │   ├── wrappers.py             # Exception-shielded model wrappers (XGBoost, MLP, TabPFN, RF, CatBoost, Transformer)
│   │   └── architecture/           # Neural network model architectures
│   │       └── transformer_encoder.py # PyTorch Transformer-based sequence regression (TransformerBasedRegression)
│   │
│   └── util/                       # 🛠️ Visualization & Utility subpackage
│       ├── __init__.py
│       └── visualizer.py           # Premium dark-theme visualizer & JSON reporting
│
└── tests/                          # 🧪 Comprehensive Test Suite
    ├── __init__.py
    ├── test_dataloader.py          # Tests for dataloading & preprocessing
    ├── test_model.py               # Tests for model initialization & executor
    └── test_visualizer.py          # Tests for premium visualizations & JSON reporting
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
You can install all necessary dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Optional) If you want to use the TabPFN and Kaggle API integrations:*
```bash
pip install tabpfn kaggle
```

### 2. Run the Benchmarking Pipeline

To run the benchmark pipeline instantly using the included synthetic dataset:
```bash
python main.py --dataset-path data/synthetic_regression.csv --target Target_Y
```

To run with a custom local CSV dataset:
```bash
python main.py --dataset-path path/to/dataset.csv --target name_of_target_column
```

To automatically pull a dataset from Kaggle and execute the benchmark:
```bash
python main.py --kaggle-dataset "sobhanmoosavi/us-accidents" --target "Severity"
```

To download from a URL (e.g. UCI ML Database) and run:
```bash
python main.py --url "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data" --target "MEDV"
```

---

## ⚙️ Configuration File (`config.yml`)

The framework is entirely governed by `config.yml` located at the root. You can tune active models, data pathways, and model-specific hyperparameters **without modifying a single line of python code**!

### Example Settings

```yaml
# Global Framework Settings
framework:
  random_state: 42
  test_size: 0.2
  active_models:
    - XGBoost
    - MLP
    - TabPFN
    - RandomForest
    - CatBoost

# Model Hyperparameters
models:
  XGBoost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
  MLP:
    hidden_layer_sizes: [128, 64]
    activation: "relu"
    max_iter: 500
  CatBoost:
    iterations: 100
    learning_rate: 0.1
    depth: 6
```

### Command Precedence:
- Arguments specified directly on the command line (like `--test-size 0.3` or `--target target_col`) will seamlessly **override** their counterpart values inside `config.yml`.
- A resilient fallback is programmed: if `config.yml` is missing or corrupted, the system continues running automatically using robust default configurations.

---

## 📊 Outputs & Reports
After every execution, the framework saves production-quality assets in:
- `outputs/turn_1_model_comparison_r2.png` - Horizontal bar chart comparing model R2 scores side-by-side.
- `outputs/turn_1_model_comparison_rmse.png` - Horizontal bar chart comparing model RMSE scores.
- `outputs/turn_1_[Model]_actual_vs_pred.png` - Visualizing prediction variance scatter plot with identity fit line.
- `outputs/turn_1_[Model]_residuals.png` - Diagnosing heteroscedasticity residual plot.
- `outputs/turn_1_report.json` - Complete metadata report highlighting the best-performing algorithm (champion).