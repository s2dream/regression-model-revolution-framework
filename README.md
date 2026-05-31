# 🚀 Regression Model Revolution Framework

Welcome to the **Regression Model Revolution Framework**, a premium, production-grade **AutoML Regression pipeline** built in pure Python. It integrates state-of-the-art tabular algorithms (including **TabPFN** and **XGBoost**) with an automated, ultra-sleek dark-slate visualization suite to make tabular model benchmarking fast, gorgeous, and effortless.

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
├── configs/                        # ⚙️ Config profiles (experiment with diverse settings)
│   ├── default.yml                 # Default AutoML configuration profile
│   ├── kfold_split.yml             # K-Fold split configuration profile
│   ├── timeseries_split.yml        # Time-Series split configuration profile
│   └── custom_features.yml         # Custom feature columns selection profile
├── scripts/                        # 🏃 Automated scenario-specific bash execution scripts
│   ├── run_local_csv.sh            # Run AutoML benchmark pipeline with local CSV dataset
│   ├── run_local_jsonl.sh           # Run AutoML benchmark pipeline with local JSONL (dynamic columns)
│   └── run_url.sh                  # Download dataset from direct URL and run benchmark
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
│   │   ├── loaders.py              # Modular loaders (supports CSV, TSV, Parquet, and JSONL)
│   │   ├── preprocessors.py        # Imputation, encoding & ABCDataPreprocessor
│   │   ├── splitters.py            # Dataset split strategy & ABCDataSplitter
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
    ├── test_dataloader.py          # Tests for dataloading, preprocessing, and JSONL formats
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

The easiest way to run the pipeline with pre-configured settings is using the provided executable bash scripts under the `scripts/` directory:

```bash
# Run with a local CSV dataset
./scripts/run_local_csv.sh

# Run with a local JSONL dataset (supports dynamically aligned schemas!)
./scripts/run_local_jsonl.sh

# Run downloading a dataset from a remote URL
./scripts/run_url.sh
```

Alternatively, you can call `main.py` directly with custom command line arguments:

To run with a custom local CSV dataset:
```bash
python main.py --dataset-path path/to/dataset.csv --target name_of_target_column
```

To run with a local JSON Lines (`.jsonl`) dataset where some rows have missing keys:
```bash
python main.py --dataset-path data/synthetic_regression.jsonl --target Target_Y
```

To automatically pull a dataset from Kaggle and execute the benchmark:
```bash
python main.py --kaggle-dataset "sobhanmoosavi/us-accidents" --target "Severity"
```

To download from a URL (e.g. UCI ML Database or GitHub dataset) and run:
```bash
python main.py --url "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv" --target "medv"
```

---

## ⚙️ Configuration Profile Directory (`configs/`)

The framework supports dynamic, profile-driven configurations located inside the `configs/` directory. You can tune active models, custom feature subsets, data splitting strategies (`train_test_split`, `kfold`, `timeseries`), and model-specific hyperparameters **without modifying a single line of Python code**!

To execute using a specific profile, pass the `--config` parameter:
```bash
# Run with default settings (train_test_split)
python main.py --config configs/default.yml --dataset-path data/synthetic_regression.csv

# Run with K-Fold cross validation split strategy
python main.py --config configs/kfold_split.yml --dataset-path data/synthetic_regression.csv

# Run with sequential Time-Series split strategy
python main.py --config configs/timeseries_split.yml --dataset-path data/synthetic_regression.csv

# Run extracting only a designated custom features subset
python main.py --config configs/custom_features.yml --dataset-path data/synthetic_regression.csv
```

### Example Settings (`configs/default.yml`)

```yaml
# Global Framework Settings
framework:
  random_state: 42
  active_models:
    - XGBoost
    - MLP
    - TabPFN
    - RandomForest
    - CatBoost

# Data Pipeline Settings (Target/Features & Split selection)
data:
  data_dir: "data"
  output_dir: "outputs"
  target_column: "Target_Y"
  feature_columns: null   # e.g., ["Feature_Num", "Feature_Cat"]
  split:
    method: "train_test_split"  # Options: "train_test_split", "kfold", "timeseries"
    test_size: 0.2
    n_splits: 5
    shuffle: true

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
```

### Command Precedence & Fallbacks:
- Arguments specified directly on the command line (like `--test-size 0.3` or `--target target_col`) will seamlessly **override** their counterpart values inside configuration files.
- A resilient fallback is programmed: if a configuration file is missing or corrupted, the system continues running automatically using robust default configurations.

---

## 📊 Outputs & Reports
After every execution, the framework saves production-quality assets in:
- `outputs/turn_1_model_comparison_r2.png` - Horizontal bar chart comparing model R2 scores side-by-side.
- `outputs/turn_1_model_comparison_rmse.png` - Horizontal bar chart comparing model RMSE scores.
- `outputs/turn_1_[Model]_actual_vs_pred.png` - Visualizing prediction variance scatter plot with identity fit line.
- `outputs/turn_1_[Model]_residuals.png` - Diagnosing heteroscedasticity residual plot.
- `outputs/turn_1_report.json` - Complete metadata report highlighting the best-performing algorithm (champion).