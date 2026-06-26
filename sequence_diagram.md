# 📊 AutoML Regression Framework - Full Sequence Diagram

This document contains a comprehensive sequence diagram representing the entire workflow of the **Regression Model Revolution Framework**, including the newly refactored **Factory Method** design pattern for model creation.

---

## 🗺️ Mermaid Sequence Diagram

Below is the complete sequence diagram mapping out initialization, data loading/preprocessing, training, evaluation, and visualization.

```mermaid
sequenceDiagram
    autonumber
    actor User as User/CLI
    participant Main as main.py (AutoMLPipeline)
    participant Pool as ModelPool
    participant Type as ModelType (Enum)
    participant Factory as ModelFactory
    participant Helper as DataLoaderHelper
    participant Loader as LocalFileDataLoader
    participant Prep as StandardDataPreprocessor
    participant Split as TrainTestSplitter
    participant Exec as StandardBenchmarkExecutor
    participant Wrapper as ModelWrapper (e.g. XGBoost)
    participant Vis as Visualizer

    %% ==========================================
    %% 1. INITIALIZATION PHASE
    %% ==========================================
    Note over User, Main: [1. Initialization Phase]
    User->>Main: Instantiate AutoMLPipeline(config_path, turn, target, test_size)
    activate Main
    Main->>Helper: Instantiate DataLoaderHelper(data_dir, config)
    
    Main->>Pool: Instantiate ModelPool(random_state, config)
    activate Pool
    Pool->>Pool: Call _initialize_default_models()
    
    loop For each active_model name in config ("XGBoost", "MLP", etc.)
        Pool->>Type: Resolve string name via ModelType.from_str(model_name)
        activate Type
        Type-->>Pool: Return ModelType Enum member
        deactivate Type
        
        Pool->>Factory: Create model via create_model(model_type, config, random_state)
        activate Factory
        Note over Factory: Resolve imports & build raw regressor
        Factory-->>Pool: Return concrete ModelWrapper instance
        deactivate Factory
        
        Pool->>Pool: Store wrapper in self.models[model_type.value]
    end
    Pool-->>Main: Return ModelPool instance
    deactivate Pool
    
    Main->>Exec: Instantiate StandardBenchmarkExecutor(pool)
    Main->>Vis: Instantiate Visualizer(output_dir)
    Main-->>User: Pipeline Initialized
    deactivate Main

    %% ==========================================
    %% 2. DATA PREPARATION PHASE
    %% ==========================================
    Note over User, Main: [2. Ingestion & Preprocessing Phase]
    User->>Main: Call pipeline.run(dataset_path, kaggle_dataset, url)
    activate Main
    Main->>Main: Execute prepare_data(dataset_path, kaggle_dataset, url)
    Main->>Helper: Call fetch_dataset(...)
    Helper-->>Main: Return resolved local dataset filepath
    
    Main->>Helper: Call load_and_preprocess_data(dataset_file, target, test_size, random_state)
    activate Helper
    
    Helper->>Loader: Call load_data()
    activate Loader
    Note over Loader: Parse format (CSV/TSV/Parquet/JSONL)<br/>Handle dynamic JSONL keys
    Loader-->>Helper: Return raw DataFrame X, Series y
    deactivate Loader

    Helper->>Prep: Call preprocess(X)
    activate Prep
    Note over Prep: Median/Mode Imputation<br/>One-Hot Dummy Encoding
    Prep-->>Helper: Return preprocessed DataFrame X_processed
    deactivate Prep

    Helper->>Split: Call split(X_processed, y)
    activate Split
    Note over Split: Partition train/test splits based on random_state
    Split-->>Helper: Return X_train, y_train, X_test, y_test
    deactivate Split

    Helper-->>Main: Return X_train, y_train, X_test, y_test
    deactivate Helper

    %% ==========================================
    %% 3. MODEL TRAINING & SCORING PHASE
    %% ==========================================
    Note over Main, Exec: [3. Model Training & Scoring Phase]
    Main->>Main: Execute train_and_evaluate()
    Main->>Exec: Call fit_all(X_train, y_train)
    activate Exec
    loop For each wrapped model in ModelPool
        Exec->>Pool: Retrieve model wrapper instance
        Exec->>Wrapper: Call fit(X_train, y_train)
        activate Wrapper
        Note over Wrapper: Fit raw estimator under exception-shielding
        Wrapper-->>Exec: Done
        deactivate Wrapper
    end
    Exec-->>Main: Training Completed
    deactivate Exec

    Main->>Exec: Call evaluate_all(X_test, y_test)
    activate Exec
    loop For each trained model wrapper in ModelPool
        Exec->>Wrapper: Call predict(X_test)
        activate Wrapper
        Wrapper-->>Exec: Return y_pred array
        deactivate Wrapper
        Exec->>Exec: Calculate metrics (RMSE, MAE, R2)
    end
    Exec-->>Main: Return metrics Dict
    deactivate Exec

    %% ==========================================
    %% 4. VISUALIZATION & REPORTING PHASE
    %% ==========================================
    Note over Main, Vis: [4. Visuals & Reporting Phase]
    Main->>Main: Execute generate_reports()
    Main->>Vis: Call plot_model_comparison(metrics, metric_name, turn)
    Note over Vis: Save model_comparison_r2/rmse horizontal bar chart
    
    Main->>Exec: Call get_predictions(X_test)
    activate Exec
    loop For each model wrapper in ModelPool
        Exec->>Wrapper: Call predict(X_test)
        activate Wrapper
        Wrapper-->>Exec: Return y_pred array
        deactivate Wrapper
    end
    Exec-->>Main: Return predictions Dict
    deactivate Exec

    loop For each model prediction array
        Main->>Vis: Call plot_actual_vs_predicted(y_test, y_pred, model_name, turn)
        Note over Vis: Save Actual vs Predicted Scatter Plot with Identity line
        Main->>Vis: Call plot_residuals(y_test, y_pred, model_name, turn)
        Note over Vis: Save Heteroscedasticity Residual Plot
    end

    Main->>Vis: Call save_json_report(metrics, turn)
    Vis-->>Main: Return turn_report.json filepath
    
    Main-->>User: Pipeline Execution Finished (Show Best Champion Model)
    deactivate Main
```
