# 📊 AutoML Regression Framework - Comprehensive Sequence Diagrams

본 문서는 **AutoML Regression Framework**의 파이프라인 전체 실행 제어 흐름, Optuna HPO 하이퍼파라미터 최적화 루프, SHAP 모델 해석 및 피처 기여도 분석 흐름, 그리고 Streamlit WebUI 상호작용 흐름을 Mermaid 시퀀스 다이어그램으로 기술한 **동적 모델 설계서**입니다.

---

## 1. End-to-End AutoML Pipeline Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / CLI Entry
    participant Main as main.py (AutoMLPipeline)
    participant Pool as ModelPool
    participant Type as ModelType (Enum)
    participant Factory as ModelFactory
    participant Helper as DataLoaderHelper
    participant Loader as LocalFileDataLoader
    participant Prep as StandardDataPreprocessor
    participant Split as TrainTestSplitter
    participant Exec as StandardBenchmarkExecutor
    participant HPO as OptunaHPOTuner
    participant Wrapper as ModelWrapper (XGB/Cat/RF/MLP/TabPFN/TabICL/NN)
    participant Vis as Visualizer
    participant SHAP as SHAPAnalyzer

    %% ==========================================
    %% 1. INITIALIZATION PHASE
    %% ==========================================
    Note over User, Main: [1. 초기화 단계 - Initialization Phase]
    User->>Main: AutoMLPipeline(config_path, turn, target, test_size, enable_shap, shap_model)
    activate Main
    Main->>Helper: DataLoaderHelper(data_dir, config)
    
    Main->>Pool: ModelPool(random_state, config)
    activate Pool
    Pool->>Pool: _initialize_default_models()
    
    loop YAML active_models 순회 ("XGBoost", "CatBoost", "TabICL", etc.)
        Pool->>Type: ModelType.from_str(model_name)
        activate Type
        Type-->>Pool: ModelType Enum 반환
        deactivate Type
        
        Pool->>Factory: create_model(model_type, config, random_state)
        activate Factory
        Factory-->>Pool: concrete ModelWrapper 인스턴스 반환
        deactivate Factory
        
        Pool->>Pool: self.models[model_type.value]에 래퍼 저장
    end
    Pool-->>Main: ModelPool 인스턴스 반환
    deactivate Pool
    
    Main->>Exec: StandardBenchmarkExecutor(pool)
    Main->>Vis: Visualizer(output_dir)
    Main->>SHAP: SHAPAnalyzer(output_dir, max_samples, random_state)
    Main-->>User: 파이프라인 초기화 완료
    deactivate Main

    %% ==========================================
    %% 2. DATA PREPARATION PHASE
    %% ==========================================
    Note over User, Main: [2. 데이터 수집 및 전처리 단계 - Data Ingestion & Prep]
    User->>Main: pipeline.run(dataset_path, kaggle_dataset, url)
    activate Main
    Main->>Main: prepare_data(dataset_path, kaggle_dataset, url)
    Main->>Helper: fetch_dataset(...)
    Helper-->>Main: 검증된 로컬 데이터 파일 경로 반환
    
    Main->>Helper: prepare_data(dataset_file, target, test_size, random_state)
    activate Helper
    
    Helper->>Loader: load_data()
    activate Loader
    Loader-->>Helper: 원본 X (DataFrame), y (Series) 반환
    deactivate Loader

    Helper->>Prep: preprocess(X)
    activate Prep
    Prep-->>Helper: 전처리 완료된 X_processed 반환
    deactivate Prep

    Helper->>Split: split(X_processed, y)
    activate Split
    Split-->>Helper: X_train, y_train, X_test, y_test 반환
    deactivate Split

    Helper-->>Main: X_train, y_train, X_test, y_test 반환
    deactivate Helper

    %% ==========================================
    %% 3. HPO & MODEL TRAINING PHASE
    %% ==========================================
    Note over Main, Exec: [3. HPO 튜닝 및 모델 훈련 단계 - Tuning & Training]
    Main->>Main: train_and_evaluate()
    Main->>Exec: fit_all(X_train, y_train)
    activate Exec

    opt config['hpo']['enabled'] == True
        Exec->>HPO: run_hpo_tuning(pool, X_train, y_train)
        activate HPO
        loop TabPFN/TabICL 제외 각 튜닝 대상 모델 순회
            loop 1..n_trials 반복
                HPO->>Factory: create_model(model_type, trial_config, random_state)
                activate Factory
                Factory-->>HPO: Trial용 ModelWrapper 반환
                deactivate Factory
                HPO->>Wrapper: fit & validation RMSE/MAE/R2 스코어링
            end
            HPO->>Factory: create_model(model_type, best_config, random_state)
            activate Factory
            Factory-->>HPO: 최적 파라미터 적용된 ModelWrapper 반환
            deactivate Factory
            HPO->>Pool: self.models[name]을 최적 모델로 교체
        end
        HPO-->>Exec: HPO 최적화 완료
        deactivate HPO
    end

    loop ModelPool 내 전체 활성 모델 순회
        Exec->>Pool: model = get_model(name)
        Exec->>Wrapper: fit(X_train, y_train)
        activate Wrapper
        Wrapper-->>Exec: 학습 완료
        deactivate Wrapper
    end
    Exec-->>Main: 전체 모델 학습 완료
    deactivate Exec

    %% ==========================================
    %% 4. EVALUATION & STANDARD REPORTS
    %% ==========================================
    Note over Main, Vis: [4. 성능 평가 및 표준 리포트 생성 단계]
    Main->>Exec: evaluate_all(X_test, y_test)
    activate Exec
    Exec-->>Main: metrics 딕셔너리 반환
    deactivate Exec

    Main->>Main: generate_reports()
    Main->>Vis: plot_model_comparison(metrics, metric_name, turn)
    Main->>Exec: get_predictions(X_test)
    activate Exec
    Exec-->>Main: 전체 모델의 y_pred 딕셔너리 반환
    deactivate Exec

    loop 각 모델별 예측값 순회
        Main->>Vis: plot_actual_vs_predicted(...)
        Main->>Vis: plot_residuals(...)
    end
    Main->>Vis: save_json_report(metrics, turn)
    Main->>Vis: save_html_report(metrics, turn)
    Main->>Vis: save_markdown_summary(metrics, turn)

    %% ==========================================
    %% 5. SHAP INTERPRETABILITY PHASE
    %% ==========================================
    opt config['shap']['enabled'] == True
        Note over Main, SHAP: [5. SHAP 모델 해석 및 피처 기여도 분석 단계]
        Main->>Main: run_shap_analysis()
        Main->>Pool: get_model(shap_model_name or champion)
        activate Pool
        Pool-->>Main: target_model_wrapper 반환
        deactivate Pool

        Main->>SHAP: analyze_model(target_model_wrapper, model_name, X_train, X_test, feature_names, turn)
        activate SHAP
        
        alt Model is TabICL
            Note over SHAP: TabICL Dedicated In-Context Explainer 파이프라인 가동<br/>배경 데이터 샘플링 및 In-Context 예측 함수 바인딩
        else Model is Tree Ensemble (XGB/Cat/RF)
            Note over SHAP: TreeExplainer 가동 (Exact Tree SHAP 고속 연산)
        else Model is Neural / Blackbox (MLP/Transformer)
            Note over SHAP: KernelExplainer / ModelExplainer 가동
        end

        SHAP->>SHAP: 피처별 Mean Absolute SHAP 점수 계산 및 순위 정렬
        SHAP->>SHAP: turn_{turn}_{model}_shap_bar.png 저장
        SHAP->>SHAP: turn_{turn}_{model}_shap_summary.png 저장
        SHAP->>SHAP: turn_{turn}_{model}_shap_report.json 저장
        
        SHAP-->>Main: shap_report 메타데이터 반환
        deactivate SHAP
    end

    Main-->>User: 파이프라인 전체 완료 (성능 지표 및 SHAP 리포트 출력)
    deactivate Main
```
