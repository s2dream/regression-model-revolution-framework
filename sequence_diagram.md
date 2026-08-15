# 📊 AutoML Regression Framework - Comprehensive Sequence Diagrams

본 문서는 **AutoML Regression Framework**의 파이프라인 전체 실행 제어 흐름, Optuna HPO 하이퍼파라미터 최적화 루프, 그리고 Streamlit WebUI 상호작용 흐름을 Mermaid 시퀀스 다이어그램으로 상세히 기술한 **동적 모델 설계서**입니다.

---

## 1. End-to-End AutoML Pipeline Sequence Diagram

CLI 또는 라이브러리 호출 시 `AutoMLPipeline`을 중심으로 초기화 -> 데이터 준비 -> Optuna HPO & 모델 훈련 -> 성능 평가 -> 프리미엄 차트 생성 및 리포트 저장까지의 전체 순차 흐름입니다.

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

    %% ==========================================
    %% 1. INITIALIZATION PHASE
    %% ==========================================
    Note over User, Main: [1. 초기화 단계 - Initialization Phase]
    User->>Main: AutoMLPipeline(config_path, turn, target, test_size)
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
        Note over Factory: 동적 임포트 및 알고리즘별<br/>Concrete ModelWrapper 인스턴스화
        Factory-->>Pool: concrete ModelWrapper 인스턴스 반환
        deactivate Factory
        
        Pool->>Pool: self.models[model_type.value]에 래퍼 저장
    end
    Pool-->>Main: ModelPool 인스턴스 반환
    deactivate Pool
    
    Main->>Exec: StandardBenchmarkExecutor(pool)
    Main->>Vis: Visualizer(output_dir)
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
    Note over Loader: CSV/TSV/Parquet/JSONL 파싱<br/>JSONL 누락 키 동적 병합 및 NaN 매핑
    Loader-->>Helper: 원본 X (DataFrame), y (Series) 반환
    deactivate Loader

    Helper->>Prep: preprocess(X)
    activate Prep
    Note over Prep: 수치형 Median, 범주형 Mode 결측치 대체<br/>drop_first=True 원-핫 인코딩
    Prep-->>Helper: 전처리 완료된 X_processed 반환
    deactivate Prep

    Helper->>Split: split(X_processed, y)
    activate Split
    Note over Split: 난수 시드 기반 학습/테스트 분할
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
            loop 1..n_trials 반복 (TPE 베이지안 최적화)
                HPO->>Factory: create_model(model_type, trial_config, random_state)
                activate Factory
                Factory-->>HPO: Trial용 ModelWrapper 반환
                deactivate Factory
                HPO->>Wrapper: fit & validation RMSE 스코어링
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
        Note over Wrapper: 모델별 훈련 수행 (예외 감내 쉴딩)
        Wrapper-->>Exec: 학습 완료
        deactivate Wrapper
    end
    Exec-->>Main: 전체 모델 학습 완료
    deactivate Exec

    %% ==========================================
    %% 4. EVALUATION PHASE
    %% ==========================================
    Note over Main, Exec: [4. 성능 평가 단계 - Evaluation Phase]
    Main->>Exec: evaluate_all(X_test, y_test)
    activate Exec
    loop 학습 완료된 각 모델 순회
        Exec->>Wrapper: predict(X_test)
        activate Wrapper
        Wrapper-->>Exec: 예측값 y_pred 반환
        deactivate Wrapper
        Exec->>Exec: 회귀 평가 메트릭 (RMSE, MAE, R²) 계산
    end
    Exec-->>Main: metrics 딕셔너리 반환
    deactivate Exec

    %% ==========================================
    %% 5. VISUALIZATION & REPORTING PHASE
    %% ==========================================
    Note over Main, Vis: [5. 시각화 및 리포트 저장 단계 - Visuals & Reports]
    Main->>Main: generate_reports()
    Main->>Vis: plot_model_comparison(metrics, metric_name, turn)
    Note over Vis: 모델간 성능 비교 수평 막대 차트 PNG 저장
    
    Main->>Exec: get_predictions(X_test)
    activate Exec
    Exec-->>Main: 전체 모델의 y_pred 딕셔너리 반환
    deactivate Exec

    loop 각 모델별 예측값 순회
        Main->>Vis: plot_actual_vs_predicted(y_test, y_pred, model_name, turn)
        Note over Vis: 실제값 vs 예측값 산포도 및 y=x 라인 저장
        Main->>Vis: plot_residuals(y_test, y_pred, model_name, turn)
        Note over Vis: 잔차 분포 산포도 저장
    end

    Main->>Vis: save_json_report(metrics, turn)
    Vis-->>Main: turn_{turn}_report.json 저장 경로 반환
    
    Main-->>User: 최고 성능 챔피언 모델 출력 및 결과 반환
    deactivate Main
```

---

## 2. Streamlit WebUI Interaction Sequence Diagram

사용자가 Streamlit WebUI(`app.py`)에서 설정을 조작하고 실험을 실행하여 실시간 결과를 확인하는 상호작용 흐름입니다.

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant App as app.py (Streamlit UI)
    participant State as st.session_state
    participant Runner as Subprocess Runner
    participant Pipeline as main.py (AutoMLPipeline)
    participant Storage as outputs/ File System

    User->>App: 브라우저 접속 (사이드바 뷰 선택: 🚀 Run Experiment)
    activate App
    App->>State: 현재 실험 구성(데이터셋, 컬럼, HPO) 로드/동기화
    
    User->>App: "🚀 Start AutoML Training" 버튼 클릭
    App->>App: configs/web_config.yml 파일 컴파일 및 디스크 저장
    
    App->>Runner: subprocess.Popen(["python", "main.py", "--config", "configs/web_config.yml"])
    activate Runner
    
    loop 파이프라인 구동 중 (실시간 로그 스트리밍)
        Runner->>Pipeline: 실행 진행
        Pipeline-->>Runner: stdout 로그 라인 방출
        Runner-->>App: 실시간 콘솔 로그 스트리밍 렌더링
    end
    
    Runner-->>App: 프로세스 종료 (Exit Code 0)
    deactivate Runner
    
    App->>Storage: outputs/turn_*_report.json 및 PNG 차트 파일 탐색
    activate Storage
    Storage-->>App: JSON 리포트 데이터 및 유효한 차트 이미지 반환
    deactivate Storage
    
    App->>App: is_valid_image() 검증 후 챔피언 모델 요약, 성적표 테이블, 프리미엄 차트 갤러리 렌더링
    App-->>User: 인터랙티브 결과 대시보드 화면 표시
    deactivate App
```
