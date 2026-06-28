# 📊 AutoML Regression Framework - 전체 시퀀스 다이어그램 (Full Sequence Diagram)

본 문서는 모델 생성을 위해 새로 리팩토링된 **팩토리 메서드(Factory Method)** 디자인 패턴을 포함하여, **Regression Model Revolution Framework**의 전체 파이프라인 워크플로우를 보여주는 종합 시퀀스 다이어그램을 제공합니다.

---

## 🗺️ Mermaid 시퀀스 다이어그램

아래는 초기화, 데이터 로드/전처리, 학습, 평가 및 시각화 프로세스를 설명하는 전체 시퀀스 다이어그램입니다.

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
    participant Wrapper as ModelWrapper (예: XGBoost)
    participant Vis as Visualizer

    %% ==========================================
    %% 1. 초기화 단계 (INITIALIZATION PHASE)
    %% ==========================================
    Note over User, Main: [1. 초기화 단계]
    User->>Main: AutoMLPipeline(config_path, turn, target, test_size) 인스턴스 생성
    activate Main
    Main->>Helper: DataLoaderHelper(data_dir, config) 인스턴스 생성
    
    Main->>Pool: ModelPool(random_state, config) 인스턴스 생성
    activate Pool
    Pool->>Pool: _initialize_default_models() 호출
    
    loop 설정 파일 내 각 활성 모델 이름 순회 (active_models: "XGBoost", "MLP" 등)
        Pool->>Type: ModelType.from_str(model_name) 호출을 통해 문자열 변환
        activate Type
        Type-->>Pool: ModelType Enum 멤버 반환
        deactivate Type
        
        Pool->>Factory: create_model(model_type, config, random_state) 호출
        activate Factory
        Note over Factory: 관련 패키지 동적 임포트 및<br/>개별 Regressor 객체 빌드
        Factory-->>Pool: 구체적인 ModelWrapper 인스턴스 반환
        deactivate Factory
        
        Pool->>Pool: self.models[model_type.value]에 래퍼 저장
    end
    Pool-->>Main: ModelPool 인스턴스 반환
    deactivate Pool
    
    Main->>Exec: StandardBenchmarkExecutor(pool) 인스턴스 생성
    Main->>Vis: Visualizer(output_dir) 인스턴스 생성
    Main-->>User: 파이프라인 초기화 완료
    deactivate Main

    %% ==========================================
    %% 2. 데이터 준비 단계 (DATA PREPARATION PHASE)
    %% ==========================================
    Note over User, Main: [2. 데이터 수집 및 전처리 단계]
    User->>Main: pipeline.run(dataset_path, kaggle_dataset, url) 호출
    activate Main
    Main->>Main: prepare_data(dataset_path, kaggle_dataset, url) 실행
    Main->>Helper: fetch_dataset(...) 호출
    Helper-->>Main: 확인된 로컬 데이터셋 파일 경로 반환
    
    Main->>Helper: load_and_preprocess_data(dataset_file, target, test_size, random_state) 호출
    activate Helper
    
    Helper->>Loader: load_data() 호출
    activate Loader
    Note over Loader: 파일 포맷 파싱 (CSV/TSV/Parquet/JSONL)<br/>JSONL의 누락된 키 및 동적 컬럼 해결
    Loader-->>Helper: 원본 DataFrame X, Series y 반환
    deactivate Loader

    Helper->>Prep: preprocess(X) 호출
    activate Prep
    Note over Prep: 결측치 보정 (Median/Mode)<br/>원-핫 더미 인코딩 (One-Hot Dummy Encoding)
    Prep-->>Helper: 전처리 완료된 DataFrame X_processed 반환
    deactivate Prep

    Helper->>Split: split(X_processed, y) 호출
    activate Split
    Note over Split: random_state 기준으로 Train/Test 분할 수행
    Split-->>Helper: 분할된 X_train, y_train, X_test, y_test 반환
    deactivate Split

    Helper-->>Main: X_train, y_train, X_test, y_test 반환
    deactivate Helper

    %% ==========================================
    %% 3. 모델 학습 및 평가 단계 (MODEL TRAINING & SCORING PHASE)
    %% ==========================================
    Note over Main, Exec: [3. 모델 학습 및 스코어링 단계]
    Main->>Main: train_and_evaluate() 실행
    Main->>Exec: fit_all(X_train, y_train) 호출
    activate Exec

    opt HPO (Optuna) 활성화 시
        Exec->>Pool: run_hpo_tuning(pool, X_train, y_train) 호출
        activate Pool
        loop 모델 풀 내 각 활성 모델 순회 (TabPFN 제외)
            loop 1..n_trials 횟수만큼 반복
                Pool->>Factory: create_model(model_type, trial_config, random_state) 호출
                activate Factory
                Factory-->>Pool: 임시 trial 래퍼 인스턴스 반환
                deactivate Factory
                Pool->>Pool: Train 분할로 학습 및 validation RMSE 점수 검증
            end
            Pool->>Factory: create_model(model_type, best_config, random_state) 호출
            activate Factory
            Factory-->>Pool: 최적 파라미터 래퍼 인스턴스 반환
            deactivate Factory
            Pool->>Pool: 최적 래퍼로 self.models[name] 업데이트
        end
        Pool-->>Exec: HPO 최적화 완료
        deactivate Pool
    end

    loop ModelPool 내 각 모델 Wrapper 순회
        Exec->>Pool: 모델 래퍼 인스턴스 조회
        Exec->>Wrapper: fit(X_train, y_train) 호출
        activate Wrapper
        Note over Wrapper: 예외 감내 쉴딩(Exception-shielding) 하에<br/>개별 모델 학습
        Wrapper-->>Exec: 완료
        deactivate Wrapper
    end
    Exec-->>Main: 일괄 학습 완료
    deactivate Exec

    Main->>Exec: evaluate_all(X_test, y_test) 호출
    activate Exec
    loop 학습 완료된 각 모델 Wrapper 순회
        Exec->>Wrapper: predict(X_test) 호출
        activate Wrapper
        Wrapper-->>Exec: 예측값 y_pred 배열 반환
        deactivate Wrapper
        Exec->>Exec: 평가 지표(RMSE, MAE, R2 Score) 계산
    end
    Exec-->>Main: 지표 결과 Dict 반환
    deactivate Exec

    %% ==========================================
    %% 4. 시각화 및 리포팅 단계 (VISUALIZATION & REPORTING PHASE)
    %% ==========================================
    Note over Main, Vis: [4. 시각화 및 리포트 파일 저장 단계]
    Main->>Main: generate_reports() 실행
    Main->>Vis: plot_model_comparison(metrics, metric_name, turn) 호출
    Note over Vis: 모델간 R2 및 RMSE 비교 수평 바 차트 저장
    
    Main->>Exec: get_predictions(X_test) 호출
    activate Exec
    loop ModelPool 내 각 모델 Wrapper 순회
        Exec->>Wrapper: predict(X_test) 호출
        activate Wrapper
        Wrapper-->>Exec: 예측값 y_pred 배열 반환
        deactivate Wrapper
    end
    Exec-->>Main: 전체 예측값 Dict 반환
    deactivate Exec

    loop 각 모델 예측값 순회
        Main->>Vis: plot_actual_vs_predicted(y_test, y_pred, model_name, turn) 호출
        Note over Vis: 실제값 vs 예측값 산포도 및 y=x 기준선 저장
        Main->>Vis: plot_residuals(y_test, y_pred, model_name, turn) 호출
        Note over Vis: 등분산성 검증용 잔차 분석 플롯 저장
    end

    Main->>Vis: save_json_report(metrics, turn) 호출
    Vis-->>Main: 저장된 turn_report.json 파일 경로 반환
    
    Main-->>User: 파이프라인 실행 종료 (우승 Champion 모델 출력)
    deactivate Main
```
