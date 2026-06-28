# 📊 AutoML Regression Framework - 시퀀스 다이어그램 (Sequence Diagrams)

본 문서는 **Regression Model Revolution Framework**의 핵심 실행 흐름을 쉽게 이해할 수 있도록 **전체 요약 흐름도**와 **기능 단계별 세부 시퀀스 다이어그램**으로 분할하여 제공합니다.

---

## 1. 🏆 전체 요약 흐름도 (High-Level Pipeline Workflow)

이 다이어그램은 전체 실행의 전체 라이프사이클을 고수준에서 요약해 보여줍니다.

```mermaid
sequenceDiagram
    autonumber
    actor User as User/CLI
    participant Main as main.py (AutoMLPipeline)
    participant Loader as DataLoaderHelper
    participant Exec as StandardBenchmarkExecutor
    participant Vis as Visualizer

    User->>Main: run(dataset_path, config) 호출
    activate Main
    Main->>Loader: load_and_preprocess_data() 호출
    Loader-->>Main: 전처리 및 분할 완료된 데이터셋 반환
    
    Main->>Exec: fit_all(X_train, y_train) 일괄 학습
    activate Exec
    Note over Exec: HPO 선행 튜닝 및 학습 진행
    Exec-->>Main: 완료
    deactivate Exec
    
    Main->>Exec: evaluate_all(X_test, y_test) 일괄 평가
    Exec-->>Main: 모델별 평가지표 결과 반환
    
    Main->>Vis: plot_actual_vs_predicted / plot_residuals / SHAP / Loss Curves 저장
    Main->>Vis: save_markdown_report() & save_json_report() 실행
    Vis-->>Main: 최종 리포트 주소 반환
    Main-->>User: 최고 우승(Champion) 모델 출력하며 종료
    deactivate Main
```

---

## 2. 📥 [1단계] 초기화 및 데이터 전처리 (Initialization & Data Preparation)

설정 파일 파싱, `ModelPool`의 `ModelFactory` 기반 동적 생성, 그리고 원본 데이터 수집 및 정형 전처리/스플릿 분할 프로세스입니다.
- **적용 패턴**: 
  - **[팩토리 메서드 패턴]**: `ModelFactory`를 활용해 활성 모델 래퍼 인스턴스를 빌드합니다. ([Refactoring Guru 공식 설명 참조](https://refactoring.guru/ko/design-patterns/factory-method))
  - **[퍼사드 패턴]**: `DataLoaderHelper` 및 `AutoMLPipeline`이 하위 모듈 호출을 감싸 단일 진입점을 제공합니다. ([Refactoring Guru 공식 설명 참조](https://refactoring.guru/ko/design-patterns/facade))
  - **[전략 패턴]**: 분할 방식을 동적 교체합니다. ([Refactoring Guru 공식 설명 참조](https://refactoring.guru/ko/design-patterns/strategy))

```mermaid
sequenceDiagram
    autonumber
    actor User as User/CLI
    participant Main as main.py (AutoMLPipeline)
    participant Pool as ModelPool
    participant Factory as ModelFactory
    participant Helper as DataLoaderHelper
    participant Loader as LocalFileDataLoader
    participant Prep as StandardDataPreprocessor
    participant Split as TrainTestSplitter/KFold

    User->>Main: AutoMLPipeline 생성 (config_path, turn, target)
    activate Main
    Main->>Helper: DataLoaderHelper 생성
    Main->>Pool: ModelPool 생성 (config)
    activate Pool
    loop active_models 이름 순회
        Pool->>Factory: create_model(model_type, config, random_state)
        activate Factory
        Note over Factory: 관련 패키지 동적 임포트 및 래퍼 조립
        Factory-->>Pool: 구체적인 ModelWrapper 반환
        deactivate Factory
        Pool->>Pool: self.models에 보관
    end
    Pool-->>Main: 반환
    deactivate Pool
    Main-->>User: 파이프라인 초기화 완료
    deactivate Main

    User->>Main: run(dataset_path, kaggle_dataset, url) 호출
    activate Main
    Main->>Helper: load_and_preprocess_data(dataset_file, target)
    activate Helper
    Helper->>Loader: load_data() 호출
    Note over Loader: CSV/Parquet/JSONL 등 포맷 파싱<br/>JSONL의 누락된 키 자동 정형화
    Loader-->>Helper: raw X, y 데이터프레임 반환
    Helper->>Prep: preprocess(X) 호출
    Note over Prep: 결측치 보정 (Median/Mode)<br/>범주형 원-핫 인코딩
    Prep-->>Helper: 전처리 완료된 X_processed 반환
    Helper->>Split: split(X_processed, y) 호출
    Note over Split: 설정된 분할 전략(Holdout/KFold/TimeSeries) 실행
    Split-->>Helper: 분할 완료된 데이터셋 반환
    Helper-->>Main: X_train, y_train, X_test, y_test 반환
    deactivate Helper
    deactivate Main
```

---

## 3. 🎯 [2단계] 하이퍼파라미터 최적화 (Hyperparameter Optimization - HPO)

파이프라인 실행 시 HPO가 활성화되었을 때 **Optuna**를 활용해 가변 평가지표를 기반으로 최적의 모델 하이퍼파라미터를 탐색 및 갱신하는 단계입니다.
- **적용 패턴**: 
  - **[팩토리 메서드 패턴]**: 튜닝용 임시 인스턴스 빌드에 사용됩니다.

```mermaid
sequenceDiagram
    autonumber
    participant Main as main.py (AutoMLPipeline)
    participant Exec as StandardBenchmarkExecutor
    participant Pool as ModelPool
    participant Factory as ModelFactory
    participant Optuna as Optuna Optimizer

    Main->>Exec: fit_all(X_train, y_train) 호출
    activate Exec
    Note over Exec: HPO 설정이 True인 경우 선행 수행
    Exec->>Pool: run_hpo_tuning(pool, X_train, y_train) 호출
    activate Pool
    Note over Pool: 튜닝을 위한 3-way Validation Split (80/20) 생성
    
    loop HPO 대상 활성 모델 순회 (TabPFN 등 자동 제외)
        Pool->>Optuna: optimize() 기동
        activate Optuna
        loop 1..n_trials 횟수만큼 반복
            Optuna->>Factory: create_model(trial_config) 임시 생성
            Factory-->>Optuna: 래퍼 반환
            Optuna->>Optuna: Validation Metric 계산 (RMSE, MAE, R2 중 선택 지표)
        end
        Optuna-->>Pool: 최적의 하이퍼파라미터 설정 반환
        deactivate Optuna
        
        Pool->>Factory: create_model(best_config) 최적 객체 생성
        Factory-->>Pool: 최적화된 래퍼 반환
        Pool->>Pool: self.models[name]을 최적 객체로 교체 업데이트
    end
    Pool-->>Exec: HPO 최적화 완료 (ModelPool 갱신)
    deactivate Pool
    deactivate Exec
```

---

## 4. 🤖 [3단계] 일괄 학습 및 스코어링 (Model Training & Evaluation)

갱신된 모델 풀을 대상으로 학습을 수행하고, 테스트 데이터를 활용해 최종 평가지표를 도출합니다.
- **적용 패턴**: 
  - **[어댑터 패턴]**: 외부 머신러닝 모듈들의 규격을 표준 규격으로 정렬합니다. ([Refactoring Guru 공식 설명 참조](https://refactoring.guru/ko/design-patterns/adapter))
  - **[전략 패턴]**: 교체 가능한 실행 전략으로 다형적으로 운영합니다.

```mermaid
sequenceDiagram
    autonumber
    participant Main as main.py (AutoMLPipeline)
    participant Exec as StandardBenchmarkExecutor
    participant Wrapper as ModelWrapper (예: XGBoost)

    Main->>Exec: fit_all(X_train, y_train) 호출
    activate Exec
    loop ModelPool 내 활성 모델 순회
        Exec->>Wrapper: fit(X_train, y_train) 호출
        activate Wrapper
        Note over Wrapper: 결함 방지(Exception-shielding) 하에 학습 수행<br/>반복 모델은 이터레이션별 Loss를 누적 기록
        Wrapper-->>Exec: 학습 완료
        deactivate Wrapper
    end
    Exec-->>Main: 일괄 학습 완료
    deactivate Exec

    Main->>Exec: evaluate_all(X_test, y_test) 호출
    activate Exec
    loop 학습 완료된 각 모델 순회
        Exec->>Wrapper: predict(X_test) 호출
        activate Wrapper
        Wrapper-->>Exec: y_pred 예측 어레이 반환
        deactivate Wrapper
        Exec->>Exec: 회귀 평가지표(RMSE, MAE, R² Score) 산출
    end
    Exec-->>Main: 모델별 최종 메트릭 결과 Dict 반환
    deactivate Exec
```

---

## 5. 🧠 [4단계] 시각화, SHAP 분석 및 보고서 작성 (Visualization, SHAP & Reporting)

평가가 종료된 후 결과 차트, SHAP 변수 중요도 분석, 학습 곡선(Loss Curve) 이미지를 생성하고 전문 보고서 파일 2종을 디스크에 저장합니다.

```mermaid
sequenceDiagram
    autonumber
    participant Main as main.py (AutoMLPipeline)
    participant Vis as Visualizer
    participant Exec as StandardBenchmarkExecutor
    participant Wrapper as ModelWrapper

    Main->>Main: generate_reports() 기동
    
    loop 각 활성 모델 순회
        Main->>Vis: plot_actual_vs_predicted() 호출
        Note over Vis: 실제값 vs 예측값 산포도 PNG 저장
        Main->>Vis: plot_residuals() 호출
        Note over Vis: 등분산성 진단용 잔차 분포 PNG 저장
        
        opt 학습 이력(Loss History)이 있는 경우
            Main->>Vis: plot_learning_curve() 호출
            Note over Vis: 학습 손실 곡선(Loss Curve) PNG 저장
        end
    end
    
    opt Explainability(SHAP) 설정 활성화 시
        loop 분석 대상 모델 순회
            Main->>Vis: plot_shap_explainability() 호출
            activate Vis
            Note over Vis: TreeExplainer / agnostic Explainer 연동<br/>Beeswarm Plot 및 Bar Plot PNG 저장
            Vis-->>Main: SHAP 플롯 파일 주소 반환
            deactivate Vis
        end
    end

    Main->>Vis: save_markdown_report(dataset_info, shap_reports, learning_curves) 호출
    Note over Vis: Leaderboard 테이블, 차트 링크, Recommendations가 포함된<br/>turn_report.md 보고서 파일 작성
    Vis-->>Main: 저장된 마크다운 보고서 절대 경로 반환

    Main->>Vis: save_json_report(metrics, markdown_report_path) 호출
    Note over Vis: 대시보드 연동용 JSON 결과 파일 작성
    Vis-->>Main: 저장된 JSON 보고서 경로 반환
    
    Note over Main: 최종 최우수 Champion 모델 정보를 출력하며 전체 완료
```
