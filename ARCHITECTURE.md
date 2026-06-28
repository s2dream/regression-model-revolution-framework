# AutoML Regression Framework Architecture

본 문서는 **Regression Model Revolution Framework**의 모듈 및 클래스 구조, 데이터 흐름, 디자인 패턴 적용 방식, 그리고 이들 간의 상호작용 관계를 텍스트 및 Mermaid 다이어그램과 함께 상세히 설명합니다.

---

## 1. High-Level Architecture Diagram (아키텍처 다이어그램)

아래 다이어그램은 프레임워크의 핵심 실행 제어 흐름과 데이터의 파이프라인 처리 과정을 시각화한 것입니다.

```text
                     ┌───────────────────────────────┐
                     │      configs/default.yml      │ (중앙 스키마 및 기본 설정 프로파일)
                     └───────┬───────────────┬───────┘
                             │               │
            ┌────────────────▼───────────────▼───────────────┐
            │            app.py (Streamlit WebUI)            │ (대화형 웹 인터페이스 스튜디오)
            │  - dynamic parameter & model widgets rendering │
            │  - real-time console log streaming             │
            │  - executive report expansion panel            │
            └────────────────┬───────────────────────────────┘
                             │configs/web_config.yml 생성 및 실행
                             ▼
                      ┌───────────────────────────────┐
                      │     CLI / User Entry Point    │
                      │         (root/main.py)        │
                      └───────────────┬───────────────┘
                                      │ 생성 및 실행 위임 (Facade)
                                      ▼
                      ┌───────────────────────────────┐
                      │        AutoMLPipeline         │
                      │   (Orchestrator inside main)  │
                      └──────┬────────┬────────────┬──┘
                            │        │            │
               ① Load &     │        │ ② Fit &    │ ③ Metrics, SHAP, Loss Curves
               Preprocess   │        │ Evalu-     │    for Executive Report
               Data         ▼        │ ate        ▼
                                     │
┌──────────────────────────────┐    │    ┌──────────────────────────────┐
│      DataLoaderHelper        │    │    │         Visualizer           │
│  (automl_framework/          │    │    │  (automl_framework/          │
│   dataloader/                │    │    │   util/visualizer.py)        │
│   data_loader_helper.py)     │    │    ├──────────────────────────────┤
├──────────────────────────────┤    │    │ - plot_actual_vs_predicted() │
│ - download_from_kaggle()     │    │    │ - plot_residuals()           │
│ - download_from_url()        │    │    │ - plot_learning_curve()      │
│ - preprocess_data()          │    │    │ - plot_shap_explainability() │
│ - split_data()               │    │    │ - save_markdown_report()     │
│ - prepare_data()             │    │    │ - save_json_report()         │
│  (Delegates to modular       │    │    └──────────────────────────────┘
│   strategies under-the-hood) │    │
└──────────────────────────────┘    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         StandardBenchmarkExecutor                         │
│                  (automl_framework/model/model_executor.py)               │
├───────────────────────────────────────────────────────────────────────────┤
│ - fit_all(X_train, y_train)                                               │
│ - evaluate_all(X_test, y_test) -> metrics (RMSE, MAE, R2)                 │
│ - get_predictions(X) -> dict of predictions                               │
└──────────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       │ Dynamically Configured Inventory 조작
                                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                 ModelPool                                 │
│                   (automl_framework/model/model_pool.py)                  │
├───────────────────────────────────────────────────────────────────────────┤
│ - models: Dict[str, ABCModelWrapper]                                      │
│ - _initialize_default_models() -> Delegates creation to ModelFactory      │
│ - add_custom_model(name, model_instance)                                  │
│ - list_available_models()                                                 │
└──────────────────────────────────────┬────────────────────────────────────┘
                                       │ Factory Method 적용
                                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                               ModelFactory                                │
│                  (automl_framework/model/model_factory.py)                │
├───────────────────────────────────────────────────────────────────────────┤
│ + create_model(model_type, config, random_state) -> ABCModelWrapper       │
│ - ModelType (Enum): XGBoost, MLP, TabPFN, RandomForest, CatBoost, etc.     │
└──────────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       │ Adapt & Standardize
                                       ▼
                    ┌──────────────────────────────────────┐
                    │             ModelWrapper             │
                    │   (automl_framework/model/wrappers.py)   │
                    ├──────────────────────────────────────┤
                    │ - fit(X, y)                          │
                    │ - predict(X)                         │
                    │ - get_loss_history()                 │
                    └──────────────────┬───────────────────┘
                                       │
               ┌───────────────┬───────┴───────┬───────────────┐ 인스턴스화 및 어댑팅
               ▼               ▼               ▼               ▼
      ┌─────────────────┐┌───────────────┐┌─────────────────┐┌─────────────────┐
      │   XGBoost / RF  ││   MLP (NN)    ││     TabPFN      ││   Transformer   │
      │ (xgboost/sklearn││(scikit-learn) ││    (tabpfn)     ││ (PyTorch Model) │
      └─────────────────┘└───────────────┘└─────────────────┘└─────────────────┘
```

---

## 2. 📐 디자인 패턴 적용 명세 (Design Patterns Mapping)

프레임워크의 고결성과 확장성을 극대화하기 위해 다음과 같은 4대 디자인 패턴이 적용되어 있습니다. 각 링크를 클릭하여 패턴에 대한 상세한 가이드를 학습하실 수 있습니다:

### 2.1. [팩토리 메서드 (Factory Method) 패턴](https://refactoring.guru/ko/design-patterns/factory-method)
- **적용처**: [ModelFactory](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/model_factory.py) 및 [ModelType](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/model_factory.py)
- **역할 및 이점**: 파이프라인에서 XGBoost, MLP, CatBoost 등 서로 다른 Regressor 클래스를 직접 하드코딩해 빌드하지 않고, `ModelType` 상수를 기반으로 초기 매개변수를 전개(`**kwargs`)하여 Wrapper 객체를 생산하는 책임을 Factory 객체로 완전히 일임합니다.

### 2.2. [전략 (Strategy) 패턴](https://refactoring.guru/ko/design-patterns/strategy)
- **적용처**: 데이터 분할 전략 클래스들([splitters.py](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/dataloader/splitters.py)) 및 일괄 벤치마크 실행기([model_executor.py](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/model_executor.py))
- **역할 및 이점**: 
  - 데이터셋을 나누는 구체적인 방식(`Holdout`, `KFold`, `TimeSeries`)을 캡슐화하여, 런타임에 동적으로 변경 가능하도록 다형적으로 주입합니다.
  - 모델 실행 흐름을 제어하는 `ABCModelExecutor` 인터페이스를 제공해, 기본형인 `StandardBenchmarkExecutor` 외에 향후 분산 학습 기법이 도입된 실행 방식으로의 전환을 지원합니다.

### 2.3. [퍼사드 (Facade) 패턴](https://refactoring.guru/ko/design-patterns/facade)
- **적용처**: [DataLoaderHelper](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/dataloader/data_loader_helper.py) 및 [AutoMLPipeline](file:///Users/jeonghoon/github/regression-model-revolution-framework/main.py)
- **역할 및 이점**: 
  - 복잡한 데이터 수집, 포맷 감지 로딩, 결측치 보정 전처리, 스플리팅 등 여러 하위 인스턴스를 조작해야 하는 절차를 `load_and_preprocess_data` 단일 호출 인터페이스로 감싸 제공합니다.
  - 최상단 `AutoMLPipeline`은 데이터 준비, 학습, 시각화, 아카이빙을 단 한 번의 기동으로 완수할 수 있는 통합 퍼사드 인터페이스의 역할을 수행합니다.

### 2.4. [어댑터 (Adapter) 패턴](https://refactoring.guru/ko/design-patterns/adapter)
- **적용처**: [ABCModelWrapper](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/wrappers.py) 및 하위 모델군 래퍼
- **역할 및 이점**: Scikit-Learn 계열, XGBoost, CatBoost, TabPFN, PyTorch 등 서로 다른 라이브러리 인터페이스 명세를 `fit(X, y)` 및 `predict(X)`라는 단일 규격에 부합하도록 일치시켜 다형적 실행을 뒷받침합니다.

---

## 3. 🗺️ 상세 호출 흐름 요약 (Call Sequences)

전체 실행 흐름의 세부적인 시퀀스 다이어그램은 단계별로 분리되어 기술된 [sequence_diagram.md](file:///Users/jeonghoon/github/regression-model-revolution-framework/sequence_diagram.md) 문서를 참고해 주시기 바랍니다. 

이하는 핵심 단계의 요약 흐름입니다:
1. **초기화 및 준비**: `main.py`가 설정을 판독하고 `ModelPool`에 모델들을 팩토리로 동적 생성 후, `DataLoaderHelper`를 통해 전처리 및 분할 완료된 $X$, $y$ 데이터셋을 적재합니다.
2. **최적화 및 학습**: HPO 활성화 조건에 따라 **Optuna**를 통해 대상 모델들의 최적 하이퍼파라미터를 탐색 후, `StandardBenchmarkExecutor`가 일괄 `fit()`을 안전하게 호출합니다. 이때 MLP, XGBoost, CatBoost, Transformer 모델은 이터레이션별 Loss 정보를 누적 기록합니다.
3. **진단 및 리포팅**: 실제값 vs 예측값 산포도, 잔차 분포, 모델별 학습 곡선(Loss Curve) 및 SHAP Beeswarm/Bar Plot 이미지를 순차 저장합니다. 최종적으로 데이터 구조, Leaderboard, 플롯 링크가 들어간 전문 Markdown 보고서(`turn_{turn}_report.md`)와 JSON 보고서를 디스크에 덤프합니다.

---

## 4. Directory Structure (디렉토리 구조)

[README.md](file:///Users/jeonghoon/github/regression-model-revolution-framework/README.md)의 디렉토리 구조 섹션을 참고해 주시기 바랍니다.

---

## 5. 핵심 모듈 및 클래스 명세 (Core Modules)

### A. 오케스트레이터: `main.py`
- **`AutoMLPipeline` 클래스**:
  - HPO 설정 및 전처리 데이터 형상 정보를 총괄 제어하며, 훈련을 마친 후 `Visualizer`를 활용해 모든 시각화 산출물과 전문 마크다운 리포트, 대시보드 상태 JSON 파일을 종합 생산해 저장하는 파이프라인 마스터 클래스입니다.

### B. 데이터 처리 퍼사드: `dataloader/data_loader_helper.py`
- **`DataLoaderHelper` 클래스**:
  - `LocalFileDataLoader` (CSV/Parquet/JSONL 판독 및 JSONL의 동적 열 정형화 복원 대행), `StandardDataPreprocessor` (Median/Mode 임퓨테이션 및 원-핫 인코더), `ABCDataSplitter` 전략 클래스군을 캡슐화해 훈련용 데이터셋을 통일되게 공급합니다.

### C. 모델 도메인: `model/`
- **`ModelPool` 클래스**: 설정 사양에 맞춘 활성 모델 래퍼 인스턴스들을 관리하는 순수 인벤토리 컨테이너입니다.
- **`ModelFactory` 클래스**: 입력 모델 상수에 부합하는 래퍼 클래스를 매개변수 전개와 함께 런타임에 빌드합니다.
- **`wrappers.py` 모듈**: Scikit-Learn 계열, XGBoost, CatBoost, TabPFN, PyTorch Transformer의 API 차이를 어댑터 기법으로 극복하며, 훈련 중 손실 수집을 위한 `.get_loss_history()` 명세를 상속 구조로 지원합니다.
- **`hpo.py` 모듈**: 최적화 목적 지표(RMSE, MAE, R2)에 부합하도록 탐색 방향을 지정하고 Optuna를 활용해 하이퍼파라미터를 자동 탐색합니다.
- **`StandardBenchmarkExecutor` 클래스**: 전체 모델에 대한 fit 및 predict, 평가지표 일괄 수집 흐름을 대행하는 실행 전략 객체입니다.

### D. 시각화 및 리포팅: `util/visualizer.py`
- **`Visualizer` 클래스**:
  - `plot_actual_vs_predicted()`, `plot_residuals()` 차트 렌더링.
  - `plot_learning_curve()`: 에포크 손실 감소 훈련 학습 곡선 렌더링.
  - `plot_shap_explainability()`: Beeswarm(요약 분포) 및 Bar(중요도) 시각화 렌더링.
  - `save_markdown_report()`: 상세한 전처리 정보, Leaderboard 테이블, 차트 이미지 링크 및 추천 사항이 담긴 전문 Markdown 리포트(`.md`) 자동 빌드 및 디스크 보존.
  - `save_json_report()`: 웹 UI 데이터 연동 및 마크다운 파일 경로를 바인딩하는 JSON 메타데이터 저장.

---

## 6. Technology Stack & Key Dependencies (기술 스택)

- **언어 및 직렬화**: Python 3.x, YAML (PyYAML)
- **머신러닝 & 알고리즘**: `scikit-learn`, `xgboost`, `catboost`, `tabpfn`, `torch`
- **시각화 및 보고서**: `matplotlib`, `seaborn`, `shap` (>= 0.42.0)
- **최적화**: `optuna` (>= 3.0.0)
- **대시보드**: `streamlit`
