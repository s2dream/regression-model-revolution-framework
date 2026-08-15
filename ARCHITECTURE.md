# AutoML Regression Framework Architecture

본 문서는 **AutoML Regression Framework**의 시스템 아키텍처, 소프트웨어 모듈 및 클래스 구조, 데이터 흐름, 디자인 패턴, 그리고 이들 간의 상호작용 관계를 상세히 정의한 **소프트웨어 아키텍처 설계서 (SAD)**입니다.

---

## 1. High-Level Architecture Diagram (아키텍처 다이어그램)

프레임워크의 핵심 실행 제어 흐름과 데이터 파이프라인 처리 과정은 아래와 같습니다.

```text
                     ┌───────────────────────────────┐
                     │      configs/default.yml      │ (Central Schema & Config Profiles)
                     └───────┬───────────────┬───────┘
                             │               │
            ┌────────────────▼───────────────▼───────────────┐
            │            app.py (Streamlit WebUI)            │ (Interactive Web Studio)
            │  - 4-View Sidebar (Overview, Run, Studio, Hist)│
            │  - Dynamic Parameter Rendering / Schema Sync   │
            │  - Subprocess Real-time Log Streamer           │
            └────────────────┬───────────────────────────────┘
                             │ Generates configs/web_config.yml & Executes
                             ▼
                     ┌───────────────────────────────┐
                     │     CLI / User Entry Point    │
                     │         (root/main.py)        │
                     └───────────────┬───────────────┘
                                     │ Instantiates & Runs
                                     ▼
                     ┌───────────────────────────────┐
                     │        AutoMLPipeline         │
                     │   (Orchestrator inside main)  │
                     └──────┬────────┬────────────┬──┘
                           │        │            │
              ① Load &     │        │ ② Fit &    │ ③ Metrics & Predictions
              Preprocess   │        │ Evalu-     │    for Premium Reports
              Data         ▼        │ ate        ▼
                                    │
┌──────────────────────────────┐    │    ┌──────────────────────────────┐
│      DataLoaderHelper        │    │    │         Visualizer           │
│  (automl_framework/          │    │    │  (automl_framework/          │
│   dataloader/                │    │    │   util/visualizer.py)        │
│   data_loader_helper.py)     │    │    ├──────────────────────────────┤
├──────────────────────────────┤    │    │ - plot_actual_vs_predicted() │
│ - download_from_kaggle()     │    │    │ - plot_residuals()           │
│ - download_from_url()        │    │    │ - plot_model_comparison()    │
│ - load_dataset()             │    │    │ - save_json_report()         │
│ - preprocess_data()          │    │    └──────────────────────────────┘
│ - split_data()               │    │
│ - prepare_data()             │    │
│  (Delegates to modular       │    │
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
└───────────────────┬───────────────────────────────────┬───────────────────┘
                    │                                   │
                    │ Optional HPO Optimization         │ Operates on Inventory
                    ▼                                   ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────┐
│            OptunaHPOTuner             │   │           ModelPool           │
│ (automl_framework/model/hpo.py)       │   │(automl_framework/model/       │
├───────────────────────────────────────┤   │ model_pool.py)                │
│ - tune_model(model_type, pool, ...)   │   ├───────────────────────────────┤
│ - run_hpo_tuning(pool, X_tr, y_tr)    │   │ - models: Dict[str, Wrapper]  │
│ - objective(trial, model_type, ...)   │   │ - _initialize_default_models()│
└───────────────────┬───────────────────┘   │ - add_custom_model()          │
                    │ Rebuilds Best Models  │ - list_available_models()     │
                    └───────────────────►   └───────────────┬───────────────┘
                                                            │ Uses Factory
                                                            ▼
                    ┌───────────────────────────────────────────────────────┐
                    │                     ModelFactory                      │
                    │       (automl_framework/model/model_factory.py)       │
                    ├───────────────────────────────────────────────────────┤
                    │ + create_model(model_type, config, random_state)      │
                    │ - ModelType (Enum): XGBoost, MLP, TabPFN, TabICL,     │
                    │                     RandomForest, CatBoost, etc.      │
                    └───────────────────────────────┬───────────────────────┘
                                                    │
                                                    │ Adapts via Standard Wrapper
                                                    ▼
                                ┌───────────────────────────────────────┐
                                │             ModelWrapper              │
                                │   (automl_framework/model/wrappers.py)│
                                ├───────────────────────────────────────┤
                                │ - fit(X, y)                           │
                                │ - predict(X)                          │
                                └───────────────────┬───────────────────┘
                                                    │
        ┌───────────────┬───────────────┬───────────┴───┬───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼               ▼
┌───────────────┐┌───────────────┐┌───────────┐┌─────────────────┐┌───────────┐┌─────────────────┐
│ XGBoost / RF  ││   CatBoost    ││  MLP (NN) ││     TabPFN      ││  TabICL   ││   Transformer   │
│(sklearn/xgbr) ││  (catboost)   ││(sklearn)  ││    (tabpfn)     ││ (tabicl)  ││ (PyTorch Model) │
└───────────────┘└───────────────┘└───────────┘└─────────────────┘└───────────┘└─────────────────┘
```

---

## 2. Directory Structure (디렉토리 구조)

프로젝트 루트 디렉토리의 레이아웃과 소스 파일 위치는 다음과 같습니다.

```text
regression-model-revolution-framework/
│
├── main.py                         # 프로젝트 전체 실행 진입점 (CLI Orchestrator & AutoMLPipeline)
├── app.py                          # Streamlit 기반 대화형 웹 인터페이스 스튜디오 (WebUI Studio)
│
├── configs/                        # 📂 설정 프로파일 보관소 (실험 목적별 YAML 설정 파일)
│   ├── default.yml                 # 기본 통합 설정 프로파일 (모델별 파라미터, HPO, 프레임워크 설정)
│   ├── kfold_split.yml             # 교차 검증(K-Fold Split) 실험 설정 프로파일
│   ├── timeseries_split.yml        # 시계열 분할(TimeSeries Split) 실험 설정 프로파일
│   ├── custom_features.yml         # 커스텀 피처 변수 지정 실험 설정 프로파일
│   └── web_config.yml              # Web UI 실행에 의해 자동 컴파일되는 런타임 프로파일
│
├── scripts/                        # 🏃 시나리오별 파이프라인 일괄 실행 스크립트 디렉토리
│   ├── run_local_csv.sh            # 로컬 CSV 데이터셋 학습 실행기
│   ├── run_local_jsonl.sh          # 로컬 JSONL 데이터셋(동적 컬럼 지원) 학습 실행기
│   ├── run_url.sh                  # 원격 HTTP URL 파일 다운로드 후 학습 실행기
│   └── run_webui.sh                # Streamlit Web UI 기동 실행기
│
├── automl_framework/               # 📦 프레임워크 메인 패키지
│   ├── __init__.py                 # 패키지 파사드 진입점 (핵심 모듈 클래스 외부 노출)
│   │
│   ├── dataloader/                 # 📂 데이터 처리 서브패키지 (Data Domain)
│   │   ├── __init__.py
│   │   ├── loaders.py              # 데이터 로더 추상 베이스 클래스 및 로컬/Kaggle/URL 로더 구현체
│   │   ├── preprocessors.py        # 전처리기 추상 베이스 클래스 및 결측치/원-핫 인코딩 구현체
│   │   ├── splitters.py            # 데이터셋 분할기 추상 베이스 클래스 (TrainTest, KFold, TimeSeries)
│   │   └── data_loader_helper.py   # 퍼사드(Facade) DataLoaderHelper 클래스
│   │
│   ├── model/                      # 📂 머신러닝 학습 서브패키지 (Model Domain)
│   │   ├── __init__.py
│   │   ├── model_pool.py           # 모델 인벤토리 저장소 (ModelPool)
│   │   ├── model_factory.py        # 모델 팩토리 (ModelFactory) 및 상수 정의 (ModelType)
│   │   ├── model_executor.py       # 추상 실행기(ABCModelExecutor) 및 표준 벤치마커(StandardBenchmarkExecutor)
│   │   ├── wrappers.py             # 모델 어댑터 Wrapper 구현체들 (XGB, RF, CatBoost, MLP, TabPFN, TabICL, NN)
│   │   ├── hpo.py                  # Optuna 기반 하이퍼파라미터 자동 최적화 튜너 (OptunaHPOTuner)
│   │   └── architecture/           # 딥러닝/신경망 모델 커스텀 아키텍처
│   │       └── transformer_encoder.py # 시퀀스 기반 트랜스포머 회귀 신경망 (TransformerBasedRegression)
│   │
│   ├── util/                       # 📂 분석/유틸리티 서브패키지 (Utility Domain)
│   │   ├── __init__.py
│   │   ├── visualizer.py           # 프리미엄 다크 테마 차트 생성 및 JSON 리포트 작성
│   │   └── logger.py               # 콘솔/파일 로깅 설정 모듈
│   │
│   └── README.md                   # 패키지 명세서
│
├── tests/                          # 🧪 종합 테스트 스위트 (Unit & Integration Tests)
│   ├── __init__.py
│   ├── test_dataloader.py          # 데이터 로더, JSONL 동적 스키마 로딩 및 분할 기능 테스트
│   ├── test_model.py               # 모델 초기화, 수동 등록 및 실행기 기본 테스트
│   ├── test_model_factory.py       # ModelFactory 인스턴스화 및 ModelType 파싱 테스트
│   ├── test_all_models.py          # 전체 활성 회귀 모델 Wrapper 학습/추론 단위 테스트
│   ├── test_tabicl.py              # TabICL In-Context Learning 회귀 모델 테스트
│   ├── test_transformer_regression.py # PyTorch 트랜스포머 회귀 모델 및 확률 모드 테스트
│   ├── test_hpo.py                 # Optuna HPO 튜닝 및 파라미터 업데이트 테스트
│   ├── test_visualizer.py          # 시각화 플롯 생성 및 JSON 리포트 작성 테스트
│   ├── test_logger.py              # 로거 구성 및 로그 파일 기록 테스트
│   ├── test_webui_helpers.py       # WebUI 헬퍼 함수, 스키마 플래트닝 및 미디어 검증 테스트
│   └── test_pipeline_e2e.py        # 모의 데이터셋 기반 End-to-End 전체 파이프라인 통합 테스트
│
├── data/                           # 📂 (데이터 저장소) 모의 데이터셋 및 벤치마크 데이터
├── outputs/                        # 📂 (결과 저장소) 시각화 이미지(.png) 및 JSON 실행 리포트
├── ARCHITECTURE.md                 # [본 문서] 시스템 아키텍처 설계서
├── REQ_SPEC.md                     # 소프트웨어 요구사항 명세서
├── sequence_diagram.md             # 세부 상호작용 시퀀스 다이어그램 문서
├── TEST_SPEC.md                    # 소프트웨어 테스트 계획 및 명세서
├── INTERFACE_SPEC.md               # 프레임워크 API 및 인터페이스 정의서
└── README.md                       # 프로젝트 개요 및 빠른 시작 가이드
```

---

## 3. Core Modules & Classes (핵심 모듈 및 클래스 구성)

### A. CLI 및 파이프라인 오케스트레이션: `main.py`
- **`AutoMLPipeline` (Class)**: 데이터 수집, 전처리, 모델 초기화, HPO 튜닝, 일괄 학습, 성능 평가, 프리미엄 시각화 및 리포트 파일 아카이빙까지의 전체 생명주기를 조율하는 마스터 오케스트레이터입니다.
  - `__init__(config_path, turn, target, test_size)`: 설정 파일을 로드하고 CLI 인수를 오버라이드하여 각 도메인 컴포넌트들을 바인딩합니다.
  - `prepare_data(dataset_path, kaggle_dataset, url)`: 데이터 로딩, 전처리, 분할을 수행합니다.
  - `train_and_evaluate() -> dict`: 모델 풀에 대해 HPO 튜닝 및 학습/평가를 수행하고 지표를 수집합니다.
  - `generate_reports()`: Visualizer를 통해 산포도, 잔차도, 바 차트 및 최종 성적 JSON 리포트를 생성합니다.
  - `run(...)`: 파이프라인 전체를 원클릭으로 순차 실행합니다.

---

### B. 데이터 로더 및 전처리: `automl_framework/dataloader/`
- **`DataLoaderHelper` (Facade Class)**: `loaders.py`, `preprocessors.py`, `splitters.py`의 구현체들을 조합하여 클라이언트에게 단순화된 단일 인터페이스(`fetch_dataset`, `prepare_data`)를 제공합니다.
- **`LocalFileDataLoader`**: CSV, TSV, Parquet, JSONL 파일을 로드합니다. JSON Lines(`.jsonl`) 포맷에 대해 누락된 키를 자동 정렬/확장하고 `NaN`을 매핑하는 동적 스키마 로딩 기능을 내장합니다.
- **`StandardDataPreprocessor`**: 수치형 결측치는 Median, 범주형 결측치는 Mode로 임퓨팅한 뒤, `drop_first=True` 옵션으로 One-Hot Dummy 인코딩을 수행합니다.
- **`TrainTestSplitter`, `KFoldSplitter`, `TimeSeriesSplitter`**: 전략 패턴을 적용하여 재현 가능한 난수 시드 기반 데이터 분할을 수행합니다.

---

### C. 모델 관리, 팩토리 및 실행기: `automl_framework/model/`
- **`ModelType` (Enum, `model_factory.py`)**: 지원되는 모든 모델의 표준 식별자(`XGBOOST`, `CATBOOST`, `RANDOM_FOREST`, `MLP`, `TABPFN`, `TABICL`, `TRANSFORMER`)를 정의하며, `from_str()`을 통해 문자열을 안전하게 Enum으로 변환합니다.
- **`ModelFactory` (Class, `model_factory.py`)**: Factory Method 패턴을 구현하여 `create_model(model_type, config, random_state)` 호출 시 해당 알고리즘의 원본 모델을 생성하고 공통 인터페이스인 `ABCModelWrapper`로 감싸 반환합니다.
- **`ModelPool` (Class, `model_pool.py`)**: 초기화된 활성 모델 래퍼 인스턴스들을 보관하는 순수 인벤토리 컨테이너입니다.
- **`ABCModelWrapper` 및 Concrete Wrappers (`wrappers.py`)**:
  - `fit(X, y)`와 `predict(X)`의 표준 인터페이스를 제공하는 어댑터(Adapter) 클래스입니다.
  - **`ModelWrapperXGBoost`**: XGBoost 회귀 어댑터.
  - **`ModelWrapperCatBoost`**: CatBoost 회귀 어댑터.
  - **`ModelWrapperRandomForest`**: Scikit-Learn RandomForest 어댑터.
  - **`ModelWrapperMLP`**: Scikit-Learn Multi-layer Perceptron 어댑터.
  - **`ModelWrapperTabPFN`**: 사전학습 정형 트랜스포머 TabPFN 어댑터.
  - **`ModelWrapperTabICL`**: 정형 데이터 In-Context Learning 파운데이션 모델 TabICL 어댑터.
  - **`ModelWrapperTransformer`**: PyTorch 커스텀 신경망(`TransformerBasedRegression`) 어댑터 (스칼라 회귀 및 Gaussian NLL 분포 모드 지원).
- **`OptunaHPOTuner` (Class, `hpo.py`)**:
  - `configs/default.yml`의 `hpo.enabled: true`일 때 기동되어 모델별 하이퍼파라미터 탐색 공간(Search Space)을 정의하고 TPE 베이지안 최적화로 Validation RMSE를 최소화하는 최적 설정을 도출합니다.
- **`StandardBenchmarkExecutor` (Class, `model_executor.py`)**:
  - `ModelPool`을 주입받아 HPO 기동, 일괄 학습(`fit_all`), 일괄 평가(`evaluate_all`), 예측값 수집(`get_predictions`)을 안전한 예외 감내 쉴드 하에서 대행합니다.

---

### D. 프리미엄 시각화 및 리포팅: `automl_framework/util/`
- **`Visualizer` (Class, `visualizer.py`)**:
  - 다크 테마 규격(캔버스 `#0d1117`, 도표 `#161b22`, 그리드 `#30363d`)을 적용한 차트 렌더링.
  - `plot_actual_vs_predicted`: 실제값 vs 예측값 산포도 및 $y=x$ 일치선.
  - `plot_residuals`: 예측값 대비 오차 잔차 분포 산포도.
  - `plot_model_comparison`: 모델별 $R^2$ 및 RMSE 성능 비교 수평 막대 차트.
  - `save_json_report`: 실행 회차 메타데이터와 지표, 챔피언 모델 정보를 구조화된 JSON으로 보관.

---

### E. 대화형 웹 인터페이스 스튜디오: `app.py` (Streamlit WebUI)
- **4대 독립 뷰 사이드바 내비게이션**:
  1. `📊 Overview & Dashboard`: 전체 시스템 개요, 기능 소개, 지원 모델 및 아키텍처 다이어그램 표시.
  2. `🚀 Run Experiment`: 데이터셋 지정, 타겟/피처 동적 바인딩, HPO 옵션 설정, 실시간 터미널 로그 스트리밍 및 실행.
  3. `⚙️ Config Studio`: `default.yml` 스키마 기반의 동적 위젯 렌더링을 통한 하이퍼파라미터 및 프레임워크 설정 튜닝.
  4. `📁 History & Artifacts`: 과거 회차별 실행 결과, JSON 성적표, 프리미엄 차트 갤러리 탐색.
- **견고한 세션 상태 및 미디어 검증**:
  - `st.session_state`를 통한 화면 전환 시 데이터 보존.
  - `is_valid_image()`를 통한 0바이트/손상 이미지 렌더링 방어.

---

## 4. Design Patterns Applied (적용된 디자인 패턴)

| 디자인 패턴 | 적용 위치 | 설계 목적 및 이점 |
| :--- | :--- | :--- |
| **Facade Pattern** | `DataLoaderHelper`, `AutoMLPipeline` | 복잡한 서브시스템(로더, 전처리기, 분할기, 훈련기 등)의 인터페이스를 단순화하여 단일 진입점 제공 |
| **Factory Method** | `ModelFactory`, `ModelType` | 모델 객체 생성 책임을 캡슐화하여 `ModelPool`과의 결합도를 낮추고 신규 모델 추가 용이성 확보 |
| **Adapter Pattern** | `ABCModelWrapper` 및 하위 래퍼들 | Scikit-learn, XGBoost, CatBoost, TabPFN, TabICL, PyTorch 모델들의 상이한 API를 `fit/predict`로 통일 |
| **Strategy Pattern** | `ABCDataLoader`, `ABCDataSplitter`, `ABCModelExecutor` | 알고리즘군과 실행 루프를 런타임에 유연하게 교체할 수 있도록 추상화 |
| **Shield / Fallback** | `ModelFactory`, `LocalFileDataLoader`, `AutoMLPipeline` | 외부 라이브러리 미설치, 런타임 누락, 설정 파일 유실 등 환경 결함 시에도 전체 시스템 크래시 방어 |

---

## 5. Technology Stack & Key Dependencies

- **Language**: Python 3.9+
- **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`, `tabpfn`, `tabicl`, `torch`, `optuna`
- **Data Engineering**: `pandas`, `numpy`, `pyyaml`
- **Visualization & UI**: `matplotlib`, `seaborn`, `streamlit`
- **Testing**: `pytest`
