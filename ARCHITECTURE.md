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
            │  - Multi-Menu (Data, Models, SHAP, Custom, Run)│
            │  - Live Explainer Engine Badge Indicators      │
            │  - Real-time Subprocess Console Log Streamer   │
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
│ - preprocess_data()          │    │    │ - save_html_report()         │
│ - split_data()               │    │    │ - save_markdown_report()     │
│ - prepare_data()             │    │    └──────────────────────────────┘
│  (Delegates to modular       │    │
│   strategies under-the-hood) │    │    ┌──────────────────────────────┐
└──────────────────────────────┘    │    │         SHAPAnalyzer         │
                                    │    │  (automl_framework/          │
                                    │    │   util/shap_analyzer.py)     │
                                    │    ├──────────────────────────────┤
                                    │    │ - TabICL In-Context Explainer│
                                    │    │ - TreeExplainer (Tree Models)│
                                    │    │ - Kernel/ModelExplainer (NN) │
                                    │    │ - plot_shap_bar / summary    │
                                    │    │ - save_shap_json_report()    │
                                    │    └──────────────────────────────┘
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

```text
regression-model-revolution-framework/
│
├── main.py                         # 프로젝트 전체 실행 진입점 (CLI Orchestrator & AutoMLPipeline)
├── app.py                          # Streamlit 기반 대화형 웹 인터페이스 스튜디오 (WebUI Studio)
│
├── configs/                        # 📂 설정 프로파일 보관소 (실험 목적별 YAML 설정 파일)
│   ├── default.yml                 # 기본 통합 설정 프로파일 (모델별 파라미터, HPO, SHAP 설정)
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
│   │   ├── shap_analyzer.py        # SHAP 모델 해석 및 TabICL 전용 In-Context Explainer 엔진
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
│   ├── test_shap.py                # SHAP 모델 해석 및 TabICL 전용 Explainer 단위/통합 테스트
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
- **`AutoMLPipeline` (Class)**: 데이터 수집, 전처리, 모델 초기화, HPO 튜닝, 일괄 학습, 성능 평가, 프리미엄 시각화, SHAP 피처 해석 및 리포트 파일 아카이빙까지의 전체 생명주기를 조율하는 마스터 오케스트레이터입니다.

### B. 프리미엄 시각화 및 SHAP 해석 모듈: `automl_framework/util/`
- **`Visualizer` (Class, `visualizer.py`)**: 다크 테마 플롯, 반응형 HTML 리포트 및 Markdown 요약본 생성.
- **`SHAPAnalyzer` (Class, `shap_analyzer.py`)**: `TreeExplainer`, `TabICL Dedicated In-Context Explainer`, `ModelExplainer` 다형성 엔진 지원.

### C. 대화형 웹 인터페이스 스튜디오: `app.py` (Streamlit WebUI)
- 사이드바 내비게이션 기반 6대 핵심 뷰 및 실시간 Explainer 엔진 뱃지 표시.
