# [AutoML Regression Framework] 역요구사항 명세서 (Reverse Requirements Specification)

본 문서는 현재 구현되어 있는 **AutoML Regression Framework**의 전체 소스 코드와 아키텍처를 분석하여 시스템의 설계 의도, 핵심 기능, 세부 요구사항, 그리고 품질 속성을 체계적으로 정의한 **소프트웨어 요구사항 명세서 (SRS)**입니다.

---

## 1. 개요 (Overview)

### 1.1 시스템 정의
본 시스템은 정형 데이터(Tabular Data)를 입력받아 다양한 기계학습/딥러닝 회귀(Regression) 알고리즘(트리 기반, 신경망, 트랜스포머, 사전학습 In-Context Learning 모델 등)을 활용해 성능을 일괄 학습, 자동 하이퍼파라미터 튜닝(HPO), 평가, SHAP 모델 해석 및 시각화 리포트를 생성하는 **AutoML Regression Framework**입니다. 

### 1.2 시스템 목표
- **엔드투엔드 솔루션 제공**: 데이터 수집/수신, 결측치 보정 및 인코딩 전처리, 모델 팩토리 인스턴스화, Optuna HPO 자동 튜닝, 일괄 벤치마크 학습 및 평가, SHAP 피처 기여도 분석, 프리미엄 다크 테마 분석 플롯 생성 및 구조화 리포트 저장을 아우르는 단일 파이프라인을 구축합니다.
- **개발 부담 최소화 및 무코드 설정**: 기계학습 모델 훈련에 소요되는 데이터 가공 및 알고리즘 탐색 노력을 최소화하며, 외부 설정 파일(`configs/*.yml`) 및 Streamlit Web UI 연동을 통해 코딩 없이 모델의 세부 하이퍼파라미터 및 구동 방식을 제어합니다.
- **높은 신뢰성과 결함 감내**: 실행 환경의 차이(특정 라이브러리 누락, C-library / OpenMP 런타임 부재 등)에도 파이프라인이 즉시 크래시되지 않고, 최선의 유효 모델 세트를 활용하여 성공적인 완료를 보장합니다.
- **세련된 분석 결과 제공**: 의사결정권자 또는 연구원에게 분석 결과를 명확하게 전달할 수 있는 프리미엄 다크 테마 시각화 차트와 기계 가독성이 뛰어난 표준 JSON 리포트를 자동 발행합니다.
- **단일 책임 및 디자인 패턴 기반 확장성**: 모델 팩토리(`ModelFactory`), 모델 저장소(`ModelPool`), 학습/평가 실행기(`BenchmarkExecutor`), HPO 튜너(`OptunaHPOTuner`), SHAP 분석기(`SHAPAnalyzer`), 데이터 퍼사드(`DataLoaderHelper`)를 철저히 모듈화하여 새로운 알고리즘과 분석 전략을 손쉽게 추가할 수 있도록 설계합니다.

---

## 2. 시스템 요구사항 (System Requirements)

### 2.1 개발 및 실행 환경
- **언어**: Python 3.9+ (Conda `py313` 환경 권장)
- **운영체제**: OS 독립적 (Linux, macOS, Windows 크로스 플랫폼 지원)

### 2.2 기술 스택 및 의존성 (Key Dependencies)
- **데이터 분석 및 프레임 처리**: `pandas` (>= 1.x), `numpy` (>= 1.x)
- **설정 파일 포맷**: `PyYAML` (>= 6.0)
- **머신러닝 & 알고리즘**:
  - `scikit-learn`: 데이터 분할(`train_test_split`), 회귀 평가 지표 계산, `RandomForestRegressor`, `MLPRegressor`
  - `xgboost`: Gradient Boosting 기반 고성능 트리 알고리즘 `XGBRegressor`
  - `catboost`: 범주형 피처 처리에 특화된 `CatBoostRegressor`
  - `tabpfn`: 사전 학습된 정형 데이터 트랜스포머 모델 `TabPFNRegressor`
  - `tabicl`: 정형 데이터 기반 In-Context Learning 딥러닝 회귀 모델 `TabICLRegressor`
  - `torch`: 신경망 및 트랜스포머 기반 딥러닝 회귀 모델 (`TransformerBasedRegression`)
  - `optuna`: 하이퍼파라미터 자동 최적화 프레임워크 (Bayesian / TPE Sampler)
  - `shap`: 모델 해석 및 피처 기여도 분석 라이브러리 (TreeExplainer, KernelExplainer)
- **시각화 및 UI**:
  - `matplotlib`, `seaborn`: 프리미엄 다크 테마 시각화 렌더링
  - `streamlit`: 대화형 웹 인터페이스 스튜디오
- **외부 데이터 소스 연동**:
  - `kaggle`: Kaggle API 연동 및 원격 데이터셋 파일 다운로드 지원
  - `urllib.request`: 내장 다운로더 모듈을 통한 직접 HTTP URL 파싱 지원

---

## 3. 기능적 요구사항 (Functional Requirements)

### 3.1 설정 정보 관리 및 주입 (Configuration Management)
시스템은 외부 설정 파일을 통해 동작 흐름과 파라미터를 중앙 통제하며, 명령줄 인수를 조합하여 유연하게 제어할 수 있어야 합니다.

* **REQ-CF-01: 중앙 YAML 설정 및 동적 파라미터 팽창 (Dynamic Hyperparameters)**
  - **설명**: 프로젝트 내 `configs/` 디렉토리에 정의된 각종 실험 프로파일 설정을 파싱 및 로드하여 분석 대상 컬럼 설정, 학습 테스트 비율, 각 개별 모델의 수많은 파라미터들을 코드 수정 없이 제어해야 합니다.
* **REQ-CF-02: 활성 모델 동적 리스트화 및 보정 (Active Models)**
  - **설명**: YAML 내 `framework.active_models` 목록에 명시된 모델들만 필터링하여 풀(`ModelPool`)에 동적으로 초기화 및 적재하여, 불필요한 모델의 로딩과 메모리 낭비를 제어합니다.
* **REQ-CF-03: CLI 인수 우선순위 보장 및 견고한 예외 안전망 (Precedence & Fallback)**
  - **설명**: 셸 상에서 인가된 실행 인수(예: `--target`, `--test-size`, `--enable-shap`, `--shap-model`)가 있다면, 불러온 YAML 설정 파일의 수치를 우선하여 덮어써서(Override) 적용해야 합니다.
  - **Fallback**: 설정 파일이 유실되었거나 형식이 깨졌을 때에도 크래시를 발생시키지 않고, 내부의 디폴트 딕셔너리 설정을 로드하여 파이프라인 정상 구동을 유지해야 합니다.

### 3.2 데이터 로드 및 수집 (DataLoader - Ingestion & Parsing)
* **REQ-DL-01: Kaggle 데이터 자동 연동 및 다운로드**
* **REQ-DL-02: HTTP URL 데이터 자동 다운로드**
* **REQ-DL-03: 다양한 데이터 포맷 파싱 및 대상 열 분리**
  - CSV, TSV, Parquet, JSONL 포맷 지원 및 JSONL 동적 스키마(Dynamic Schema) 자동 병합 지원.
  - `ignored_columns` 및 `feature_columns` 사전 필터링 지원.

### 3.3 데이터 전처리 및 분할 (DataLoader - Preprocessing & Splitting)
* **REQ-DL-04: 자동 결측치 보정 및 범주형 데이터 변환 (Preprocessing)**
  - 수치형 결측치 Median, 범주형 결측치 Mode 임퓨팅 및 One-Hot Dummy(`drop_first=True`) 인코딩.
* **REQ-DL-05: 난수 고정을 포함한 데이터셋 분할 (Data Splitting)**
  - `TrainTestSplitter`, `KFoldSplitter`, `TimeSeriesSplitter` 지원.

### 3.4 통일화된 모델 제어 및 모델 저장소 구성 (ModelPool & ModelFactory)
* **REQ-MP-01: 어댑터 패턴 기반 인터페이스 규격화 (ModelWrapper)**
  - `fit(X, y)` 및 `predict(X)` 공통 인터페이스 어댑팅.
* **REQ-MP-02: 순수 모델 데이터 저장소 구성 (ModelPool)**
* **REQ-MP-03: 환경 결함에 따른 패키지 부재 시의 생존력 보장 (Robust Shielding)**
* **REQ-MP-04: 팩토리 메서드 패턴 기반 모델 생성 (ModelFactory & ModelType)**
* **REQ-MP-05: TabICL 기반 정형 데이터 In-Context Learning 지원**
  - 사전 학습된 거대 정형 파운데이션 모델 `TabICLRegressor`를 연동하여 그래디언트 업데이트 없는 원샷/퓨샷 정형 추론 수행.

### 3.5 실행 전략 분리, HPO 및 SHAP 분석 (BenchmarkExecutor, HPO & SHAP)
* **REQ-EX-01: 실행기 추상화와 전략 주입 (Strategy Pattern)**
* **REQ-EX-02: 예외 감내형 일괄 학습 및 스코어링 (Executor Fit & Predict)**
* **REQ-EX-03: Optuna 기반 자동 하이퍼파라미터 최적화 (Auto HPO)**
  - `hpo.enabled: true` 시 Optuna TPE 베이지안 최적화를 통해 Validation RMSE를 최소화하는 파라미터 자동 탐색 및 풀 갱신.
* **REQ-EX-04: SHAP 기반 모델 해석 및 피처 기여도 분석 (SHAP Interpretability)**
  - 설정 파일의 `shap.enabled: true` 또는 CLI `--enable-shap` 시 `SHAPAnalyzer`를 호출하여 지정된 모델(`shap.model` 또는 `Champion`)에 대한 SHAP 값을 계산하고 피처별 영향도를 도출해야 합니다.
* **REQ-EX-05: TabICL 전용 In-Context Explainer 파이프라인 (TabICL Dedicated Explainer)**
  - TabICL 모델 분석 시에는 인컨텍스트 예측 함수와 훈련 데이터셋 배경 샘플링을 최적으로 결합한 **TabICL Dedicated In-Context Explainer** 파이프라인을 자동 적용하여, 파운데이션 모델의 피처 기여도를 신속하고 정확하게 산출해야 합니다.

### 3.6 평가 지표 산출 및 시각화 (Evaluation & Visualizer)
* **REQ-EV-01: 회귀 모델 전용 정량 메트릭 일괄 계산** (RMSE, MAE, R²)
* **REQ-VI-01: 프리미엄 다크 테마 디자인 시스템 (Aesthetic Standards)**
* **REQ-VI-02: 실제값 vs 예측값 대칭 분석 차트 (Actual vs Predicted Scatter Plot)**
* **REQ-VI-03: 잔차 분포 산포도 (Residual Scatter Plot)**
* **REQ-VI-04: 일괄 성능 바 차트 (Model Comparison Horizontal Bar Chart)**
* **REQ-VI-05: 이력 관리용 회차별 구조화 JSON 리포트 생성 (Structured Metadata Reporting)**
* **REQ-VI-06: SHAP 피처 중요도 시각화 및 독립 JSON 리포트 발행 (SHAP Reporting)**
  - SHAP Feature Importance 수평 막대 차트(`turn_{turn}_{model}_shap_bar.png`), Summary Beeswarm 플롯(`turn_{turn}_{model}_shap_summary.png`), 그리고 피처별 기여도 순위와 사용된 Explainer 엔진명이 명시된 독립 JSON 리포트(`turn_{turn}_{model}_shap_report.json`)를 자동 생성해야 합니다.

### 3.7 대화형 웹 인터페이스 (Interactive Web UI Studio)
* **REQ-UI-01: 동적 설정 구성 및 스키마 기반 렌더링**
* **REQ-UI-02: 데이터셋 기반 컬럼 동적 바인딩**
* **REQ-UI-03: 실시간 로그 스트리밍 콘솔**
* **REQ-UI-04: 성적표 및 시각화 결과 대시보드**
* **REQ-UI-05: HPO 튜닝 활성화 및 최적화 시도 횟수 지정 UI 지원**
* **REQ-UI-06: 6대 핵심 메뉴 사이드바 내비게이션 (Sidebar Multi-Menu Navigation)**
  - Dataset & Splitting, Models & Active Pool, SHAP Interpretability, Custom Configurations, Runner Console, Results & Metrics 메뉴 구조.
* **REQ-UI-07: 미디어 파일 유효성 검증 및 안전 로딩 (Media Validation Shield)**
* **REQ-UI-08: WebUI SHAP 모델 선택 및 Explainer 엔진 식별 대시보드 (Interactive SHAP UI)**
  - 사이드바 내비게이션 메뉴(`🔍 SHAP Interpretability`)에서 사용자가 SHAP 대상 모델(예: `Champion`, `TabICL`, `XGBoost` 등)을 선택할 수 있으며, 선택 시 사용될 Explainer 엔진(예: `⚡ Engine: TabICL Dedicated In-Context Explainer`, `🌲 Engine: TreeExplainer`)이 실시간 뱃지로 표시되고, 결과 화면에서 피처 중요도 차트와 JSON 리포트를 인터랙티브하게 조회/다운로드할 수 있어야 합니다.

---

## 4. 품질 및 비기능적 요구사항 (Non-Functional & QA Requirements)

### 4.1 사용성 및 접근성 (Usability & Config Driven Control)
- CLI (`python main.py --enable-shap --shap-model TabICL`), Shell 스크립트, WebUI Studio 모두에서 원클릭으로 구동 가능해야 합니다.

### 4.2 도메인 격리형 패키징 및 아키텍처 원칙 (Domain Segregation)
- `SHAPAnalyzer`는 `automl_framework/util/shap_analyzer.py`에 격리되어 기존 모델 및 데이터 로더 인터페이스에 불필요한 결합을 발생시키지 않습니다.

### 4.3 테스트 가능성 및 품질 보증 (QA & Verification)
* **REQ-QA-01: 단위 테스트 커버리지 (Unit Testing)**
* **REQ-QA-02: 종단간 파이프라인 통합 테스트 (E2E Integration Testing)**
* **REQ-QA-03: WebUI 헬퍼 기능 테스트 (UI Helper Testing)**
* **REQ-QA-04: SHAP 해석 및 Explainer 무결성 검증 (SHAP QA Testing)**
  - `tests/test_shap.py`를 통해 TabICL Dedicated Explainer, TreeExplainer, ModelExplainer의 피처 중요도 산출 및 산출물 파일 무결성을 100% 검증해야 합니다.

---

## 5. 데이터 흐름 및 실행 아키텍처 (Data Flow Diagram)

```text
               ┌───────────────────────────────┐
               │       configs/*.yml           │ (다양한 설정 프로파일 보관 디렉토리)
               └───────────────┬───────────────┘
                               │ 설정 로드 및 인수 오버라이드
                               ▼
[1. CLI & AutoMLPipeline] ──> [2. Ingestion/Prep/Split] ──> [3. HPO & Fit & Evaluation] ──> [4. Premium Visuals & Reports]
     (main.py)              (pipeline.prepare_data())     (pipeline.train_and_evaluate())   (pipeline.generate_reports())
                                                                       │                                   │
                                                                       ▼                                   ▼
                                                              [ModelPool Inventory]             [5. SHAPAnalyzer (Optional)]
                                                          (XGBoost, CatBoost, MLP, RF,           (TabICL Dedicated / Tree /
                                                           TabPFN, TabICL, Transformer)           KernelExplainer & Report)
```
