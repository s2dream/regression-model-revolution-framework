# AutoML Regression Framework Architecture

본 문서는 **AutoML Regression Framework**의 모듈 및 클래스 구조, 데이터 흐름, 그리고 이들 간의 상호작용 관계를 텍스트 기반 다이어그램과 함께 상세히 설명합니다.

---

## 1. High-Level Architecture Diagram (아키텍처 다이어그램)

아래 다이어그램은 프레임워크의 핵심 실행 제어 흐름과 데이터의 파이프라인 처리 과정을 텍스트(ASCII/Unicode Art)로 시각화한 것입니다.

```text
                               ┌──────────────────────────────────────────────┐
                               │             CLI / User Entry Point           │
                               │                  (main.py)                   │
                               └──────┬────────────────────────────────┬──────┘
                                      │                                │
                     ① Load &         │                                │ ④ Metrics & Predictions
                     Preprocess       │ ② Train Splits (X_train, y_train)│    for Premium Reports
                     Data             ▼                                ▼
               ┌──────────────────────────────┐              ┌──────────────────────────────┐
               │         DataLoader           │              │         Visualizer           │
               │     (core/data_loader.py)    │              │     (core/visualizer.py)     │
               ├──────────────────────────────┤              ├──────────────────────────────┤
               │ - download_from_kaggle()     │              │ - plot_actual_vs_predicted() │
               │ - download_from_url()        │              │ - plot_residuals()           │
               │ - load_dataset()             │              │ - plot_model_comparison()    │
               │ - preprocess_data()          │              │ - save_json_report()         │
               │ - split_data()               │              └──────────────────────────────┘
               └──────────────┬───────────────┘
                              │
                              │ ③ Evaluate Splits (X_test, y_test)
                              ▼
               ┌────────────────────────────────────────────────────────────────────────────┐
               │                                 ModelPool                                  │
               │                             (core/models.py)                               │
               ├────────────────────────────────────────────────────────────────────────────┤
               │ - fit_all(X_train, y_train)                                                │
               │ - evaluate_all(X_test, y_test) -> metrics (RMSE, MAE, R2)                  │
               │ - get_predictions(X) -> dict of predictions                                │
               │ - add_custom_model(name, model_instance)                                   │
               └──────────────────────────────────────┬─────────────────────────────────────┘
                                                      │
                                                      │ Manages & Standardizes
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │             ModelWrapper             │
                                   │           (core/models.py)           │
                                   ├──────────────────────────────────────┤
                                   │ - fit(X, y)                          │
                                   │ - predict(X)                         │
                                   └──────────────────┬───────────────────┘
                                                      │
                                      ┌───────────────┼───────────────┐ Instantiates & Adapts
                                      ▼               ▼               ▼
                             ┌─────────────────┐┌───────────────┐┌─────────────────┐
                             │   XGBoost / RF  ││   MLP (NN)    ││     TabPFN      │
                             │ (xgboost/sklearn││(scikit-learn) ││    (tabpfn)     │
                             └─────────────────┘└───────────────┘└─────────────────┘
```

---

## 2. Directory Structure (디렉토리 구조)

프로젝트 루트 디렉토리의 전체 레이아웃과 핵심 소스 파일 위치는 다음과 같습니다.

```text
regression-model-revolution-framework/
│
├── automl_framework/               # 프레임워크 메인 패키지
│   ├── core/                       # 핵심 모듈 서브패키지
│   │   ├── __init__.py             # 패키지 진입점 (DataLoader, ModelPool, Visualizer 노출)
│   │   ├── data_loader.py          # 데이터 수집, 로드, 전처리 및 데이터 분할
│   │   ├── models.py               # 모델 관리 풀 및 개별 모델 어댑터 (Wrapper)
│   │   └── visualizer.py           # 프리미엄 차트 생성 및 JSON 실행 보고서 작성
│   │
│   ├── main.py                     # 프레임워크 CLI 엔트리포인트 (파이프라인 통합 실행)
│   └── test.py                     # 간단한 환경 확인용 테스트 스크립트
│
├── data/                           # (자동 생성) 다운로드되거나 생성된 데이터셋 저장소
├── outputs/                        # (자동 생성) 시각화 이미지(.png) 및 JSON 실행 보고서 저장소
│
├── .gitignore                      # Git 제외 목록 설정 파일
├── LICENSE                         # Apache 라이선스 파일
├── README.md                       # 프로젝트 기본 설명서
└── ARCHITECTURE.md                 # [본 파일] 시스템 아키텍처 명세서
```

---

## 3. Core Modules & Classes (핵심 모듈 및 클래스 구성)

프레임워크는 각 역할에 따라 단일 책임 원칙(Single Responsibility Principle)을 준수하는 모듈들로 설계되었습니다.

### A. CLI 및 전체 파이프라인 제어: `main.py`
* **역할**: 전체 AutoML 프로세스를 조율하는 마스터 컨트롤러입니다.
* **주요 흐름**:
  1. CLI 인자 파싱 (`--dataset-path`, `--target`, `--test-size`, `--turn` 등).
  2. 외부 데이터셋(Kaggle API, UCI URL) 또는 모의 데이터(Synthetic Dataset)의 자동 수집 분기 처리.
  3. `DataLoader`를 사용해 데이터를 전처리하고 Train/Test 분할.
  4. `ModelPool`에 정의된 모델 세트 전체에 대한 일괄 학습(`fit_all`) 및 성능 평가(`evaluate_all`).
  5. `Visualizer`를 활용해 프리미엄 다크 테마 분석 플롯 및 최종 JSON 지표 리포트 생성.

---

### B. 데이터 로더 및 전처리 모듈: `core/data_loader.py`
#### `DataLoader` (Class)
* **책임**: 데이터 획득(Kaggle, HTTP URL)부터 학습 전 단계까지의 모든 데이터 처리를 담당합니다.
* **핵심 메서드**:
  * `download_from_kaggle(dataset_name)`: Kaggle API를 사용하여 원격 데이터셋을 다운로드하고 압축을 해제합니다.
  * `download_from_url(url, filename)`: 외부 웹 서버(예: UCI 머신러닝 리포지토리)에서 직접 데이터셋 파일을 가져옵니다.
  * `load_dataset(filepath, target_column)`: 로컬 CSV, TSV, Parquet 포맷 데이터를 판다스 데이터프레임으로 자동 읽어 들이고 독립 변수(X)와 종속 변수(y)로 분리합니다.
  * `preprocess_data(X)`: 결측치 보정(수치형은 중앙값, 범주형은 최빈값 임퓨테이션) 및 범주형 변수의 원-핫 인코딩(Dummy Encoding)을 자동으로 수행합니다.
  * `split_data(X, y, test_size, val_size)`: 학습, 검증, 테스트 셋으로 데이터를 안정적으로 분할합니다.

---

### C. 모델 관리 및 표준화 모듈: `core/models.py`
#### `ModelWrapper` (Class)
* **책임**: 서로 다른 패키지(Scikit-Learn, XGBoost, TabPFN 등) 모델의 인터페이스를 동일한 규격(`fit` / `predict`)으로 감싸 일관되게 사용할 수 있게 하는 **어댑터(Adapter) 역할**을 합니다.
* **핵심 메서드**:
  * `fit(X, y)`: 데이터 형태를 NumPy Array 등으로 정렬하여 내부 모델 객체를 학습시킵니다.
  * `predict(X)`: 일관된 예측 결과를 반환합니다.

#### `ModelPool` (Class)
* **책임**: 다채로운 알고리즘군(Tree 기반, 신경망 기반, 사전 학습 기반 등)의 모델들을 일괄 초기화하고 보관 및 일괄 처리하는 매니저 클래스입니다.
* **핵심 메서드**:
  * `_initialize_default_models()`: XGBoost, Multi-Layer Perceptron(MLP), TabPFN(소/중형 정형 데이터용 사전 학습 트랜스포머 모델), RandomForest 모델들을 환경 탐색 후 자동 인스턴스화합니다.
  * `add_custom_model(name, model_instance)`: 사용자가 정의한 임의의 회귀 모델을 풀에 동적으로 추가합니다.
  * `fit_all(X_train, y_train)`: 풀 내 모든 유효 모델에 대해 학습을 진행합니다.
  * `evaluate_all(X_val, y_val)`: 검증 데이터에 대한 각 모델의 예측을 수행하고 RMSE, MAE, R² score 성능 평가 지표를 생성합니다.
  * `get_predictions(X)`: 활성 모델 전체의 개별 예측값을 딕셔너리로 반환합니다.

---

### D. 프리미엄 시각화 및 레포팅 모듈: `core/visualizer.py`
#### `Visualizer` (Class)
* **책임**: 데이터 분석 결과 및 모델 성능 지표를 화려하고 세련된 그래픽 플롯(Premium Dark Theme) 및 구조화된 JSON 실행 메타데이터 파일로 보관합니다.
* **핵심 메서드**:
  * `plot_actual_vs_predicted(y_true, y_pred, model_name, turn)`: 실제값과 예측값의 산점도를 1:1 선(Perfect Fit Line)과 함께 시각화하여 예측 오차의 일관성을 직관적으로 관찰할 수 있도록 합니다.
  * `plot_residuals(y_true, y_pred, model_name, turn)`: 잔차 분석 산점도를 출력하여 등분산성(Heteroscedasticity) 유무를 진단할 수 있도록 지원합니다.
  * `plot_model_comparison(metrics, metric_name, turn)`: 전체 모델들의 성능(R², RMSE 등)을 한눈에 볼 수 있는 깔끔한 수평 바 차트(Horizontal Bar Chart)를 생성합니다.
  * `save_json_report(metrics, turn)`: 학습된 모든 모델의 상세 평가 수치 지표와 베스트 모델의 정보를 JSON 파일로 깔끔하게 포매팅하여 저장합니다.

---

## 4. Pipeline Execution & Data Flow (파이프라인 실행 흐름)

AutoML 프레임워크의 실행 흐름은 순차적으로 아래 단계들을 거칩니다:

```text
[1. CLI 실행]  ──> [2. 데이터 수집/로드]  ──> [3. 결측치 보정/인코딩]  ──> [4. 데이터 분할]
   (main.py)       (DataLoader.load)      (DataLoader.preprocess)     (DataLoader.split)
                                                                               │
                                                                               ▼
[8. 분석 결과 확인] <── [7. 프리미엄 차트 생성] <── [6. 성능 메트릭 평가] <── [5. 모델 일괄 학습]
   (outputs/)        (Visualizer)            (ModelPool.evaluate)       (ModelPool.fit)
```

---

## 5. Technology Stack & Key Dependencies (기술 스택)

- **언어**: Python 3.x
- **데이터 분석 및 머신러닝**:
  - `pandas`, `numpy`: 데이터 조작 및 행렬 연산
  - `scikit-learn`: 머신러닝 모델(RF, MLP), 데이터 스플릿, 평가 메트릭 산출
  - `xgboost`: Gradient Boosting 기반 트리 모델
  - `tabpfn`: 정형 데이터 특화 Prior-Data Fitted Network 모델
- **시각화 및 레포팅**:
  - `matplotlib`, `seaborn`: 프리미엄 다크 테마 기반 맞춤 플롯 렌더링
  - `json`: 표준 구조화 실행 리포트 아카이빙
