# 🚀 Regression Model Revolution Framework

정형 데이터(Tabular Data)에 특화된 고성능 **AutoML 회귀(Regression) 파이프라인**인 **Regression Model Revolution Framework**에 오신 것을 환영합니다. 이 프레임워크는 최신 정형 데이터 알고리즘(**TabPFN**, **XGBoost** 등)과 자동화된 다크 슬레이트 테마의 시각화 분석 도구를 결합하여, 다양한 회귀 모델의 벤치마킹을 빠르고 아름답고 손쉽게 수행할 수 있도록 설계되었습니다.

---

## 🎨 시각적 프리뷰 & 디자인 철학
본 프레임워크는 **프리미엄 다크 슬레이트(Premium Dark Slate)** 미학을 적용하여 설계되었습니다. 생성되는 모든 플롯은 정밀하게 구성된 고대비 색상 팔레트, 세련된 마이크로 애니메이션 컨셉, 전문적인 레이아웃을 사용하여 과학 논문, 비즈니스 프레젠테이션, 또는 고급 대시보드에 즉시 활용할 수 있는 품질을 제공합니다.

---

## 📂 프로젝트 구조

제어 계층(Orchestration Layer)과 프레임워크 내부 패키지 도메인이 깔끔하고 모듈화된 형태로 분리되어 있습니다:

```
regression-model-revolution-framework/
│
├── main.py                         # 🌟 CLI 파이프라인 총괄 오케스트레이터 (Facade 임포트 제공)
├── app.py                          # 🖥️ Streamlit 기반 WebUI 대시보드 애플리케이션
├── configs/                        # ⚙️ 설정 프로파일 디렉토리 (다양한 설정 실험용)
│   ├── default.yml                 # 기본 AutoML 구성 프로파일
│   ├── kfold_split.yml             # K-Fold 분할 실험 설정 프로파일
│   ├── timeseries_split.yml        # 시계열(Time-Series) 분할 실험 설정 프로파일
│   ├── custom_features.yml         # 커스텀 피처(Feature) 컬럼 선택 실험 프로파일
│   └── web_config.yml              # WebUI에서 자동 생성되는 설정 프로파일
├── scripts/                        # 🏃 시나리오별 자동화 실행 Bash 스크립트
│   ├── run_local_csv.sh            # 로컬 CSV 데이터셋으로 AutoML 벤치마크 파이프라인 실행
│   ├── run_local_jsonl.sh          # 로컬 JSONL 데이터셋(동적 컬럼 지원)으로 실행
│   ├── run_url.sh                  # 원격 URL에서 데이터셋을 다운로드하여 실행
│   └── run_webui.sh                # Streamlit WebUI 스튜디오 실행
├── ARCHITECTURE.md                 # 시스템 모듈 및 실행 흐름 상세 명세서
├── REQ_SPEC.md                     # 요구사항 명세서 및 기능 정의서
├── requirements.txt                # 프레임워크 실행에 필요한 핵심 라이브러리 목록
│
├── automl_framework/               # 📦 AutoML 엔진 코어 패키지
│   ├── __init__.py                 # 주요 API를 노출하는 파사드(Facade) 레이어
│   ├── README.md                   # 서브패키지 기술 문서
│   │
│   ├── dataloader/                 # 📥 데이터 수집 및 처리 도메인 (Facade & 전략 패턴)
│   │   ├── __init__.py
│   │   ├── loaders.py              # 모듈형 로더 (CSV, TSV, Parquet, JSONL 지원)
│   │   ├── preprocessors.py        # 결측치 보정(Imputation), 인코딩 및 ABCDataPreprocessor
│   │   ├── splitters.py            # 데이터셋 분할 전략 및 ABCDataSplitter
│   │   └── data_loader_helper.py   # DataLoaderHelper Facade 및 파이프라인 조율자
│   │
│   ├── model/                      # 🤖 머신러닝/딥러닝 학습 도메인
│   │   ├── __init__.py
│   │   ├── model_pool.py           # 가용 모델 인벤토리 저장소 (ModelPool)
│   │   ├── model_factory.py        # ModelFactory 및 ModelType Enum (Factory Method 패턴)
│   │   ├── model_executor.py       # 벤치마크 실행기 (StandardBenchmarkExecutor)
│   │   ├── wrappers.py             # 예외 복구(Exception Shielded) 기능이 포함된 모델 어댑터 (XGBoost, MLP, TabPFN, RF, CatBoost, Transformer)
│   │   └── architecture/           # 딥러닝 모델 아키텍처 정의
│   │       └── transformer_encoder.py # PyTorch 기반 트랜스포머 시퀀스 회귀 모델
│   │
│   └── util/                       # 🛠️ 시각화 및 유틸리티 도메인
│       ├── __init__.py
│       └── visualizer.py           # 프리미엄 다크 테마 시각화 도구 및 JSON 리포트 작성기
│
└── tests/                          # 🧪 종합 테스트 스위트
    ├── __init__.py
    ├── test_dataloader.py          # 데이터 로딩, 전처리, JSONL 포맷 테스트
    ├── test_model.py               # 모델 초기화 및 실행기 테스트
    └── test_visualizer.py          # 시각화 플롯 및 JSON 리포트 작성 테스트
```

---

## 🛠️ 빠른 시작 (Quick Start)

### 1. 패키지 설치
제공되는 `requirements.txt` 파일을 사용하여 필수 라이브러리를 설치합니다:
```bash
pip install -r requirements.txt
```

*(선택사항) TabPFN 및 Kaggle API 연동을 사용하려면 아래 패키지도 설치해 주세요:*
```bash
pip install tabpfn kaggle
```

### 2. Streamlit WebUI로 실행 (추천)
실험 설정을 시각적으로 구성하고, 실시간 실행 로그를 모니터링하며, 분석 결과를 직관적으로 평가할 수 있는 대화형 WebUI 스튜디오를 실행합니다:
```bash
./scripts/run_webui.sh
```
이 명령어를 실행하면 Streamlit 로컬 서버(기본값: `http://localhost:8501`)가 실행되며, 브라우저를 통해 동적 설정 구성, 학습 수행, 시각화 결과 확인이 가능합니다.

### 3. 명령줄 인터페이스(CLI)로 실행
`scripts/` 디렉토리에 정의된 실행 스크립트를 사용하여 사전 구성된 설정으로 손쉽게 파이프라인을 구동할 수 있습니다:

```bash
# 로컬 CSV 데이터셋으로 실행
./scripts/run_local_csv.sh

# 로컬 JSONL 데이터셋으로 실행 (동적 컬럼 매핑 지원!)
./scripts/run_local_jsonl.sh

# 원격 URL에서 데이터셋을 다운로드하여 실행
./scripts/run_url.sh
```

또는 다음과 같이 CLI 인자들을 조합해 `main.py`를 직접 호출할 수도 있습니다:

로컬 CSV 데이터셋 지정 및 특정 타겟 설정:
```bash
python main.py --dataset-path path/to/dataset.csv --target name_of_target_column
```

일부 행에 특정 컬럼(Key)이 누락되어 있는 JSON Lines(`.jsonl`) 데이터셋 파싱 및 학습:
```bash
python main.py --dataset-path data/synthetic_regression.jsonl --target Target_Y
```

Kaggle에서 데이터셋을 자동으로 내려받아 벤치마크 학습 구동:
```bash
python main.py --kaggle-dataset "sobhanmoosavi/us-accidents" --target "Severity"
```

웹 URL 주소(예: UCI ML 데이터베이스 혹은 GitHub Raw 주소)에서 데이터를 다운로드하여 실행:
```bash
python main.py --url "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv" --target "medv"
```

---

## ⚙️ 설정 프로파일 디렉토리 (`configs/`)

본 프레임워크는 `configs/` 디렉토리 하위의 YAML 파일을 통해 완전히 동적이고 프로파일 기반으로 제어됩니다. 파이썬 소스 코드를 단 한 줄도 수정하지 않고도 활성 모델 목록, 커스텀 피처 하위 집합, 데이터 분할 전략(`train_test_split`, `kfold`, `timeseries`), 그리고 개별 모델의 하이퍼파라미터를 구성할 수 있습니다.

원하는 특정 설정 프로파일로 실행하려면 `--config` 파라미터를 넘겨주면 됩니다:
```bash
# 기본 설정(train_test_split)으로 실행
python main.py --config configs/default.yml --dataset-path data/synthetic_regression.csv

# K-Fold 교차 검증 분할 전략으로 실행
python main.py --config configs/kfold_split.yml --dataset-path data/synthetic_regression.csv

# 순차적 시계열(Time-Series) 분할 전략으로 실행
python main.py --config configs/timeseries_split.yml --dataset-path data/synthetic_regression.csv

# 지정된 일부 커스텀 피처만 추출하여 실행
python main.py --config configs/custom_features.yml --dataset-path data/synthetic_regression.csv
```

### 설정 파일 예시 (`configs/default.yml`)

```yaml
# 글로벌 프레임워크 설정
framework:
  random_state: 42
  active_models:
    - XGBoost
    - MLP
    - TabPFN
    - RandomForest
    - CatBoost

# Hyperparameter Optimization (Optuna)
hpo:
  enabled: false   # Enable automatic HPO via Optuna before training
  n_trials: 10     # Number of trial runs per model
  metric: "RMSE"   # Target metric for optimization (Options: "RMSE", "MAE", "R2")

# 데이터 파이프라인 설정 (타겟/피처 및 분할 전략)
data:
  data_dir: "data"
  output_dir: "outputs"
  target_column: "Target_Y"
  feature_columns: null   # 예: ["Feature_Num", "Feature_Cat"]
  ignored_columns: null   # 예: ["Unwanted_Col1", "Unwanted_Col2"]
  split:
    method: "train_test_split"  # 옵션: "train_test_split", "kfold", "timeseries"
    test_size: 0.2
    n_splits: 5
    shuffle: true

# 모델별 기본 하이퍼파라미터
models:
  XGBoost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
  MLP:
    hidden_layer_sizes: [128, 64]
    activation: "relu"
    max_iter: 500
```

### 실행 인자 우선순위 & Fallback 메커니즘
- CLI 명령줄에 직접 입력한 인수(예: `--test-size 0.3` 또는 `--target target_col`)는 YAML 설정 파일 내의 동일한 설정 값을 자동으로 **오버라이드(Override)**하여 반영합니다.
- 예외 대비 설계: 설정 파일이 누락되거나 일부 깨졌을 경우에도 프로그램이 중단되지 않고, 사전에 정의된 안전한 기본(default) 딕셔너리 설정을 불러와 파이프라인 구동을 안정적으로 유지합니다.

---

## 🎯 Optuna를 활용한 하이퍼파라미터 최적화 (HPO)

본 프레임워크는 **Optuna**를 활용한 자동화된 하이퍼파라미터 튜닝 기능을 갖추고 있습니다. 이 기능이 활성화되면, AutoML 파이프라인은 모델 학습을 기동하기 전에 내부 검증 분할 세트에서 오차(RMSE)를 최소화하도록 하이퍼파라미터 탐색을 선행 수행합니다.

### HPO 작동 방식:
1. **검증 데이터 분할 (Validation Split)**: 최종 평가를 진행할 테스트 데이터셋이 하이퍼파라미터 튜닝 과정에 노출(Data Leakage)되지 않도록, 학습 데이터셋 내부에서 검증을 위한 80/20 비율 분할을 1회 추가로 실행합니다.
2. **대상 모델 (Model Coverage)**: 활성 모델 중 HPO 튜닝에 부합하는 알고리즘인 `XGBoost`, `CatBoost`, `RandomForest`, `MLP`, 그리고 PyTorch 기반 `Transformer` 회귀 신경망이 튜닝 프로세스에 진입합니다.
3. **스마트 스킵 로직 (Smart Skip Logic)**: 사전 학습 구조로 튜닝이 불필요한 `TabPFN`이나 사용자가 임의 등록한 커스텀 모델 등은 탐색 단계에서 안전하게 스킵됩니다.
4. **최종 재학습 (Refitting)**: 탐색이 성공적으로 끝나면 가장 우수한 점수를 보인 하이퍼파라미터를 `ModelPool` 설정에 업데이트하고, 해당 최적 설정을 사용하여 전체 훈련 분할 데이터를 대상으로 최종 피팅(fit)을 완성합니다.

### 1. YAML을 통한 HPO 설정
CLI 파이프라인 실행 시 HPO를 구동하려면 설정 프로필(예: `configs/default.yml`) 내부의 `hpo` 블록을 수정해 줍니다:

```yaml
# 하이퍼파라미터 최적화 설정
hpo:
  enabled: true   # 최종 학습 전에 Optuna 최적화를 구동하려면 true로 변경
  n_trials: 15    # 모델당 튜닝 시도 횟수 (탐색 횟수가 높을수록 최적의 결과를 얻지만 오래 걸림)
  metric: "RMSE"  # 최적화 타겟 메트릭 (선택 가능 옵션: "RMSE", "MAE", "R2")
```

### 2. Streamlit WebUI Studio를 통한 HPO 설정
인터랙티브 웹 대시보드를 사용하는 경우, 왼쪽 사이드바에서 직관적으로 HPO를 활성화할 수 있습니다:
- **Enable HPO (Optuna)**: 간단한 체크박스 토글로 활성화 여부 조절.
- **HPO Trials per Model**: 슬라이더 및 넘버 위젯을 통해 탐색 횟수를 2회에서 100회 사이로 조정 가능.
- **HPO Optimization Metric**: 최적화의 타겟 평가지표 드롭다운 선택 (RMSE, MAE, R2 중 선택 가능).

---

## 📊 출력 결과 & 보고서

매 실행이 완료되면 프레임워크는 `outputs/` 디렉토리에 전문가 수준의 분석 에셋을 저장합니다:
- `outputs/turn_1_model_comparison_r2.png` - 모델별 R2 Score를 나란히 비교 분석하는 수평 바 차트.
- `outputs/turn_1_model_comparison_rmse.png` - 모델별 오차(RMSE) 지표 비교 수평 바 차트.
- `outputs/turn_1_[Model]_actual_vs_pred.png` - 실제값과 예측치 간 편차와 y=x 가이드 피팅 라인을 표시하는 산포도.
- `outputs/turn_1_[Model]_residuals.png` - 오차의 등분산성(Heteroscedasticity)을 진단하기 위한 잔차 분석 플롯.
- `outputs/turn_1_report.json` - 최고 성적의 알고리즘(Champion) 및 전 모델 평가지표 세부정보를 수록한 JSON 결과 리포트.