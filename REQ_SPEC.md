# [AutoML Regression Framework] 역요구사항 명세서 (Reverse Requirements Specification)

본 문서는 현재 구현되어 있는 **AutoML Regression Framework**의 소스 코드와 `ARCHITECTURE.md` 파일을 리버스 엔지니어링하여 시스템의 설계 의도, 핵심 기능, 신규 기능(HPO 메트릭 선택, SHAP, 학습 곡선, Markdown 보고서), 그리고 디자인 패턴 적용 사항을 역으로 도출한 **요구사항 명세서**입니다.

---

## 1. 개요 (Overview)

### 1.1 시스템 정의
본 시스템은 정형 데이터(Tabular Data)를 입력받아 다양한 기계학습 회귀(Regression) 알고리즘을 활용해 성능을 일괄 학습하고 예측하는 **AutoML Regression Framework**입니다. 

### 1.2 시스템 목표
- **종합 솔루션 제공**: 데이터 수집, 전처리, 모델 학습, HPO 튜닝, 설명력 분석, 학습 과정 시각화, 고화질 분석 플롯 생성 및 상세 마크다운/JSON 보고서 저장을 아우르는 단일 파이프라인을 구축합니다.
- **개발 부담 최소화 및 무코드 설정**: 기계학습 모델 훈련에 소요되는 데이터 전처리(결측치 임퓨테이션, 원-핫 인코딩 등) 및 알고리즘 탐색 노력을 최소화하며, 외부 설정 파일(`default.yml` 등) 연동을 통해 코딩 없이 모델의 세부 하이퍼파라미터 및 구동 방식을 제어합니다.
- **높은 신뢰성과 결함 감내**: 실행 환경의 차이(특정 라이브러리 누락, OpenMP 런타임 부재 등)에도 파이프라인이 즉시 크래시되지 않고, 최선의 유효 모델 세트를 활용하여 성공적인 완료를 지향합니다.
- **세련된 분석 결과 제공**: 의사결정권자 또는 연구원에게 분석 결과를 명확하게 인지시킬 수 있는 프리미엄 다크 테마 시각화 자료와 기계 가독성이 뛰어난 구조화된 JSON/Markdown 리포트를 자동 발행합니다.
- **디자인 패턴 기반의 구조적 미학 확보**: 팩토리 메서드, 전략, 퍼사드, 어댑터 등의 핵심 디자인 패턴들을 엄격히 적용하여 결합도를 낮추고 모듈의 재사용성과 가독성을 극대화합니다.

---

## 2. 시스템 요구사항 (System Requirements)

### 2.1 개발 및 실행 환경
- **언어**: Python 3.x
- **운영체제**: OS 독립적 (Linux, macOS, Windows 등 크로스 플랫폼 지원)

### 2.2 기술 스택 및 의존성 (Key Dependencies)
- **데이터 분석 및 프레임 처리**: `pandas` (>= 1.x), `numpy` (>= 1.x)
- **설정 파일 포맷**: `PyYAML` (>= 6.0)을 통한 YAML 구성 파일 로드 및 유효성 판단
- **머신러닝 & 알고리즘**:
  - `scikit-learn`: 데이터 분할, 평가 지표 계산, `RandomForestRegressor`, `MLPRegressor` 제공
  - `xgboost`: 트리 부스팅 기반 고성능 알고리즘 `XGBRegressor` 제공
  - `catboost`: 트리 부스팅 기반의 고성능 카테고리 특화 알고리즘 `CatBoostRegressor` 제공
  - `tabpfn`: 소/중형 정형 데이터 특화 사전 학습 정형 데이터 트랜스포머 모델 `TabPFNRegressor` 제공
- **시각화 및 설명력**:
  - `matplotlib`, `seaborn`: 프리미엄 디자인 규격이 적용된 테마 차트 렌더링
  - `shap` (>= 0.42.0): 피처 기여도 요약(Beeswarm) 및 바 차트 시각화
  - `json`: 표준 구조화 리포트 저장을 위한 내장 객체
- **외부 데이터 소스 연동**:
  - `kaggle`: Kaggle API 연동 및 원격 데이터셋 파일 다운로드 지원
  - `urllib.request`: 내장 다운로더 모듈을 통한 직접 HTTP URL 파싱 지원

---

## 3. 기능적 요구사항 (Functional Requirements)

### 3.1 설정 정보 관리 및 주입 (Configuration Management)
시스템은 외부 설정 파일을 통해 동작 흐름과 파라미터를 중앙 통제하며, 명령줄 인수를 조합하여 유연하게 제어할 수 있어야 합니다.

* **REQ-CF-01: 중앙 YAML 설정 및 동적 파라미터 팽창 (Dynamic Hyperparameters)**
  - **설명**: 프로젝트 내 `configs/` 디렉토리에 정의된 각종 실험 프로파일 설정을 파싱 및 로드하여 분석 대상 컬럼 설정, 학습 테스트 비율, 각 개별 모델의 수많은 파라미터들을 코드 수정 없이 제어해야 합니다.
  - **동적 바인딩**: `ModelPool`은 YAML 설정 구조 내 개별 모델 사전 데이터를 획득한 뒤, python dynamic argument expansion (`**kwargs`) 구조를 취하여 모델 인스턴스 초기화 시 파라미터를 유연하게 일괄 전개해 주입해야 합니다.
* **REQ-CF-02: 활성 모델 동적 리스트화 및 보정 (Active Models)**
  - **설명**: YAML 내 `framework.active_models` 목록에 명시된 모델들만 필터링하여 풀(`ModelPool`)에 동적으로 초기화 및 적재하여, 불필요한 모델의 로딩과 메모리 낭비를 제어합니다.
* **REQ-CF-03: CLI 인수 우선순위 보장 및 견고한 예외 안전망 (Precedence & Fallback)**
  - **설명**: 셸 상에서 인가된 실행 인수(예: `--target`, `--test-size`, `--config`)가 있다면, 불러온 YAML 설정 파일의 수치를 우선하여 덮어써서(Override) 적용해야 합니다. 이 인수 오버라이드 및 정책 조합 처리는 `AutoMLPipeline` 클래스의 생성자(`__init__`) 레벨에서 일괄 캡슐화되어 조율되고 유효성이 검증되어야 합니다.
  - **Fallback**: 설정 파일이 유실되었거나 형식이 깨졌을 때에도 크래시를 뿜으며 종료되지 않고, 내부의 견고한 디폴트 딕셔너리 설정을 로드하여 파이프라인 정상 구동을 유지해야 합니다.
  - **디자인 패턴 연동**: 이 프로세스는 클라이언트와 복잡한 하부 설정 처리 및 전처리 서브시스템을 분리하는 **[퍼사드(Facade) 패턴](https://refactoring.guru/ko/design-patterns/facade)**의 원칙에 따라 설계되었습니다.

### 3.2 데이터 로드 및 수집 (DataLoader - Data Fetching & Loading)
* `DataLoader`는 **[퍼사드(Facade) 패턴](https://refactoring.guru/ko/design-patterns/facade)**을 취해 데이터 로드, 전처리, 분할에 대한 클라이언트 단일 진입점을 제공하며, 내부적으로는 `loaders.py`에 정의된 `ABCDataLoader` 규격 하의 개별 전략 클래스들(`loaders.py`)에 실행을 위임해야 합니다.
* **REQ-DL-01: Kaggle 데이터 자동 연동 및 다운로드**
  - **설명**: Kaggle의 데이터셋 고유 식별자(예: `'sobhanmoosavi/us-accidents'`)를 제공받았을 때, 내장 `KaggleDataLoader` 연동 모듈을 호출하여 데이터를 자동으로 다운로드하고 압축을 풀어 로컬 지정 디렉토리에 배치할 수 있어야 합니다.
* **REQ-DL-02: HTTP URL 데이터 자동 다운로드**
  - **설명**: 외부 다이렉트 다운로드 주소(URL)를 입력받았을 때, `URLDataLoader`를 통해 원본 데이터를 직접 내려받아 로컬에 보관할 수 있어야 합니다.
* **REQ-DL-03: 다양한 데이터 포맷 파싱 및 대상 열 분리**
  - **설명**: `LocalFileDataLoader`가 로컬의 `.csv`, `.tsv`, `.txt` (탭 구분자 포맷), `.parquet`, 그리고 `.jsonl` 확장자를 자동으로 감지하여 DataFrame 형태로 읽어 들일 수 있어야 합니다.
  - **JSONL 동적 스키마 로딩**: 각 행이 독립적인 JSON 객체로 기술되는 JSON Lines(`.jsonl`) 포맷의 경우, 누락된 값에 의해 행마다 존재하지 않는 키가 생길 수 있습니다. 데이터를 라인별로 읽으면서 새로운 키를 발견할 때마다 새롭게 컬럼을 동적으로 결합 및 확장하고, 키가 생략된 위치는 자동으로 `NaN`을 안전 매핑하여 판다스 데이터프레임으로 정렬 정형화해 로드하는 기능을 완수해야 합니다.
  - **제외 컬럼 사전 필터링 (Ignored Columns)**: `ignored_columns` 옵션을 지원하여, 데이터 가공의 가장 첫 단계에서 불필요하거나 제외하고 싶은 열(Column)들을 지정하여 일괄적으로 드롭(drop)한 뒤 후속 분석을 실행할 수 있도록 해야 합니다.
  - **커스텀 피처 지정**: `feature_columns` 옵션을 지원하여, 지정된 특정 피처 컬럼들만 데이터셋에서 안전하게 추출해 $X$ 매트릭스를 구성하고 타겟 $y$를 매핑하는 기능을 완수해야 합니다.

### 3.3 데이터 전처리 및 분할 (DataLoader - Preprocessing & Splitting)
* **REQ-DL-04: 자동 결측치 보정 및 범주형 데이터 변환 (Preprocessing)**
  - **설명**: `StandardDataPreprocessor`를 호출하여 수치형 결측치는 열의 **중앙값(Median)**으로, 범주형 결측치는 열의 **최빈값(Mode)**으로 임퓨팅한 후 **더미 변수화(Dummy Encoding, 원-핫 인코딩)**를 자동으로 실행해야 합니다. 다중공선성 문제를 방지하기 위해 첫 번째 범주를 제외하는 `drop_first=True` 전략을 고수합니다.
* **REQ-DL-05: 난수 고정을 포함한 데이터셋 분할 (Data Splitting)**
  - **설명**: 다양한 스플릿 방식 중 설정 프로파일에 지정된 기법을 선택적으로 로딩 및 동작시켜 데이터를 학습(Train) 및 최종 성능 측정(Test) 셋으로 분할하되, 난수 시드(`random_state`)를 지정하여 실행 재현성을 확보해야 합니다.
  - **디자인 패턴 연동**: Holdout, K-Fold, TimeSeries 등 교체 가능한 스플리터 구현체들은 독립 인터페이스를 지닌 **[전략(Strategy) 패턴](https://refactoring.guru/ko/design-patterns/strategy)**을 준수해 유연한 상호교체가 가능해야 합니다.

### 3.4 통일화된 모델 제어 및 모델 저장소 구성 (ModelPool & ModelWrapper)
* **REQ-MP-01: 어댑터 패턴 기반 인터페이스 규격화 (ModelWrapper)**
  - **설명**: 다양한 외부 프레임워크 기반 모델(Scikit-Learn, XGBoost, CatBoost, TabPFN, PyTorch 등)의 제어 체계를 표준화하기 위해 `fit(X, y)` 및 `predict(X)`라는 공통 규격을 제공하는 **[어댑터(Adapter) 패턴](https://refactoring.guru/ko/design-patterns/adapter)**을 내재화해야 합니다.
* **REQ-MP-02: 순수 모델 데이터 저장소 구성 (ModelPool)**
  - **설명**: `ModelPool`은 동작 제어 성격의 메서드를 모두 배제하고, 오직 설정 기반으로 모델 인스턴스들을 생성하고 보관하는 **순수 데이터 컨테이너(Inventory Repository)**의 책임만 완수해야 합니다.
* **REQ-MP-03: 환경 결함에 따른 패키지 부재 시의 생존력 보장 (Robust Shielding)**
  - **설명**: 사용자 개발 환경에 특정 라이브러리가 미설치되었거나, Mac OSX 환경에서 XGBoost에 필수적인 OpenMP `libomp.dylib` 런타임 파일이 누락되어 링크 로딩 에러가 발생하는 등 어떠한 물리적 결함이 발생하더라도, 프로그램이 즉사하지 않고 경고 로그를 출력한 뒤 가용한 나머지 정상 모델들만으로 풀을 완성시켜 구동의 중단을 차단해야 합니다.
* **REQ-MP-04: 팩토리 메서드 패턴 기반 모델 생성 (ModelFactory & ModelType)**
  - **설명**: 모델 인스턴스의 빌드 및 하이퍼파라미터 주입 로직을 모델 저장소(`ModelPool`)로부터 완전히 떼어내어 `ModelFactory` 클래스로 캡슐화해야 합니다. 또한 모델 상수를 관리하고 대소문자나 특수문자 무관하게 유연히 상수로 변환할 수 있도록 `ModelType` Enum을 설계 및 내장해야 합니다.
  - **디자인 패턴 연동**: 이 구조는 객체 생성 책임을 격리하는 **[팩토리 메서드(Factory Method) 패턴](https://refactoring.guru/ko/design-patterns/factory-method)**을 실현합니다.

### 3.5 실행 전략 분리 및 HPO 튜닝 (BenchmarkExecutor & HPO)
* **REQ-EX-01: 실행기 추상화와 전략 주입 (Strategy Pattern)**
  - **설명**: 일괄 학습, 일괄 성능 지표 산출, 일괄 예측값 수집의 제어 흐름을 추상화한 인터페이스인 `ABCModelExecutor`를 정의하고, 구체적인 일괄 벤치마크 학습 루프 전략인 `StandardBenchmarkExecutor`를 함께 제공합니다.
  - **디자인 패턴 연동**: 실행 엔진과 모델 인벤토리 데이터를 완전 분리하는 **[전략(Strategy) 패턴](https://refactoring.guru/ko/design-patterns/strategy)**을 준수합니다.
* **REQ-EX-02: 예외 감내형 일괄 학습 및 스코어링 (Executor Fit & Predict)**
  - **설명**: 주입받은 `ModelPool` 내 가용한 활성 모델 전체에 대해 학습 루프(`fit_all`)를 돌리고, 테스트 셋 평가(`evaluate_all`) 및 일괄 예측값 수집을 대행해 그 결과를 구조화된 사전 형태로 리포팅해야 합니다.
* **REQ-EX-03: Optuna 기반 가변 평가지표 자동 하이퍼파라미터 튜닝 (Auto HPO)**
  - **설명**: HPO 활성화 상태에서 최적화 목적 지표(**RMSE, MAE, R2 Score**) 중 사용자가 설정한 지표에 부합하도록 탐색 방향("minimize" 또는 "maximize")을 조율하여 Optuna 자동 하이퍼파라미터 탐색을 기동합니다. 튜닝 완료 시 최적 하이퍼파라미터를 모델 풀에 자동 업데이트 및 전체 학습셋에 재학습(Refit)을 수행하며, TabPFN 등 대상에서 제외되는 모델은 스킵합니다.

### 3.6 시각화, 설명력 및 보고서 저장 (Evaluation, SHAP, Loss Curves & Reporting)
* **REQ-EV-01: 회귀 모델 전용 정량 메트릭 일괄 계산**
  - **설명**: 테스트 데이터 셋에 대해 모든 활성 모델별 예측값을 도출한 뒤, 실제값과 매칭하여 **RMSE, MAE, R² Score** 지표를 일괄 계산해야 합니다.
* **REQ-VI-01: 프리미엄 다크 테마 디자인 시스템 (Aesthetic Standards)**
  - **설명**: 차트 캔버스 배경색 차콜 블랙 슬레이트 테마 `#0d1117`, 도표 내부 `#161b22`, 그리드 `#30363d` 및 파스텔 하모니 팔레트(블루, 코랄, 민트 등)를 엄격히 고수하여 렌더링해야 합니다.
* **REQ-VI-02: 실제값 vs 예측값 대칭 분석 차트 (Actual vs Predicted Scatter Plot)**
  - **설명**: 예측치 분포 산포도를 그릴 때 투명 파스텔 블루 점들을 플로팅하고, 완벽한 오차 없는 예측을 상징하는 **대각선 perfect fit line(y=x)**을 코랄 레드의 점선 형태로 명확하게 도식화해 저장해야 합니다.
* **REQ-VI-03: 잔차 분포 산포도 (Residual Scatter Plot)**
  - **설명**: 등분산성 판별을 위해 x축 예측치, y축 잔차 산포도와 함께 오차 0 수평 기준선을 뚜렷하게 그려 넣어야 합니다.
* **REQ-VI-04: 일괄 성능 바 차트 (Model Comparison Horizontal Bar Chart)**
  - **설명**: 모델 성능 순위에 맞게 내림/오름차순 정렬된 깔끔한 수평 막대 차트를 그리고, 끝단 레이블 공간에 지표 값을 명기해 줍니다.
* **REQ-VI-05: 이력 관리용 회차별 구조화 JSON 리포트 생성 (Structured Metadata Reporting)**
  - **설명**: 실행 회차 번호(Turn ID) 및 모델별 성능 평갓값(R², RMSE, MAE), 베스트 모델 명칭, 그리고 SHAP 및 학습 곡선 절대 파일 경로들을 담아 표준 JSON 리포트 파일(`turn_{turn}_report.json`)을 자동으로 보존해야 합니다.
* **REQ-VI-06: 학습 곡선 그래프 생성 (Loss Curve Tracking)**
  - **설명**: 반복 학습을 수행하는 모델(MLP, XGBoost, CatBoost, Transformer)의 에포크/이터레이션 단위 손실(Loss) 수치를 수집하여 학습 곡선 그래프(`turn_{turn}_{[Model]}_learning_curve.png`)를 그리고 저장해야 합니다.
* **REQ-VI-07: SHAP 설명력 시각화 연동 (SHAP Explainability)**
  - **설명**: 설정 사양(우승 모델만 혹은 전 모델 대상, max_samples 제한)에 부합하도록 SHAP 분석 알고리즘(TreeExplainer 또는 agnostic 포백 Explainer)을 기동하여 요약 분포도(Beeswarm Plot) 및 중요도 바 차트(Bar Plot) 이미지를 자동 생성 및 저장합니다.
* **REQ-VI-08: 전문적 실행 보고서 저장 (Executive Markdown Report)**
  - **설명**: 데이터 형상 분석 요약(타겟, 피처 개수, 분할 수), R² 성적 순으로 정렬한 Leaderboard 마크다운 표, 각 이미지 파일의 절대경로 하이퍼링크 및 시스템 종합 권장 분석(Recommendations)이 집약된 마크다운 보고서 파일(`turn_{turn}_report.md`)을 생성 및 보존합니다.

### 3.7 대화형 웹 인터페이스 (Interactive Web UI Studio)
* **REQ-UI-01: 동적 설정 구성 및 스키마 기반 렌더링 (Dynamic Schema Rendering)**
  - **설명**: UI 화면에 모델 목록이나 파라미터를 하드코딩하지 않고, `default.yml` 설정 파일을 동적으로 파싱하여 그에 매핑되는 하이퍼파라미터 입력 위젯(Slider, Number Input 등)을 자동 생성해 주어야 합니다.
* **REQ-UI-02: 데이터셋 기반 컬럼 동적 바인딩 (Dynamic Column Binding)**
  - **설명**: 로컬의 데이터셋을 지정할 시 데이터셋 구조를 판독하여 컬럼 목록을 실시간으로 가져와 Target, Features, Ignored 컬럼 설정을 클릭 한 번으로 선택할 수 있도록 컴포넌트를 설계해야 합니다.
* **REQ-UI-03: 실시간 로그 스트리밍 콘솔 (Subprocess Real-time Log Streaming)**
  - **설명**: 웹 상에서 실험 시작 시 백그라운드 subprocess로 `main.py` 파이프라인을 기동하고, 프로세스의 표준 출력(stdout)을 한 줄씩 가로채어 실시간 터미널 스타일로 시각화해야 합니다.
* **REQ-UI-04: 성적표 및 시각화 결과 대시보드 (Interactive Result Dashboard)**
  - **설명**: 실험 실행 완료 즉시 리포트 JSON 및 출력 차트 파일들을 탐색하여 대시보드에 Champion 모델 요약 정보, 모델별 성능 정렬 테이블, 그리고 Visualizer 플롯(실제치 vs 예측치, 잔차 분석 등)을 렌더링해야 합니다.
* **REQ-UI-05: HPO 튜닝 활성화 및 최적화 시도 횟수 지정 UI 지원**
  - **설명**: 웹 대시보드 사이드바(Sidebar) 영역에서 사용자가 HPO 기능 기동 여부와 탐색 시도 횟수(`n_trials`), 튜닝 대상 목적 지표(RMSE/MAE/R2)를 명시적으로 조작하고, 이를 설정 파일에 전파할 수 있게 구성해야 합니다.
* **REQ-UI-06: SHAP 및 학습 곡선 대시보드 모듈**
  - **설명**: 웹 대시보드 결과 페이지 상단에 **View Detailed Executive Report** 접기/펼치기 위젯을 추가하여 자동 생성된 마크다운 보고서를 수려하게 직접 바인딩하고, 진단 영역에 해당 모델의 학습 손실 곡선(Loss Curve) 및 SHAP Beeswarm/Bar 플롯을 다이내믹하게 렌더링해 줍니다.

---

## 4. 비기능적 요구사항 (Non-Functional Requirements)

### 4.1 사용성 및 접근성 (Usability & Config Driven Control)
- 사용자는 `python main.py` 명령어 뿐만 아니라 `scripts/` 디렉토리에 미리 보관해 둔 실행 스크립트 파일들을 실행하는 것만으로 곧바로 프레임워크 전체 오케스트레이션 프로세스를 즉시 구동 및 재현할 수 있어야 합니다.
- 하이퍼파라미터 튜닝 시 스크립트 코드 변경 없이 `default.yml`의 키 값 수정이나 웹 대시보드에서의 조작만으로 전반적인 제어 권한을 행사할 수 있어야 합니다.

### 4.2 도메인 격리형 패키징 및 구조적 미학 (Domain Segregation)
- 프레임워크 핵심 코드는 물리적 영역에 따라 `dataloader/`, `model/`, `util/` 도메인 폴더로 철저히 세분화되어 격리되어야 합니다.
- 외부와 접촉하는 통로는 오직 `automl_framework` 루트의 `__init__.py` 파사드(Facade) 레이어만 제공하여 격식 있는 결합도 제어를 성취해야 합니다.
