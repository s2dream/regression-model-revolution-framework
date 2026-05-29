# [AutoML Regression Framework] 역요구사항 명세서 (Reverse Requirements Specification)

본 문서는 현재 구현되어 있는 **AutoML Regression Framework**의 소스 코드와 `ARCHITECTURE.md` 파일을 리버스 엔지니어링하여 시스템의 설계 의도, 핵심 기능, 그리고 품질 속성을 역으로 도출한 **요구사항 명세서**입니다.

---

## 1. 개요 (Overview)

### 1.1 시스템 정의
본 시스템은 정형 데이터(Tabular Data)를 입력받아 다양한 기계학습 회귀(Regression) 알고리즘을 활용해 성능을 일괄 학습하고 예측하는 **AutoML Regression Framework**입니다. 

### 1.2 시스템 목표
- **종합 솔루션 제공**: 데이터 수집, 전처리, 모델 학습, 성능 평가, 고화질 분석 플롯 생성 및 결과 보고서 저장을 아우르는 단일 파이프라인을 구축합니다.
- **개발 부담 최소화 및 무코드 설정**: 기계학습 모델 훈련에 소요되는 데이터 전처리(결측치 임퓨테이션, 원-핫 인코딩 등) 및 알고리즘 탐색 노력을 최소화하며, 외부 설정 파일(`config.yml`) 연동을 통해 코딩 없이 모델의 세부 하이퍼파라미터 및 구동 방식을 제어합니다.
- **높은 신뢰성과 결함 감내**: 실행 환경의 차이(특정 라이브러리 누락, OpenMP 런타임 부재 등)에도 파이프라인이 즉시 크래시되지 않고, 최선의 유효 모델 세트를 활용하여 성공적인 완료를 지향합니다.
- **세련된 분석 결과 제공**: 의사결정권자 또는 연구원에게 분석 결과를 명확하게 인지시킬 수 있는 프리미엄 다크 테마 시각화 자료와 기계 가독성이 뛰어난 구조화된 JSON 리포트를 자동 발행합니다.
- **단일 책임 및 전략적 유연성 확보**: 모델 데이터 컨테이너(`ModelPool`)와 모델 학습/평가 실행기(`BenchmarkExecutor`)를 철저히 격리(전략 패턴)하여 향후 학습 방식(교차 검증 등)의 유연한 확장을 지원합니다.

---

## 2. 시스템 요구사항 (System Requirements)

### 2.1 개발 및 실행 환경
- **언어**: Python 3.x
- **운영체제**: OS 독립적 (Linux, macOS, Windows 등 크로스 플랫폼 지원)

### 2.2 기술 스택 및 의존성 (Key Dependencies)
- **데이터 분석 및 프레임 처리**: `pandas` (>= 1.x), `numpy` (>= 1.x)
- **설정 파일 포맷**: `PyYAML` (>= 6.0)을 통한 YAML 구성 파일 로드 및 유효성 판단
- **머신러닝 & 알고리즘**:
  - `scikit-learn`: 데이터 분할(`train_test_split`), 평가 지표 계산, `RandomForestRegressor`, `MLPRegressor` 제공
  - `xgboost`: 트리 부스팅 기반 고성능 알고리즘 `XGBRegressor` 제공
  - `tabpfn`: 소/중형 정형 데이터 특화 사전 학습 정형 데이터 트랜스포머 모델 `TabPFNRegressor` 제공
- **시각화 및 파일 입출력**:
  - `matplotlib`, `seaborn`: 프리미엄 디자인 규격이 적용된 테마 차트 렌더링
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
  - **설명**: 셸 상에서 인가된 실행 인수(예: `--target`, `--test-size`, `--config`)가 있다면, 불러온 YAML 설정 파일의 수치를 우선하여 덮어써서(Override) 적용해야 합니다.
  - **Fallback**: 설정 파일이 유실되었거나 형식이 깨졌을 때에도 크래시를 뿜으며 종료되지 않고, 내부의 견고한 디폴트 딕셔너리 설정을 로드하여 파이프라인 정상 구동을 유지해야 합니다.

### 3.2 데이터 로드 및 수집 (DataLoader - Data Fetching & Loading)
* `DataLoader`는 **파사드 패턴 (Facade Pattern)**을 취해 데이터 로드, 전처리, 분할에 대한 클라이언트 단일 진입점을 제공하며, 내부적으로는 `loaders.py`에 정의된 `ABCDataLoader` 규격 하의 개별 전략 클래스들(`loaders.py`)에 실행을 위임해야 합니다.
* **REQ-DL-01: Kaggle 데이터 자동 연동 및 다운로드**
  - **설명**: Kaggle의 데이터셋 고유 식별자(예: `'sobhanmoosavi/us-accidents'`)를 제공받았을 때, 내장 `KaggleDataLoader` 연동 모듈을 호출하여 데이터를 자동으로 다운로드하고 압축을 풀어 로컬 지정 디렉토리에 배치할 수 있어야 합니다.
* **REQ-DL-02: HTTP URL 데이터 자동 다운로드**
  - **설명**: UCI 머신러닝 리포지토리 등 외부 다이렉트 다운로드 주소(URL)를 입력받았을 때, `URLDataLoader`를 통해 원본 데이터를 직접 내려받아 로컬에 보관할 수 있어야 합니다.
* **REQ-DL-03: 다양한 데이터 포맷 파싱 및 대상 열 분리**
  - **설명**: `LocalFileDataLoader`가 로컬의 `.csv`, `.tsv`, `.txt` (탭 구분자 포맷), `.parquet` 확장자를 자동으로 감지하여 적절한 판다스 판독 엔진을 통해 DataFrame 형태로 읽어 들일 수 있어야 합니다.
  - **커스텀 피처 지정**: `feature_columns` 옵션을 지원하여, 지정된 특정 피처 컬럼들만 데이터셋에서 안전하게 추출해 $X$ 매트릭스를 구성하고 타겟 $y$를 매핑하는 기능을 완수해야 합니다.

### 3.3 데이터 전처리 및 분할 (DataLoader - Preprocessing & Splitting)
* **REQ-DL-04: 자동 결측치 보정 및 범주형 데이터 변환 (Preprocessing)**
  - **설명**: `StandardDataPreprocessor`를 호출하여 수치형 결측치는 열의 **중앙값(Median)**으로, 범주형 결측치는 열의 **최빈값(Mode)**으로 임퓨팅한 후 **더미 변수화(Dummy Encoding, 원-핫 인코딩)**를 자동으로 실행해야 합니다. 다중공선성 문제를 방지하기 위해 첫 번째 범주를 제외하는 `drop_first=True` 전략을 고수합니다.
* **REQ-DL-05: 난수 고정을 포함한 데이터셋 분할 (Data Splitting)**
  - **설명**: 다양한 스플릿 방식(`TrainTestSplitter`, `KFoldSplitter`, `TimeSeriesSplitter`) 중 설정 프로파일에 지정된 기법을 선택적으로 로딩 및 동작시켜 데이터를 학습(Train) 및 최종 성능 측정(Test) 셋으로 분할하되, 난수 시드(`random_state`)를 지정하여 실행 재현성을 확보해야 합니다.

### 3.4 통일화된 모델 제어 및 모델 저장소 구성 (ModelPool & ModelWrapper)
* **REQ-MP-01: 어댑터 패턴 기반 인터페이스 규격화 (ModelWrapper)**
  - **설명**: 다양한 외부 프레임워크 기반 모델(Scikit-Learn, XGBoost, TabPFN 등)의 제어 체계를 표준화하기 위해 `fit(X, y)` 및 `predict(X)`라는 공통 규격을 제공하는 **어댑터(Adapter) 패턴**을 내재화해야 합니다.
* **REQ-MP-02: 순수 모델 데이터 저장소 구성 (ModelPool)**
  - **설명**: `ModelPool`은 동작 제어 성격의 메서드를 모두 배제하고, 오직 가용한 모델 인스턴스들을 YAML 설정 기반으로 안전하게 생성하고 보관하는 **순수 데이터 컨테이너(Inventory Repository)**의 책임만 완수해야 합니다.
* **REQ-MP-03: 환경 결함에 따른 패키지 부재 시의 생존력 보장 (Robust Shielding)**
  - **설명**: 사용자 개발 환경에 `xgboost`나 `tabpfn` 라이브러리가 미설치되었거나, Mac OSX 환경에서 XGBoost에 필수적인 OpenMP `libomp.dylib` 런타임 파일이 누락되어 링크 로딩 에러가 발생하는 등 어떠한 물리적 결함이 발생하더라도, 프로그램이 즉사하지 않고 런타임 경고(Warning) 로그를 안전하게 우회 출력한 뒤 가용한 나머지 정상 모델들만으로 풀을 완성시켜 구동의 중단을 차단해야 합니다.

### 3.5 실행 전략 분리 및 일괄 처리 (BenchmarkExecutor)
시스템은 다형성을 실현하여 모델 인벤토리와 파이프라인 구동 루프를 완벽히 격리해야 합니다.

* **REQ-EX-01: 실행기 추상화와 전략 주입 (Strategy Pattern)**
  - **설명**: 일괄 학습, 일괄 성능 지표 산출, 일괄 추론의 제어 흐름을 추상화한 인터페이스인 `ABCModelExecutor`를 `model_executor.py`에 정의하고, 구체적인 일괄 벤치마크 학습 루프 전략인 `StandardBenchmarkExecutor`를 함께 제공합니다.
  - **유연성**: 추후 교차 검증용 실행기 (`CrossValidationExecutor`) 등으로 전략을 전환할 때, 내부 데이터 저장 구조(`ModelPool`)의 변경 없이 손쉽게 실행 흐름을 통째로 갈아 끼워 교체할 수 있어야 합니다.
* **REQ-EX-02: 예외 감내형 일괄 학습 및 스코어링 (Executor Fit & Predict)**
  - **설명**: 주입받은 `ModelPool` 내 가용한 활성 모델 전체에 대해 학습 루프(`fit_all`)를 돌리고, 테스트 셋 평가(`evaluate_all`) 및 일괄 예측값 수집(`get_predictions`)을 안전하게 대행해 그 결과를 구조화된 사전 형태로 리포팅해야 합니다.

### 3.6 평가 지표 산출 및 시각화 (Evaluation & Visualizer)
* **REQ-EV-01: 회귀 모델 전용 정량 메트릭 일괄 계산**
  - **설명**: 테스트 데이터 셋에 대해 모든 활성 모델별 예측값을 도출한 뒤, 실제값과 매칭하여 **RMSE, MAE, R² Score** 지표를 계산해야 합니다.
* **REQ-VI-01: 프리미엄 다크 테마 디자인 시스템 (Aesthetic Standards)**
  - **설명**: 차트 캔버스 배경색 차콜 블랙 슬레이트 테마 `#0d1117`, 도표 내부 `#161b22`, 그리드 `#30363d` 및 파스텔 하모니 팔레트(블루, 코랄, 민트 등)를 엄격히 고수하여 렌더링해야 합니다.
* **REQ-VI-02: 실제값 vs 예측값 대칭 분석 차트 (Actual vs Predicted Scatter Plot)**
  - **설명**: 예측치 분포 산포도를 그릴 때 투명 파스텔 블루 점들을 플로팅하고, 완벽한 오차 없는 예측을 상징하는 **대각선 perfect fit line(y=x)**을 코랄 레드의 점선 형태로 명확하게 도식화해 저장해야 합니다.
* **REQ-VI-03: 잔차 분포 산포도 (Residual Scatter Plot)**
  - **설명**: 등분산성 판별을 위해 x축 예측치, y축 잔차 산포도와 함께 오차 0 수평 기준선을 뚜렷하게 그려 넣어야 합니다.
* **REQ-VI-04: 일괄 성능 바 차트 (Model Comparison Horizontal Bar Chart)**
  - **설명**: 모델 성능 순위에 맞게 내림/오름차순 정렬된 깔끔한 수평 막대 차트를 그리고, 끝단 레이블 공간에 지표 값을 명기해 줍니다.
* **REQ-VI-05: 이력 관리용 회차별 구조화 JSON 리포트 생성 (Structured Metadata Reporting)**
  - **설명**: 실행 회차 번호(Turn ID) 및 모델별 성능 평갓값(R², RMSE, MAE), 그리고 가장 우수한 R² 점수를 획득한 최종 Champion Model명과 저장소 경로를 담아 표준 JSON 리포트 파일(`turn_{turn}_report.json`)을 자동으로 보존해야 합니다.

---

## 4. 비기능적 요구사항 (Non-Functional Requirements)

### 4.1 사용성 및 접근성 (Usability & Config Driven Control)
- 사용자는 `python main.py` 명령어 하나만으로 곧바로 프레임워크 전체 프로세스를 즉시 구동할 수 있어야 하며, 하이퍼파라미터 튜닝 시 스크립트 코드 변경 없이 `config.yml`의 키 값 수정만으로 전반적인 제어 권한을 행사할 수 있어야 합니다.

### 4.2 도메인 격리형 패키징 및 구조적 미학 (Domain Segregation)
- 프레임워크 핵심 코드는 물리적 영역에 따라 `dataloader/`, `model/`, `util/` 도메인 폴더로 철저히 세분화되어 격리되어야 합니다.
- 외부와 접촉하는 통로는 오직 `automl_framework` 루트의 `__init__.py` 파사드(Facade) 레이어만 제공하여 격식 있는 결합도 제어를 성취해야 합니다.

---

## 5. 데이터 흐름 및 실행 아키텍처 (Data Flow Diagram)

시스템의 파이프라인 실행에 따른 모듈 간 데이터 교환 흐름은 아래와 같습니다.

```text
               ┌───────────────────────────────┐
               │       configs/*.yml           │ (다양한 설정 프로파일 보관 디렉토리)
               └───────────────┬───────────────┘
                               │ 설정 로드 및 인수 오버라이드
                               ▼
[1. CLI Entry]  ─────> [2. Data Ingestion] ─────> [3. Data Preprocessor] ─────> [4. Data Splitting]
   (main.py)       (DataLoader Facade ->        (DataLoader Facade ->        (DataLoader Facade ->
                    Kaggle/URL/Local loader)     StandardDataPreprocessor)    TrainTest/KFold/TS splitter)
                                                                                       │
                                                                                       ▼
[8. Champion Select] <── [7. Premium Charts] <─── [6. Evaluation Metrics] <─── [5. Model Executor Fit]
   (outputs/JSON)         (util.Visualizer)       (model_executor.             (model_executor.
                                                   StandardBenchmarkExecutor)   StandardBenchmarkExecutor)
                                                                                       │
                                                                                       ▼
                                                                              [ModelPool Inventory]
                                                                                (XGBoost, MLP, RF,
                                                                                 TabPFN, CatBoost)
```

1. **config.yml 로드**: 프로그램 부팅과 함께 `configs/` 내부의 설정 프로파일(기본 configs/default.yml) 파일을 우선 읽어 들이고 셸 환경 인수로 보정하여 정책을 정의합니다.
2. **CLI Entry & Ingestion**: Kaggle API, UCI HTTP, 로컬 CSV 판독 또는 모의 자가 생성 루틴을 돌려 원본 데이터프레임을 생성하고 타겟 벡터를 분리합니다 (`loaders.py` 위임).
3. **Data Preprocessor**: 누락 값 검측 후 수치열은 중앙값, 범주열은 최빈값 임퓨팅 및 원-핫 인코딩(Dummy 변수 변환)을 거쳐 정규 매트릭스로 조율합니다 (`preprocessors.py` 위임).
4. **Data Splitting**: 재현용 난수 시드를 걸어 학습 및 평가용 테스트 데이터로 조각냅니다 (`splitters.py` 위임).
5. **Model Executor Fit**: 설정된 `active_models` 목록으로 빌드되어 `ModelPool`에 저장된 가용 모델 어댑터들을 `StandardBenchmarkExecutor`가 일괄 학습시킵니다.
6. **Evaluation**: 계산된 예측값을 오리지널 정답과 매칭해 RMSE, MAE, R² 메트릭으로 요약하여 콘솔 표로 표시합니다.
7. **Premium Charts**: 고급 다크 슬레이트 테마 레이아웃 위에 실제값 대 예측값 분포, 잔차 오차 분포, 모델 성능 순위 수평 바 플롯을 그려 고화질 PNG 이미지로 기록합니다.
8. **Champion Select**: R² 성능이 가장 뛰어난 1위 최적의 챔피언 모델(Best Model) 명칭을 검출하여 JSON 아카이브 리포트 파일 저장을 끝으로 모든 시퀀스를 무사히 완료합니다.
