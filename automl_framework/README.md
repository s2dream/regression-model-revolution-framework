# 🚀 Regression Model Revolution Framework - 서브패키지

이 디렉토리는 정형 데이터(Tabular Data)를 활용한 고성능 머신러닝에 초점을 맞춘 **AutoML Regression Framework**의 코어 서브패키지를 포함하고 있습니다. 클래식 머신러닝 모델, 신경망 추정기, 최신 정형 데이터 모델인 **TabPFN** 및 **XGBoost** 래퍼와 함께 자동화된 시각화 리포팅 분석 도구를 통합하여 제공합니다.

---

## 📂 내부 패키지 아키텍처

프레임워크 코어는 응집도가 높고 독립적인 성격의 세부 도메인 패키지들로 나뉘어 설계되었습니다:

```
automl_framework/
│
├── __init__.py          # 상위 Facade 레이어 (DataLoaderHelper, ModelPool, Visualizer, Executor 노출)
├── README.md            # [본 파일] 서브패키지 기술 문서
│
├── dataloader/          # 데이터 가공 도메인 (DataLoader Facade & 전략 클래스)
│   ├── __init__.py
│   ├── base.py          # 로더, 전처리, 스플리터 전용 추상 베이스 클래스 (ABC)
│   ├── loaders.py       # 로컬 파일, Kaggle, 원격 URL 대상 모듈별 파일 로더
│   ├── preprocessors.py # 표준 결측치 보정(Median/Mode) 및 더미 인코더
│   ├── splitters.py     # 데이터셋 학습/테스트 분할 전략 (Train-Test, K-Fold, Time-Series)
│   └── data_loader_helper.py # DataLoaderHelper Facade 클래스 및 파이프라인 일괄 준비자
│
├── model/               # 모델 및 학습 도메인 (모델 풀, 생성 팩토리, 실행 전략)
│   ├── __init__.py
│   ├── model_pool.py    # 활성 머신러닝 모델 인벤토리를 보유하는 ModelPool 데이터 컨테이너
│   ├── model_factory.py # Factory Method 패턴을 준수하는 ModelFactory 및 ModelType Enum
│   ├── hpo.py           # Optuna 기반 자동 하이퍼파라미터 최적화(HPO) 튜너
│   ├── model_executor.py# 실행 전략 인터페이스 (ABCModelExecutor & StandardBenchmarkExecutor)
│   ├── wrappers.py      # Scikit-Learn, XGBoost, TabPFN, CatBoost, Transformer 규격 통일 래퍼
│   └── architecture/    # PyTorch 모델 정의 레이어
│       └── transformer_encoder.py # PyTorch 기반 시퀀스 회귀 트랜스포머 (TransformerBasedRegression)
│
└── util/                # 시각화 및 결과 리포팅 도메인 (다크 테마 차트 렌더링)
    ├── __init__.py
    └── visualizer.py    # Matplotlib/Seaborn 기반 시각화 및 결과 리포팅 JSON 작성
```

---

## ✨ 기능 특징 및 세부 역할

### 1. 데이터 수집 및 가공 (`dataloader/`)
- **Facade 전략 패턴 (Facade Strategy Pattern)**: `DataLoaderHelper` 클래스가 모든 가공 단계의 단일 진입점이 되어 로딩, 전처리, 스플릿의 동작을 하위 전략 인스턴스에 안전하게 위임하고, 통합 실행 메소드 `load_and_preprocess_data`를 노출합니다.
- **Kaggle API 연동 (`loaders.py`)**: Kaggle 데이터셋 식별자(ID) 입력 시 Kaggle 공식 CLI API를 호출하여 데이터 원본을 자동 다운로드합니다.
- **다이렉트 HTTP 다운로드 (`loaders.py`)**: UCI 머신러닝 저장소나 임의의 원격 URL로부터 데이터를 웹 소켓 스트림으로 즉시 다운로드하여 보관할 수 있습니다.
- **결측치 및 인코딩 자동 보정 (`preprocessors.py`)**: 수치형 변수는 **중앙값(Median)**, 범주형 변수는 **최빈값(Mode)**으로 결측치를 정밀 보정하고 범주형 열은 더미화(Dummy / One-Hot Encoding)를 자동으로 전개합니다.
- **유연한 분할 전략 (`splitters.py`)**: 기본적인 Train-Test 분할 외에도 최적 튜닝을 위한 3-way 검증 분할 및 K-Fold, Time-Series 스플릿 기법을 지원합니다.

### 2. 모델 풀 및 실행 전략 (`model/`)
- **ModelPool 컨테이너 (`model_pool.py`)**: 복잡한 실행 로직을 제외하고, 오직 설정에 부합하는 활성 모델 래퍼들을 저장하고 조회하는 **순수 데이터 컨테이너(Inventory Repository)**의 역할을 담당합니다.
- **Model Factory 설계 (`model_factory.py`)**: 객체 생성의 복잡성을 외부로 격리하기 위해 **Factory Method** 패턴을 적용하였습니다. 대소문자나 문장 형식에 무관하게 매핑되는 `ModelType` Enum을 기반으로 최적화된 하이퍼파라미터 사전을 받아 Wrapper 인스턴스를 조립합니다.
- **Optuna HPO 최적화 (`hpo.py`)**: 학습 단계 진입 전, 훈련 셋의 일부 분할 영역에서 **Optuna** 최적화 알고리즘을 가동하여 개별 튜닝 대상 모델의 RMSE 오차를 최소화하는 하이퍼파라미터를 찾고, 모델 풀을 동적 갱신합니다 (TabPFN 등 튜닝 불필요 모델은 스킵).
- **Benchmark 실행 분리 (`model_executor.py`)**: 모델 풀의 결합을 분리하기 위해 학습 및 추론 제어 흐름을 `ABCModelExecutor` 전략 객체로 추상화하였습니다. 기본으로 `StandardBenchmarkExecutor`를 제공하며, 향후 분산 학습이나 교차 검증용 실행 전략으로의 치환을 매끄럽게 지원합니다.
- **표준 어댑터 Wrapper (`wrappers.py`)**: Scikit-Learn 계열, XGBoost, CatBoost, TabPFN, 그리고 PyTorch 신경망 모델들을 동일한 호출 표준(`fit`, `predict`) 하에 구동할 수 있도록 감싸며, 런타임 쉴드 예외 처리와 함께 반복 학습 모델의 경우 이터레이션별 Loss 정보를 보관하여 반환하는 `.get_loss_history()` 기능을 탑재하고 있습니다.
- **PyTorch Transformer 기반 회귀 (`architecture/transformer_encoder.py`)**: 순차 피처 인코딩 및 Position Embedding을 연동하고, pooling(`mean`, `max`, `last`) 및 masking을 지원하는 트랜스포머 회귀 신경망을 제공합니다. 결정론적 단일 값(scalar) 예측 외에도 평균과 strictly positive 분산을 학습하여 확률 분포를 추정하는 Gaussian NLL Loss 기반 학습 모드를 지원합니다.

### 3. 고품질 분석, 설명력 및 상세 리포팅 (`util/visualizer.py`)
전문가 분석 수준의 다크 슬레이트 테마의 그래픽 차트와 문헌 형태의 리포트를 자동 발행합니다:
- **실제치 vs 예측치 산포도**: 실제값과 예측치 간의 오차 분포를 파스텔톤 플롯으로 매핑하고 이상적인 1:1 완벽 가이드 선(y=x)을 함께 렌더링합니다 (`turn_{turn}_{model_name}_actual_vs_pred.png`).
- **잔차 오차 분석 플롯**: 예측 오차의 분산 경향성과 등분산성을 한눈에 진단할 수 있는 잔차 분석 scatter plot을 그립니다 (`turn_{turn}_{model_name}_residuals.png`).
- **모델 간 비교 바 차트**: 성능비교 목적으로 모델별 $R^2$, $RMSE$, $MAE$ 성능 지표를 가로 막대 형태로 시각화하여 순위와 차이를 시각적 표현합니다 (`turn_{turn}_model_comparison_{metric}.png`).
- **SHAP 설명력 플롯**: 피처 기여도 해석을 위한 Beeswarm Plot 및 중요도 Bar Plot 차트를 저장합니다 (`turn_{turn}_{model_name}_shap_summary.png`, `turn_{turn}_{model_name}_shap_bar.png`).
- **학습 곡선 (Loss Curve)**: 반복 모델들의 에포크별 손실 감소 양상을 그려 오버피팅을 사전에 진단합니다 (`turn_{turn}_{model_name}_learning_curve.png`).
- **Markdown 상세 보고서 (Executive Report)**: 데이터 형상 분석 정보, 모델 성능 Leaderboard 테이블, 개별 차트 경로 하이퍼링크, 추천 액션 등이 총합된 마크다운 보고서를 출력합니다 (`turn_{turn}_report.md`).
- **JSON 실행 보고서**: 1위에 해당하는 Champion 모델명 및 로컬 플롯 파일 경로들을 보관하여 대시보드 데이터 바인딩을 지원합니다 (`turn_{turn}_report.json`).

---

## 📐 아키텍처 디자인 패턴 (Architecture Design Patterns)

본 프레임워크는 유지보수성과 결합도 제어를 위해 다음과 같은 객체지향 디자인 패턴들을 핵심 아키텍처 구조로 채택하고 있습니다. 패턴 학습을 위해 [Refactoring Guru(한국어 버전)](https://refactoring.guru/ko/design-patterns)의 상세 페이지를 바로 참조할 수 있습니다:

1. **[팩토리 메서드 (Factory Method) 패턴](https://refactoring.guru/ko/design-patterns/factory-method)**:
   - **적용**: [ModelFactory](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/model_factory.py)
   - **이유**: 다양한 Estimator 객체 생성 시점의 구체적인 클래스 주입 및 하이퍼파라미터 바인딩 책임을 공용 팩토리 클래스로 캡슐화하여 격리합니다.
2. **[전략 (Strategy) 패턴](https://refactoring.guru/ko/design-patterns/strategy)**:
   - **적용**: [ABCDataSplitter](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/dataloader/splitters.py) 및 [ABCModelExecutor](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/model_executor.py)
   - **이유**: 데이터셋 분할 정책(Holdout, K-Fold, TimeSeries)과 모델 실행 제어기(StandardBenchmarkExecutor)를 독립 인터페이스로 구성하여, 다른 알고리즘이나 분산 학습 환경 등으로 동적으로 교체 가능하게 지원합니다.
3. **[퍼사드 (Facade) 패턴](https://refactoring.guru/ko/design-patterns/facade)**:
   - **적용**: [DataLoaderHelper](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/dataloader/data_loader_helper.py) 및 [AutoMLPipeline](file:///Users/jeonghoon/github/regression-model-revolution-framework/main.py)
   - **이유**: 복잡하게 얽힌 데이터 로딩/전처리/스플릿 서브시스템과 모델 풀/실행기/시각화 컴포넌트 호출 절차를 상위 단일 Facade 인터페이스 메서드로 간소화하여 외부 클라이언트에 제공합니다.
4. **[어댑터 (Adapter) 패턴](https://refactoring.guru/ko/design-patterns/adapter)**:
   - **적용**: [ABCModelWrapper](file:///Users/jeonghoon/github/regression-model-revolution-framework/automl_framework/model/wrappers.py) 및 하위 모델군 래퍼
   - **이유**: Scikit-Learn 계열, XGBoost, CatBoost, TabPFN, PyTorch 등 서로 다른 모델 라이브러리의 fit/predict 및 손실 반환 명세를 `ABCModelWrapper` 규격 하에 표준화하여 일치시킵니다.

---

## 🛠️ 프로그래밍 방식 사용법 (Programmatic Usage)

본 디렉토리는 독립 패키지이므로, CLI 스크립트 외에도 자체 파이썬 프로그램 상에서 라이브러리 형태로 직접 임포트하여 활용할 수 있습니다. 

반드시 프로젝트 루트 폴더에서 작동을 권장합니다:
```bash
# 프로젝트 루트로 이동
cd /Users/jeonghoon/github/regression-model-revolution-framework
```

코드 연동 예시:
```python
# Facade 레이어로부터 깔끔하게 핵심 클래스를 임포트합니다.
from automl_framework import DataLoaderHelper, ModelPool, StandardBenchmarkExecutor, Visualizer

# 1. 데이터 수집, 결측치 임퓨테이션 및 인코딩, 분할을 원스톱으로 수행합니다.
dataloader_helper = DataLoaderHelper(data_dir="data")
X_train, y_train, X_test, y_test = dataloader_helper.load_and_preprocess_data(
    "data/your_dataset.csv", target_column="target_column_name", test_size=0.2, random_state=42
)

# 2. 모델 풀 및 실행 전략 기동
pool = ModelPool(random_state=42)
executor = StandardBenchmarkExecutor(pool)
executor.fit_all(X_train, y_train)

# 3. 모델 성능 지표 계산 및 다크 슬레이트 분석 보고서 저장
visualizer = Visualizer(output_dir="outputs")
metrics = executor.evaluate_all(X_test, y_test)
visualizer.save_json_report(metrics, turn=1)
```
