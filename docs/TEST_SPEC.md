# [AutoML Regression Framework] 소프트웨어 테스트 계획 및 명세서 (Software Test Specification)

본 문서는 **AutoML Regression Framework**의 품질 보증(QA) 및 신뢰성을 확보하기 위한 **소프트웨어 테스트 계획서(STP) 및 테스트 명세서(STD)**입니다. 프레임워크의 핵심 모듈별 단위 테스트, 통합 테스트, E2E 파이프라인 테스트, SHAP 모델 해석 테스트 및 WebUI 헬퍼 테스트 케이스와 판정 기준을 체계적으로 정의합니다.

---

## 1. 개요 (Overview)

### 1.1 목적
- 프레임워크를 구성하는 데이터 로더, 전처리기, 모델 팩토리, 개별 모델 래퍼, Optuna HPO 튜너, SHAP 해석기, 시각화 모듈, 그리고 WebUI 보조 기능의 기능적 무결성을 검증합니다.
- 외부 라이브러리 결함이나 런타임 환경 차이(패키지 누락, OpenMP 부재 등)에서도 프레임워크가 크래시 없이 안전하게 예외를 우회(Shielding)하는지 확인합니다.
- 코드 변경 및 기능 확장 시 발생할 수 있는 잠재적 회귀 버그(Regression Bug)를 사전에 차단합니다.

### 1.2 테스트 범위 (Scope)
- **단위 테스트 (Unit Testing)**: `DataLoader`, `Preprocessors`, `Splitters`, `ModelFactory`, `ModelWrappers`, `OptunaHPOTuner`, `SHAPAnalyzer`, `Visualizer`, `Logger`
- **통합 및 E2E 테스트 (Integration & End-to-End Testing)**: 파이프라인 전체 주기 검증 (데이터 로드 -> HPO -> 모델 훈련 -> 평가 -> SHAP 피처 기여도 분석 -> 리포트 아카이빙)
- **UI 헬퍼 테스트 (WebUI Helper Testing)**: 설정 파싱, 스키마 변환, 미디어 유효성 검증 로직 검증

---

## 2. 테스트 환경 및 실행 도구

- **테스트 프레임워크**: `pytest` (>= 7.0.0), `pytest-mock` (>= 3.6.0)
- **실행 언어 및 환경**: Python 3.13 (Conda `py313`)
- **테스트 소스 위치**: `tests/`
- **테스트 실행 명령**:
  ```bash
  # 전체 테스트 스위트 상세 실행 (총 70개 테스트)
  conda run -n py313 pytest tests/ -v

  # SHAP 전용 테스트 실행
  conda run -n py313 pytest tests/test_shap.py -v
  ```

---

## 3. 세부 테스트 케이스 명세서 (Test Cases Specification)

### 3.1 데이터 수집, 로딩 및 전처리 (`tests/test_dataloader.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-DL-01** | CSV 로컬 파일 로드 검증 | `LocalFileDataLoader` | 유효한 모의 CSV 파일 경로 및 타겟 컬럼 지정 | 피처 행렬 $X$(DataFrame)와 타겟 $y$(Series)가 정상 분리되어 로드됨 |
| **TC-DL-02** | JSONL 동적 스키마 로딩 검증 | `LocalFileDataLoader` | 행마다 키 구성이 다른 JSONL 파일 | 새롭게 발견된 컬럼이 동적으로 병합되고 누락된 값은 `NaN`으로 안전 매핑됨 |
| **TC-DL-03** | 제외 컬럼(Ignored Columns) 필터링 | `LocalFileDataLoader` | `ignored_columns=['id', 'date']` 옵션 부여 | 원본 데이터프레임에서 지정된 컬럼들이 첫 단계에서 정상적으로 제거됨 |
| **TC-DL-04** | 결측치 임퓨테이션 검증 | `StandardDataPreprocessor` | 수치형 및 범주형 결측치(`NaN`)가 포함된 데이터 | 수치형은 중앙값(Median), 범주형은 최빈값(Mode)으로 대체되어 결측치 0개 달성 |
| **TC-DL-05** | One-Hot Dummy 인코딩 검증 | `StandardDataPreprocessor` | 범주형 문자열 열을 포함한 데이터 | `drop_first=True` 기준의 원-핫 인코딩 수치 행렬로 변환됨 |
| **TC-DL-06** | Train/Test 데이터 분할 검증 | `TrainTestSplitter` | `test_size=0.2`, 고정 난수 시드 `random_state=42` | 지정된 비율(80:20)로 데이터가 정확히 분할되며 결과 재현성이 보장됨 |
| **TC-DL-07** | K-Fold 교차 분할 검증 | `KFoldSplitter` | `n_splits=5`, `shuffle=True` | 5개 폴드의 Train/Validation 인덱스가 정확히 분할됨 |
| **TC-DL-08** | TimeSeries 시계열 분할 검증 | `TimeSeriesSplitter` | `n_splits=3` | 시간 순서를 거스르지 않는 시계열 분할이 정상 수행됨 |

---

### 3.2 모델 팩토리 및 타입 열거형 (`tests/test_model_factory.py`, `tests/test_model.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-MF-01** | ModelType 문자열 파싱 검증 | `ModelType.from_str()` | 대소문자/하이픈/언더스코어가 혼합된 문자열 (`xgboost`, `Tab-PFN`, `TabICL`) | 매칭되는 `ModelType` Enum 상수를 정확하게 반환 |
| **TC-MF-02** | 미지원 모델 문자열 예외 처리 | `ModelType.from_str()` | 존재하지 않는 임의의 모델명 (`UnknownModel`) | `ValueError` 예외를 명확하게 발생시킴 |
| **TC-MF-03** | Factory 기반 래퍼 인스턴스 생성 | `ModelFactory.create_model()` | 각 `ModelType`과 기본 하이퍼파라미터 딕셔너리 | `ABCModelWrapper` 규격을 구현한 인스턴스가 올바르게 반환됨 |
| **TC-MF-04** | ModelPool 초기화 및 커스텀 모델 등록 | `ModelPool` | 기본 config 및 `add_custom_model()` 호출 | 활성 모델들이 풀에 적재되고, 신규 커스텀 모델이 안전하게 등록 및 조회됨 |

---

### 3.3 개별 회귀 모델 래퍼 (`tests/test_all_models.py`, `tests/test_tabicl.py`, `tests/test_transformer_regression.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-MOD-01** | XGBoost 래퍼 학습 및 추론 | `ModelWrapperXGBoost` | 합성 회귀 데이터셋 ($X_{train}, y_{train}, X_{test}$) | `fit` 성공 후 `predict` 호출 시 1차원 실수 넘파이 배열 반환 |
| **TC-MOD-02** | CatBoost 래퍼 학습 및 추론 | `ModelWrapperCatBoost` | 합성 회귀 데이터셋 | `fit` 성공 후 유효한 예측값 도출 |
| **TC-MOD-03** | RandomForest 래퍼 학습 및 추론 | `ModelWrapperRandomForest`| 합성 회귀 데이터셋 | `fit` 성공 후 유효한 예측값 도출 |
| **TC-MOD-04** | MLP 래퍼 학습 및 추론 | `ModelWrapperMLP` | 합성 회귀 데이터셋 | `fit` 성공 후 유효한 예측값 도출 |
| **TC-MOD-05** | TabPFN 래퍼 추론 검증 | `ModelWrapperTabPFN` | 1000행 미만의 합성 정형 데이터 | 사전학습 모델 기반으로 신속하게 예측값 반환 |
| **TC-MOD-06** | TabICL In-Context Learning 검증 | `ModelWrapperTabICL` | 합성 정형 데이터셋 | In-Context 러닝 방식으로 정상 학습/예측 수행 및 $R^2$ 산출 |
| **TC-MOD-07** | PyTorch Transformer 회귀 신경망 검증| `ModelWrapperTransformer` | 스칼라 회귀 및 Gaussian NLL 분포 예측 옵션 | 3D 시퀀스 자동 변환 후 미니배치 에포크 학습 완료 및 1D 예측값 반환 |

---

### 3.4 Optuna 기반 하이퍼파라미터 최적화 (`tests/test_hpo.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-HPO-01** | HPO 튜너 목적함수 검증 | `OptunaHPOTuner` | `n_trials=3`, 대상 모델 `XGBoost` | Optuna Study가 정상 실행되고 Validation RMSE가 최소화되는 방향으로 수렴 |
| **TC-HPO-02** | HPO 완료 후 Best Model 갱신 | `OptunaHPOTuner.run_hpo_tuning()` | `ModelPool` 주입 및 훈련 데이터 | 최적 파라미터로 생성된 신규 래퍼 인스턴스가 `ModelPool` 내 기존 모델을 안전하게 대체 |
| **TC-HPO-03** | HPO 스킵 모델 방어 | `OptunaHPOTuner` | `TabPFN`, `TabICL` 등 튜닝 불필요 모델 | 에러 없이 안전하게 튜닝을 스킵하고 기본 모델 상태 유지 |
| **TC-HPO-04** | HPO MAE/R2 커스텀 메트릭 검증 | `OptunaHPOTuner` | `metric="MAE"` 또는 `metric="R2"` | 선택된 지표에 맞춰 목적함수가 정확히 최적화 방향(최소화/최대화)으로 수렴 |

---

### 3.5 SHAP 모델 해석 및 피처 기여도 분석 (`tests/test_shap.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-SHAP-01**| SHAPAnalyzer 초기화 및 디렉토리 생성 | `SHAPAnalyzer.__init__()` | 출력 디렉토리 경로 지정 | 디렉토리가 생성되고 max_samples 파라미터가 정상 바인딩됨 |
| **TC-SHAP-02**| TabICL 전용 In-Context Explainer 검증 | `SHAPAnalyzer.analyze_model()` | 학습된 `ModelWrapperTabICL` 인스턴스 | `explainer_engine`이 "TabICL Dedicated In-Context Explainer"로 식별되고 피처 중요도 및 JSON 리포트/차트가 생성됨 |
| **TC-SHAP-03**| TreeExplainer 엔진 검증 | `SHAPAnalyzer.analyze_model()` | 학습된 `ModelWrapperRandomForest` 또는 `XGBoost` | `explainer_engine`이 "TreeExplainer"로 매핑되고 고속 SHAP 값이 정확히 계산됨 |
| **TC-SHAP-04**| Kernel/ModelExplainer 엔진 검증 | `SHAPAnalyzer.analyze_model()` | 학습된 `ModelWrapperMLP` 인스턴스 | `explainer_engine`이 "ModelExplainer"로 식별되고 정상적인 SHAP 리포트 반환 |
| **TC-SHAP-05**| 결함 모델 예외 감내 쉴딩 검증 | `SHAPAnalyzer.analyze_model()` | 비정상/결함 커스텀 모델 인스턴스 | 크래시 없이 에러를 로깅하고 안전하게 `None`을 반환 |
| **TC-SHAP-06**| 파이프라인 SHAP 통합 실행 검증 | `AutoMLPipeline.run()` | `enable_shap=True`, `shap_model='RandomForest'` | 파이프라인 전체 완료 후 `turn_{turn}_{model}_shap_report.json`이 디스크에 정상 생성됨 |

---

### 3.6 시각화 및 리포트 파일 아카이빙 (`tests/test_visualizer.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-VIS-01** | 실제값 vs 예측값 산포도 생성 | `Visualizer.plot_actual_vs_predicted()` | 실제 $y$ 벡터, 예측 $y_{pred}$ 벡터, 모델명 | `outputs/` 디렉토리에 다크 테마 PNG 이미지 파일이 생성되고 크기가 0보다 큼 |
| **TC-VIS-02** | 잔차 분포 산포도 생성 | `Visualizer.plot_residuals()` | 실제 $y$ 벡터, 예측 $y_{pred}$ 벡터, 모델명 | `outputs/` 디렉토리에 유효한 잔차 분석 PNG 이미지 파일이 생성됨 |
| **TC-VIS-03** | 모델 성능 비교 수평 막대 차트 생성 | `Visualizer.plot_model_comparison()` | 모델별 메트릭 사전 (R2, RMSE 등) | 모델 순위별 정렬된 수평 막대 비교 차트 PNG 파일이 생성됨 |
| **TC-VIS-04** | 회차별 JSON 실행 리포트 저장 | `Visualizer.save_json_report()` | 지표 딕셔너리 및 turn 번호 | `outputs/turn_{turn}_report.json` 파일이 표준 JSON 스키마로 유효하게 저장됨 |
| **TC-VIS-05** | 인터랙티브 HTML 대시보드 저장 | `Visualizer.save_html_report()` | 지표 딕셔너리, metadata, turn | base64 차트가 임베딩된 독립 실행형 `turn_{turn}_report.html` 파일 생성 |
| **TC-VIS-06** | Markdown 요약본 저장 | `Visualizer.save_markdown_summary()` | 지표 딕셔너리, turn | 공유 가능한 GFM `turn_{turn}_summary.md` 파일 생성 |
| **TC-VIS-07** | 학습 곡선 차트 생성 | `Visualizer.plot_learning_curve()` | `loss_history` 리스트, 모델명 | `turn_{turn}_{model}_learning_curve.png` 파일 생성 |
| **TC-VIS-08** | 종합 Markdown 진단 리포트 저장 | `Visualizer.save_markdown_report()` | 지표, SHAP 리포트, 러닝커브 경로 | `turn_{turn}_report.md` 파일 생성 |

---

### 3.7 WebUI 헬퍼 및 유틸리티 (`tests/test_webui_helpers.py`, `tests/test_logger.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-UI-01** | 이미지 파일 유효성 검사 (정상 파일) | `is_valid_image()` | 정상적으로 생성된 1KB 이상의 PNG 이미지 | `True`를 반환하여 안전하게 UI에 렌더링 허용 |
| **TC-UI-02** | 이미지 파일 유효성 검사 (0바이트/손상) | `is_valid_image()` | 0바이트 빈 파일 또는 비이미지 파일 | `False`를 반환하여 UI 크래시 방어 |
| **TC-UI-03** | 설정 저장 및 로드 검증 | `load_config()`, `save_config()` | 임시 YAML 파일 경로 | 딕셔너리 데이터가 유실 없이 저장되고 정확히 복원됨 |
| **TC-UI-04** | 컬럼 추출 및 샘플 미리보기 검증 | `get_dataset_columns()`, `preview_dataset_sample()` | CSV/TSV 파일 경로 | 컬럼 목록 및 상위 5개 행 데이터프레임이 정상 반환됨 |
| **TC-LOG-01** | 시스템 로거 초기화 및 핸들러 등록 | `setup_logger()` | 로거 이름 및 로그 파일 경로 | 콘솔 스트림 및 파일 핸들러가 올바르게 바인딩되고 로그 메시지가 기록됨 |

---

### 3.8 End-to-End 파이프라인 통합 테스트 (`tests/test_pipeline_e2e.py`)

| 테스트 ID | 테스트 명칭 | 테스트 대상 | 입력 / 조건 | 예상 결과 |
| :--- | :--- | :--- | :--- | :--- |
| **TC-E2E-01** | E2E 합성 데이터 전체 파이프라인 구동 | `AutoMLPipeline.run()` | 모의 회귀 CSV, 기본 설정 파일, `turn=99` | 1) 데이터 수집/전처리 완료<br/>2) 가용 모델 전체 일괄 학습<br/>3) 평가 지표 산출<br/>4) 차트 PNG, HTML 및 JSON 리포트가 `outputs/`에 정상 기록됨 |
| **TC-E2E-02** | HPO 활성화 상태 E2E 파이프라인 구동 | `AutoMLPipeline.run()` | `hpo.enabled: true`, `n_trials=2` | Optuna 튜닝 후 학습 및 평가가 중단 없이 원활하게 완료됨 |

---

## 4. 품질 판정 기준 (Acceptance Criteria)

1. **테스트 성공률 100%**: 전체 `pytest` 테스트 스위트의 모든 테스트 케이스(총 70개)가 Pass되어야 합니다 (`0 failed`).
2. **무결점 쉴딩 검증**: 지원 라이브러리가 미설치된 환경에서도 `ModelFactory`, `ModelPool`, `SHAPAnalyzer`가 예외로 비정상 종료되지 않고 정상 구동되어야 합니다.
3. **아티팩트 무결성**: 생성된 모든 JSON/HTML/Markdown 리포트와 PNG 차트 파일은 파싱 가능하고 0 바이트가 아니어야 합니다.
