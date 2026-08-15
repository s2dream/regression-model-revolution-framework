# [AutoML Regression Framework] 인터페이스 및 API 명세서 (Interface Specification)

본 문서는 **AutoML Regression Framework**의 외부 진입점(CLI, WebUI), 중앙 설정 파일(YAML Schema), 파사드 및 도메인 클래스 API, SHAP 해석기 API, 데이터 교환 인터페이스 규격을 정의한 **인터페이스 명세서 (Interface & API Specification)**입니다.

---

## 1. 개요 (Overview)

본 명세서는 프레임워크를 라이브러리로 연동하거나, CLI 및 웹 환경에서 제어할 때 사용되는 모든 인터페이스와 데이터 교환 규격을 표준화하여 제공합니다.

---

## 2. CLI 명령줄 인터페이스 (Command-Line Interface)

루트의 `main.py`는 명령줄 인수를 통해 파이프라인의 핵심 동작 파라미터를 제어합니다.

### 2.1 CLI 인수 정의
| 옵션 명칭 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/default.yml` | 파이프라인 구동용 YAML 설정 프로파일 경로 |
| `--run-id` | `str` | `None` (자동 생성) | 고유 실행 아이디 (예: `run_20260816_120000`). 미지정 시 날짜시간 기반 자동 생성 |
| `--overwrite-run` | `flag` | `False` | 동일한 `--run-id` 디렉토리가 이미 존재할 때 덮어쓰기 허용 (미지정 시 중복 시 `FileExistsError` 발생 후 안전 종료) |
| `--dataset-path` | `str` | `data/synthetic_regression.csv` | 로컬 입력 데이터셋 파일 경로 (CSV, TSV, JSONL 등) |
| `--kaggle-dataset`| `str` | `None` | Kaggle 데이터셋 식별자 (예: `user/dataset-name`) |
| `--url` | `str` | `None` | 원격 HTTP 다운로드 데이터 파일 URL |
| `--target` | `str` | `None` | 타겟 컬럼명 (설정 파일의 `target_column`을 오버라이드) |
| `--test-size` | `float`| `None` | 테스트 데이터 분할 비율 (예: `0.2`) |
| `--enable-shap` | `flag` | `False` | SHAP 모델 해석 및 피처 기여도 분석 활성화 |
| `--shap-model` | `str` | `Champion` | SHAP 분석 대상 모델명 (`Champion`, `TabICL`, `XGBoost` 등) |

---

## 3. 중앙 YAML 설정 스키마 규격 (Configuration Schema)

`configs/default.yml`을 비롯한 설정 파일들은 아래와 같은 계층 구조를 가집니다.

```yaml
framework:
  random_state: 42                       # 재현성을 위한 글로벌 난수 시드 (int)
  target_column: "target"                # 종속 변수(Target) 컬럼명 (str)
  test_size: 0.2                         # 테스트 데이터셋 분할 비율 (float, 0.0 ~ 1.0)
  active_models:                         # 파이프라인에서 활성화할 모델 목록 (list of str)
    - "XGBoost"
    - "CatBoost"
    - "RandomForest"
    - "MLP"
    - "TabPFN"
    - "TabICL"
    - "Transformer"

hpo:
  enabled: false                         # Optuna 자동 튜닝 기동 여부 (bool)
  n_trials: 10                           # 모델별 하이퍼파라미터 최적화 시도 횟수 (int)
  metric: "RMSE"                         # 최적화 대상 평가지표: "RMSE", "MAE", "R2" (str)

shap:
  enabled: false                         # SHAP 모델 해석 및 피처 기여도 분석 활성화 여부 (bool)
  model: "Champion"                      # 분석 대상 모델: "Champion", "TabICL", "XGBoost", "CatBoost", 등 (str)
  max_samples: 100                       # SHAP 값 계산 시 사용할 최대 테스트 샘플 수 (int)

data:
  data_dir: "data"                       # 데이터 파일 저장 디렉토리 (str)
  output_dir: "outputs"                  # 산출물 저장 디렉토리 (str)
  ignored_columns: []                    # 전처리 시작 전 제외할 컬럼 목록 (list of str)
  feature_columns: []                    # 명시적으로 사용할 독립 변수 목록 (비어있으면 전체 사용)
  split:
    method: "train_test_split"           # 분할 전략: "train_test_split", "kfold", "timeseries"
    test_size: 0.2

models:                                  # 개별 모델별 하이퍼파라미터 사전 (kwargs로 바인딩)
  XGBoost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  CatBoost:
    iterations: 100
    depth: 6
    learning_rate: 0.1
    verbose: 0
  RandomForest:
    n_estimators: 100
    max_depth: null
  MLP:
    hidden_layer_sizes: [128, 64]
    max_iter: 500
  TabPFN:
    N_ensemble_configurations: 32
  TabICL:
    n_estimators: 8
    device: "cpu"
    batch_size: 4
```

---

## 4. 핵심 컨트롤러 & 도메인 클래스 API

### 4.1 `AutoMLPipeline` (`main.py`)
```python
class AutoMLPipeline:
    def __init__(
        self,
        config_path: str = "configs/default.yml",
        turn: int = 1,
        target: Optional[str] = None,
        test_size: Optional[float] = None,
        enable_shap: Optional[bool] = None,
        shap_model: Optional[str] = None
    ) -> None:
        """설정 파일 로드, CLI 인수 병합, 서브시스템 및 SHAPAnalyzer 인스턴스화"""

    def prepare_data(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: Optional[str] = None,
        url: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터를 획득, 결측치 보정/인코딩 및 train/test로 분할하여 반환"""

    def train_and_evaluate(self) -> Dict[str, Dict[str, float]]:
        """활성 모델 풀에 대해 HPO 튜닝 및 학습을 수행하고 평가 메트릭 반환"""

    def generate_reports(self) -> None:
        """프리미엄 다크 테마 차트, HTML 대시보드, Markdown 요약 및 JSON 리포트를 outputs/ 에 저장"""

    def run_shap_analysis(self) -> Optional[Dict[str, Any]]:
        """지정된 대상 모델(또는 Champion)에 대해 SHAP 피처 기여도 분석 및 리포트 파일 생성"""

    def run(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: Optional[str] = None,
        url: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """prepare_data -> train_and_evaluate -> generate_reports -> (run_shap_analysis) 순차 실행"""
```

---

### 4.2 `Visualizer` (`automl_framework.util.visualizer`)
```python
class Visualizer:
    def __init__(self, output_dir: str = "outputs") -> None:
        """Visualizer 초기화 및 출력 폴더 생성"""

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """실제값 vs 예측값 산포도 및 y=x 대각선 플롯 생성"""

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, turn: int = 1) -> str:
        """잔차 산포도 플롯 생성"""

    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]], metric_name: str = "RMSE", turn: int = 1) -> str:
        """모델 성능 비교 수평 막대 차트 생성"""

    def plot_learning_curve(self, loss_history: list, model_name: str, turn: int = 1) -> str:
        """학습 곡선 손실 추이 차트 생성"""

    def save_json_report(self, metrics: Dict[str, Dict[str, float]], turn: int = 1, metadata: Optional[Dict[str, Any]] = None, ...) -> str:
        """구조화 JSON 실행 리포트 저장"""

    def save_html_report(self, metrics: Dict[str, Dict[str, float]], turn: int = 1, metadata: Optional[Dict[str, Any]] = None) -> str:
        """base64 임베딩 독립 실행형 반응형 HTML 대시보드 저장"""

    def save_markdown_summary(self, metrics: Dict[str, Dict[str, float]], turn: int = 1, metadata: Optional[Dict[str, Any]] = None) -> str:
        """공유 가능한 GFM 요약 리포트 저장"""
```

---

### 4.3 `SHAPAnalyzer` (`automl_framework.util.shap_analyzer`)
```python
class SHAPAnalyzer:
    def __init__(
        self,
        output_dir: str = "outputs",
        max_samples: int = 100,
        random_state: int = 42
    ) -> None:
        """SHAP 해석기 초기화 및 출력 폴더 생성"""

    def analyze_model(
        self,
        model_wrapper: Any,
        model_name: str,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        turn: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        모델 알고리즘 유형(TabICL Dedicated, TreeExplainer, ModelExplainer)을 자동 판별하여
        SHAP 값을 계산하고 피처 중요도 수평 바 차트, Beeswarm 차트 및 JSON 리포트를 생성
        """
```

---

## 5. 결과 산출물 스키마

### 5.1 표준 실행 리포트 (`outputs/<run_id>/report.json`)
```json
{
  "run_id": "run_20260816_120000",
  "champion_model": "CatBoost",
  "champion_r2": 0.9412,
  "metrics": {
    "XGBoost": { "RMSE": 1.2345, "MAE": 0.9876, "R2": 0.9321 },
    "CatBoost": { "RMSE": 1.1201, "MAE": 0.8912, "R2": 0.9412 },
    "RandomForest": { "RMSE": 1.3412, "MAE": 1.0123, "R2": 0.9123 },
    "MLP": { "RMSE": 1.1504, "MAE": 0.9102, "R2": 0.9380 }
  },
  "metadata": {
    "target_column": "Target_Y",
    "split_method": "train_test_split",
    "train_samples": 400,
    "test_samples": 100,
    "num_features": 4,
    "random_state": 42
  },
  "timestamp": "2026-08-16T12:00:00"
}
```

### 5.2 SHAP 해석 리포트 (`outputs/<run_id>/{model}_shap_report.json`)
```json
{
  "run_id": "run_20260816_120000",
  "model_name": "RandomForest",
  "explainer_engine": "TreeExplainer",
  "num_samples_analyzed": 20,
  "num_features": 4,
  "top_features": [
    "Feature_X1",
    "Feature_X2",
    "Category_X3_low",
    "Category_X3_medium"
  ],
  "mean_abs_shap": {
    "Feature_X1": 9.7789,
    "Feature_X2": 2.9238,
    "Category_X3_low": 2.0413,
    "Category_X3_medium": 1.2719
  },
  "artifacts": [
    "RandomForest_shap_bar.png",
    "RandomForest_shap_summary.png"
  ],
  "timestamp": "2026-08-16T12:00:00"
}
```
