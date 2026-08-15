# [AutoML Regression Framework] 인터페이스 및 API 명세서 (Interface Specification)

본 문서는 **AutoML Regression Framework**의 외부 진입점(CLI, WebUI), 중앙 설정 파일(YAML Schema), 파사드 및 도메인 클래스 API, 데이터 교환 인터페이스 규격을 정의한 **인터페이스 명세서 (Interface & API Specification)**입니다.

---

## 1. 개요 (Overview)

본 명세서는 프레임워크를 라이브러리로 연동하거나, CLI 및 웹 환경에서 제어할 때 사용되는 모든 인터페이스와 데이터 교환 규격을 표준화하여 제공합니다.

---

## 2. CLI 명령줄 인터페이스 (Command-Line Interface)

루트의 `main.py`는 명령줄 인수를 통해 파이프라인의 핵심 동작 파라미터를 제어합니다.

### 2.1 CLI 인수 정의
| 옵션 명칭 | 단축키 | 타입 | 기본값 | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `-c` | `str` | `configs/default.yml` | 파이프라인 구동용 YAML 설정 프로파일 경로 |
| `--turn` | `-t` | `int` | `1` | 실험 실행 회차 번호 (출력 파일명 접미사로 사용) |
| `--dataset-path` | `-d` | `str` | `data/synthetic_regression.csv` | 로컬 입력 데이터셋 파일 경로 (CSV, TSV, JSONL 등) |
| `--kaggle-dataset`| `-k` | `str` | `None` | Kaggle 데이터셋 식별자 (예: `user/dataset-name`) |
| `--url` | `-u` | `str` | `None` | 원격 HTTP 다운로드 데이터 파일 URL |
| `--target` | - | `str` | `None` | 타겟 컬럼명 (설정 파일의 `target_column`을 오버라이드) |
| `--test-size` | - | `float`| `None` | 테스트 데이터 분할 비율 (예: `0.2`) |

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

data:
  data_dir: "data"                       # 데이터 파일 저장 디렉토리 (str)
  ignored_columns: []                    # 전처리 시작 전 제외할 컬럼 목록 (list of str)
  feature_columns: []                    # 명시적으로 사용할 독립 변수 목록 (비어있으면 전체 사용)

models:                                  # 개별 모델별 하이퍼파라미터 사전 (kwargs로 바인딩)
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  catboost:
    iterations: 200
    depth: 6
    learning_rate: 0.1
    verbose: 0
  random_forest:
    n_estimators: 100
    max_depth: null
  mlp:
    hidden_layer_sizes: [100, 50]
    max_iter: 200
  tabpfn:
    device: "cpu"
  tabicl:
    device: "cpu"
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    epochs: 20
    learning_rate: 0.001
```

---

## 4. 핵심 컨트롤러 & 파사드 API

### 4.1 `AutoMLPipeline` (`main.py`)
파이프라인 전체를 총괄하는 오케스트레이터 클래스입니다.

```python
class AutoMLPipeline:
    def __init__(
        self,
        config_path: str = "configs/default.yml",
        turn: int = 1,
        target: Optional[str] = None,
        test_size: Optional[float] = None
    ) -> None:
        """설정 파일 로드, CLI 인수 병합 및 서브시스템 인스턴스화"""

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
        """프리미엄 다크 테마 차트 및 JSON 리포트를 outputs/ 에 저장"""

    def run(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: Optional[str] = None,
        url: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """prepare_data -> train_and_evaluate -> generate_reports 순차 실행"""
```

---

### 4.2 `DataLoaderHelper` (`automl_framework.dataloader.data_loader_helper`)
데이터 처리 서브시스템의 진입점 역할을 하는 파사드 클래스입니다.

```python
class DataLoaderHelper:
    def __init__(self, data_dir: str = "data", config: Optional[dict] = None) -> None:
        """데이터 로더 헬퍼 초기화"""

    def fetch_dataset(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: Optional[str] = None,
        url: Optional[str] = None
    ) -> str:
        """로컬 경로, Kaggle 식별자, 원격 URL을 분석하여 검증된 로컬 파일 경로 반환"""

    def prepare_data(
        self,
        dataset_file: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터 로드, 전처리, 분할을 원스톱으로 수행하여 (X_train, X_test, y_train, y_test) 반환"""
```

---

## 5. 모델 도메인 인터페이스 규격

### 5.1 `ModelType` Enum (`automl_framework.model.model_factory`)
지원되는 알고리즘의 표준 상수 정의입니다.

- `ModelType.XGBOOST = "XGBoost"`
- `ModelType.CATBOOST = "CatBoost"`
- `ModelType.RANDOM_FOREST = "RandomForest"`
- `ModelType.MLP = "MLP"`
- `ModelType.TABPFN = "TabPFN"`
- `ModelType.TABICL = "TabICL"`
- `ModelType.TRANSFORMER = "Transformer"`
- **메서드**: `ModelType.from_str(name: str) -> ModelType` (대소문자/구분자 무관 안전 변환)

---

### 5.2 `ModelFactory` (`automl_framework.model.model_factory`)
```python
class ModelFactory:
    @staticmethod
    def create_model(
        model_type: Union[ModelType, str],
        config: Optional[dict] = None,
        random_state: int = 42
    ) -> ABCModelWrapper:
        """지정된 ModelType에 대응하는 표준화된 Concrete ModelWrapper 인스턴스 생성"""
```

---

### 5.3 `ABCModelWrapper` (`automl_framework.model.wrappers`)
모든 회귀 알고리즘 어댑터가 준수해야 하는 추상 인터페이스입니다.

```python
class ABCModelWrapper(ABC):
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "ABCModelWrapper":
        """데이터셋에 맞춰 원본 회귀 모델 학습"""

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """입력 피처에 대한 1차원 예측값 넘파이 배열 반환"""
```

---

### 5.4 `OptunaHPOTuner` (`automl_framework.model.hpo`)
```python
class OptunaHPOTuner:
    def __init__(self, n_trials: int = 10, random_state: int = 42) -> None:
        """HPO 튜너 초기화"""

    def run_hpo_tuning(
        self,
        pool: ModelPool,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> None:
        """풀 내 튜닝 가능 모델들에 대해 Optuna 최적화를 실행하고 pool의 모델을 최적 래퍼로 교체"""
```

---

## 6. 시각화 및 리포팅 인터페이스 (`Visualizer`)

```python
class Visualizer:
    def __init__(self, output_dir: str = "outputs") -> None:
        """출력 폴더를 보장하며 시각화 엔진 초기화"""

    def plot_actual_vs_predicted(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        model_name: str,
        turn: int = 1
    ) -> str:
        """실제값 vs 예측값 산포도 PNG 파일 저장 및 경로 반환"""

    def plot_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        model_name: str,
        turn: int = 1
    ) -> str:
        """잔차 분포 산포도 PNG 파일 저장 및 경로 반환"""

    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str = "R2",
        turn: int = 1
    ) -> str:
        """모델간 성능 비교 수평 막대 차트 PNG 파일 저장 및 경로 반환"""

    def save_json_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        turn: int = 1
    ) -> str:
        """표준 JSON 실행 보고서 저장 및 파일 경로 반환"""
```

---

## 7. 결과 산출물 스키마 (JSON Report Format)

`outputs/turn_{turn}_report.json`으로 저장되는 실행 리포트의 JSON 규격입니다.

```json
{
  "turn": 1,
  "champion_model": "CatBoost",
  "champion_r2": 0.9412,
  "metrics": {
    "XGBoost": {
      "RMSE": 1.2345,
      "MAE": 0.9876,
      "R2": 0.9321
    },
    "CatBoost": {
      "RMSE": 1.1201,
      "MAE": 0.8912,
      "R2": 0.9412
    },
    "TabICL": {
      "RMSE": 1.1504,
      "MAE": 0.9102,
      "R2": 0.9380
    }
  },
  "timestamp": "2026-08-15T10:24:00"
}
```
