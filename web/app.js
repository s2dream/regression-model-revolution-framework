/**
 * =====================================================================
 * 🚀 AutoML Regression Studio - Frontend Application Engine
 * =====================================================================
 */

const API_BASE = window.location.origin;
const WS_BASE = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host;

// ==========================================
// 💡 COMPREHENSIVE ML HELP DOCUMENTATION DATABASE
// ==========================================
const HELP_DOCS = {
    // 1. Dataset Source
    'data_dir': {
        title: 'Data Directory',
        summary: '학습 데이터셋 원본 파일들이 위치한 로컬 폴더 경로입니다.',
        details: 'CSV, TSV, Parquet, JSONL 등 지원되는 형식의 파일들을 자동으로 스캔하여 감지합니다.',
        rec: '기본값: "data"',
        warn: '해당 폴더가 존재하지 않으면 자동으로 생성됩니다.'
    },
    'output_dir': {
        title: 'Output Base Directory',
        summary: '벤치마크 결과, 챔피언 모델 가중치, 평가 리포트, 진단 플롯이 저장될 루트 폴더입니다.',
        details: '각 실행마다 하위에 고유한 run_id 서브폴더가 생성되어 실험별로 독립 보관됩니다.',
        rec: '기본값: "outputs"',
        warn: '충분한 디스크 용량이 있는지 확인하세요.'
    },
    'data_source': {
        title: 'Data Source Mode',
        summary: '데이터셋을 인제스천하는 경로 방식을 지정합니다.',
        details: '• Local Directory: data/ 디렉토리 내 로컬 파일 선택\n• Kaggle Dataset: Kaggle API를 통한 자동 다운로드\n• Direct URL: 웹상의 원격 CSV 직접 로드',
        rec: '로컬 환경에서는 "Local Directory" 권장'
    },
    'dataset_select': {
        title: 'Local Dataset File',
        summary: '훈련 및 평가에 사용할 구체적인 데이터셋 파일입니다.',
        details: '선택 시 상위 5개 행 샘플 데이터와 컬럼 목록을 즉각 분석하여 미리보기를 제공합니다.',
        rec: '전처리가 완료된 정돈된 테이블 파일 권장'
    },
    'target_column': {
        title: 'Target Column (y)',
        summary: '회귀(Regression) 모델이 예측해야 하는 연속형 종속 변수(정답 컬럼)입니다.',
        details: '선택된 컬럼을 제외한 나머지 모든 컬럼들이 독립 특성(Features, X)으로 자동 구성됩니다.',
        rec: '예측하고자 하는 핵심 수치형 컬럼 선택',
        warn: '범주형/문자열 컬럼인 경우 전처리에서 수치 변환이 필요할 수 있습니다.'
    },
    'ignored_columns': {
        title: 'Ignored Columns',
        summary: '모델 학습에 불필요하거나 데이터 누수(Data Leakage)를 유발할 수 있어 제외할 컬럼 목록입니다.',
        details: '고유 ID, 고객 식별자, 텍스트 메모, 타겟과 직접 연동된 불필요 컬럼을 사전에 배제합니다.',
        rec: 'ID 번호나 식별자 컬럼 선택'
    },
    // 2. Partitioning
    'split_method': {
        title: 'Validation Split Strategy',
        summary: '모델의 일반화 성능을 공정하게 평가하기 위한 데이터셋 분할 방식입니다.',
        details: '• train_test_split: 무작위 단순 홀드아웃 분할 (빠른 탐색에 적합)\n• kfold: K개 폴드로 나누어 교차 검증 (과적합 방지 및 일반화 평가 최적)\n• timeseries: 시간 순서를 유지하여 미래 데이터 누수 방지',
        rec: '일반 데이터셋: kfold (5-fold) 또는 train_test_split (8:2)'
    },
    'test_size': {
        title: 'Test Size Ratio',
        summary: '전체 데이터 중 최종 성능 검증용 테스트 세트로 할당할 비율입니다.',
        details: '0.2인 경우 전체의 80%로 학습하고 20%로 최종 지표(R2, RMSE)를 산출합니다.',
        rec: '권장 범위: 0.15 ~ 0.30 (보통 0.2)',
        warn: '너무 작으면 평가 신뢰도 저하, 너무 크면 학습 데이터 부족 발생'
    },
    'n_splits': {
        title: 'Number of Folds (K)',
        summary: 'K-Fold 교차 검증 시 생성할 하위 폴드의 개수입니다.',
        details: 'K번 반복 훈련하며 모든 데이터가 검증에 1회씩 참여하여 과적합을 정밀하게 방지합니다.',
        rec: '권장값: 5 또는 10',
        warn: 'K가 커질수록 총 훈련 시간이 K배로 증가합니다.'
    },
    // 3. HPO & SHAP
    'hpo_enabled': {
        title: 'Automated HPO (Optuna)',
        summary: '베이지안 최적화 기반으로 각 모델의 최적 하이퍼파라미터를 자동 탐색합니다.',
        details: 'TPE(Tree-structured Parzen Estimator) 알고리즘으로 효율적인 하이퍼파라미터 조합을 탐색합니다.',
        rec: '최고 성능을 원할 때 활성화',
        warn: '탐색 횟수에 비례하여 실행 시간이 증가합니다.'
    },
    'hpo_trials': {
        title: 'HPO Search Trials',
        summary: '모델 하나당 시도할 하이퍼파라미터 조합의 탐색 횟수입니다.',
        details: '시도 횟수가 많을수록 더 우수한 파라미터를 찾을 확률이 높아집니다.',
        rec: '빠른 탐색: 10~20, 정밀 탐색: 50~100'
    },
    'shap_enabled': {
        title: 'SHAP Model Explainability',
        summary: '샤플리 값(Shapley Value) 기반으로 각 특성이 예측값에 미친 영향도를 정밀 분석합니다.',
        details: '글로벌 특성 중요도 순위(Bar plot)와 특성값 크기별 영향 분포(Beeswarm plot)를 생성합니다.',
        rec: '모델의 해석 가능성 및 XAI 리포트가 필요할 때 활성화'
    },
    'shap_model': {
        title: 'Target Model for SHAP',
        summary: 'SHAP 기여도 분석을 수행할 대상 모델을 지정합니다.',
        details: '"Champion" 선택 시 벤치마크에서 1위를 차지한 최고 성능 모델을 자동으로 분석합니다.',
        rec: '권장값: "Champion"'
    },
    'shap_max_samples': {
        title: 'SHAP Max Samples',
        summary: 'SHAP 설명자 연산에 투입할 최대 샘플 행 수입니다.',
        details: '샤플리 값 계산은 조합 연산량이 많으므로 대규모 데이터셋에서는 적절한 샘플링이 필요합니다.',
        rec: '권장값: 100 ~ 200'
    },
    // 4. Execution Runner
    'run_id': {
        title: 'Custom Run ID',
        summary: '이번 벤치마크 실행의 고유 식별자 이름입니다.',
        details: '비워둘 경우 현재 시각 기준 타임스탬프(`run_YYYYMMDD_HHMMSS`)가 자동으로 부여됩니다.',
        rec: '특정 실험 관리 시: "experiment_v1", "baseline_test" 등 입력'
    },
    'overwrite_run': {
        title: 'Overwrite Run Directory',
        summary: '동일한 Run ID 이름의 결과 폴더가 이미 존재할 때 덮어쓰기를 허용합니다.',
        details: '체크하지 않은 상태에서 중복된 Run ID가 지정되면 에러를 발생시켜 기존 결과를 보호합니다.',
        warn: '덮어쓰기 시 이전 실행의 리포트와 차트가 영구적으로 대체됩니다.'
    },
    // 5. Model Specific Hyperparameters
    'XGBoost.n_estimators': {
        title: 'XGBoost: n_estimators',
        summary: '순차적으로 생성할 결정 트리(Boosting Round)의 총 개수입니다.',
        details: '트리가 많을수록 훈련 데이터 오차를 더 줄일 수 있으나 과적합 위험이 증가합니다.',
        rec: '권장값: 100 ~ 500 (기본 100)',
        warn: '너무 높으면 과적합 및 훈련 시간 증가'
    },
    'XGBoost.learning_rate': {
        title: 'XGBoost: learning_rate',
        summary: '각 부스팅 단계에서 트리의 기여도를 축소하는 학습률(Shrinkage)입니다.',
        details: '학습률을 낮추면 모델이 더 견고해지지만 더 많은 n_estimators가 필요합니다.',
        rec: '권장값: 0.03 ~ 0.2 (기본 0.1)',
        warn: '너무 크면 최적해를 지나쳐 발산할 수 있습니다.'
    },
    'XGBoost.max_depth': {
        title: 'XGBoost: max_depth',
        summary: '개별 결정 트리의 최대 깊이(수직 레벨)입니다.',
        details: '깊이가 깊을수록 복잡한 고차원 특성 상호작용을 포착하지만 과적합에 취약해집니다.',
        rec: '권장값: 3 ~ 8 (기본 6)',
        warn: '8 이상으로 높이면 메모리 급증 및 과적합 발생'
    },
    'XGBoost.n_jobs': {
        title: 'XGBoost: n_jobs',
        summary: '트리 훈련 시 병렬로 활용할 CPU 코어 수입니다.',
        details: '-1로 설정 시 시스템의 모든 가용 CPU 스레드를 최대 활용합니다.',
        rec: '기본값: -1 (최대 병렬화)'
    },
    'CatBoost.iterations': {
        title: 'CatBoost: iterations',
        summary: 'CatBoost 부스팅 알고리즘의 최대 반복(트리 생성) 횟수입니다.',
        details: 'XGBoost의 n_estimators와 동일한 역할을 합니다.',
        rec: '권장값: 100 ~ 1000 (기본 100)'
    },
    'CatBoost.learning_rate': {
        title: 'CatBoost: learning_rate',
        summary: 'CatBoost 순서형 부스팅(Ordered Boosting)의 가중치 축소율입니다.',
        rec: '권장값: 0.03 ~ 0.2 (기본 0.1)'
    },
    'CatBoost.depth': {
        title: 'CatBoost: depth',
        summary: '대칭 결정 트리(Oblivious Trees)의 깊이입니다.',
        details: '대칭 트리 구조 덕분에 과적합 방어력이 뛰어나며 빠른 추론 속도를 보장합니다.',
        rec: '권장값: 4 ~ 8 (기본 6)'
    },
    'CatBoost.verbose': {
        title: 'CatBoost: verbose',
        summary: 'CatBoost 내부 C++ 훈련 로그의 출력 주기입니다.',
        details: '0으로 설정 시 로그 출력을 생략하여 파이프라인 콘솔을 정돈합니다.',
        rec: '기본값: 0'
    },
    'RandomForest.n_estimators': {
        title: 'RandomForest: n_estimators',
        summary: '배깅(Bagging) 앙상블을 구성하는 무작위 결정 트리의 수입니다.',
        details: '트리가 많을수록 분산(Variance)이 감소하여 예측이 안정화됩니다.',
        rec: '권장값: 100 ~ 500 (기본 100)',
        warn: '트리가 많아져도 과적합되지 않으나 메모리와 훈련 시간이 증가합니다.'
    },
    'RandomForest.max_depth': {
        title: 'RandomForest: max_depth',
        summary: '트리의 최대 분기 깊이입니다.',
        details: '"null"로 지정 시 모든 리프 노드가 순수해질 때까지 끝까지 확장합니다.',
        rec: '기본값: null (또는 과적합 방지 시 10~25)'
    },
    'RandomForest.n_jobs': {
        title: 'RandomForest: n_jobs',
        summary: '병렬 훈련 코어 수 (-1: 전체 코어).',
        rec: '기본값: -1'
    },
    'MLP.hidden_layer_sizes': {
        title: 'MLP: hidden_layer_sizes',
        summary: '다층 인공신경망(MLP)의 은닉층 레이어별 뉴런 수 구조입니다.',
        details: '[128, 64]는 1번째 은닉층 128개, 2번째 은닉층 64개 뉴런의 2계층 구조를 의미합니다.',
        rec: '권장값: [128, 64] 또는 [64, 32]',
        warn: '층이 너무 깊거나 뉴런이 많으면 소량 데이터에서 과적합될 수 있습니다.'
    },
    'MLP.activation': {
        title: 'MLP: activation',
        summary: '은닉층 뉴런에 적용할 비선형 활성화 함수입니다.',
        details: 'relu (Rectified Linear Unit), tanh (쌍곡 탄젠트), logistic (시그모이드) 등이 있습니다.',
        rec: '권장값: "relu"'
    },
    'MLP.solver': {
        title: 'MLP: solver',
        summary: '신경망 가중치 최적화 옵티마이저 알고리즘입니다.',
        details: 'adam (대용량/중용량 적응형 학습률), lbfgs (소용량 고속 수렴), sgd (확률적 경사하강법).',
        rec: '권장값: "adam"'
    },
    'MLP.max_iter': {
        title: 'MLP: max_iter',
        summary: '신경망 훈련 시 최대 반복 에포크(Epoch) 수입니다.',
        rec: '권장값: 200 ~ 1000 (기본 500)'
    },
    'TabPFN.N_ensemble_configurations': {
        title: 'TabPFN: N_ensemble_configurations',
        summary: 'TabPFN 사전학습 트랜스포머의 추론 앙상블 조합 수입니다.',
        details: '1,000행 미만의 소규모 테이블 데이터에서 1초 만에 최상급 회귀 성능을 발휘합니다.',
        rec: '권장값: 16 ~ 32 (기본 32)'
    },
    'TabICL.n_estimators': {
        title: 'TabICL: n_estimators',
        summary: 'In-Context Learning 추론을 위해 앙상블할 프롬프트 데이터셋 샘플링 수입니다.',
        details: '대규모 사전학습 기반으로 훈련 없이 문맥 내 학습(In-Context)으로 즉시 추론합니다.',
        rec: '권장값: 4 ~ 16 (기본 8)'
    },
    'TabICL.device': {
        title: 'TabICL: device',
        summary: 'TabICL 모델 연산에 사용할 하드웨어 디바이스입니다.',
        details: '"cpu", "cuda" (NVIDIA GPU), "mps" (Apple Silicon) 중 선택 가능합니다.',
        rec: '기본값: "cpu"'
    },
    'TabICL.batch_size': {
        title: 'TabICL: batch_size',
        summary: '추론 시 메모리 초과를 방지하기 위한 배치 크기입니다.',
        rec: '권장값: 2 ~ 8 (기본 4)'
    },
    'Transformer.epochs': {
        title: 'Transformer: epochs',
        summary: '자체 주의집중(Self-Attention) 회귀 모델의 총 훈련 에포크 수입니다.',
        rec: '권장값: 20 ~ 100 (기본 20)'
    },
    'Transformer.d_model': {
        title: 'Transformer: d_model',
        summary: '트랜스포머 인코더 내부 어텐션 임베딩 잠재 차원 크기입니다.',
        rec: '권장값: 16 ~ 64 (기본 32)'
    },
    'Transformer.nhead': {
        title: 'Transformer: nhead',
        summary: '멀티헤드 어텐션(Multi-Head Attention)의 독립적인 헤드 분할 수입니다.',
        details: 'd_model을 나누어 떨어지게 나누어야 합니다.',
        rec: '권장값: 2 또는 4 (기본 4)'
    },
    'Transformer.num_layers': {
        title: 'Transformer: num_layers',
        summary: '직렬로 적층할 트랜스포머 인코더 블록 레이어 수입니다.',
        rec: '권장값: 1 ~ 3 (기본 2)'
    },
    'Transformer.batch_size': {
        title: 'Transformer: batch_size',
        summary: '미니배치 확률적 경사하강법 훈련 배치 크기입니다.',
        rec: '권장값: 16 ~ 64 (기본 32)'
    }
};

// Application State
const state = {
    config: {},
    defaultConfig: {},
    datasets: [],
    currentDatasetCols: [],
    selectedView: 'view-dataset',
    runs: [],
    selectedRunId: null,
    ws: null,
    isRunning: false
};

// ==========================================
// 🏁 INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
    initTooltipSystem();
    setupNavigation();
    setupEventListeners();
    await loadInitialConfig();
    await loadDatasets();
    await loadRuns();
    setupWebSocket();
});

// ==========================================
// 💡 TOOLTIP SYSTEM ENGINE
// ==========================================
function initTooltipSystem() {
    let popover = document.getElementById('globalTooltipPopover');
    if (!popover) {
        popover = document.createElement('div');
        popover.id = 'globalTooltipPopover';
        popover.className = 'tooltip-popover';
        document.body.appendChild(popover);
    }

    // Global listener for info tips (delegation)
    document.addEventListener('click', (e) => {
        const tipBtn = e.target.closest('.info-tip');
        if (tipBtn) {
            e.stopPropagation();
            const tipKey = tipBtn.getAttribute('data-tip');
            showTooltip(tipKey, tipBtn);
        } else if (!e.target.closest('.tooltip-popover')) {
            hideTooltip();
        }
    });

    // Optional hover support
    document.addEventListener('mouseover', (e) => {
        const tipBtn = e.target.closest('.info-tip');
        if (tipBtn) {
            const tipKey = tipBtn.getAttribute('data-tip');
            showTooltip(tipKey, tipBtn);
        }
    });

    document.addEventListener('scroll', hideTooltip, true);
}

function showTooltip(docKey, targetElement) {
    const doc = HELP_DOCS[docKey];
    if (!doc) return;

    const popover = document.getElementById('globalTooltipPopover');
    popover.innerHTML = `
        <div class="tooltip-header">
            <div class="tooltip-title">💡 ${doc.title}</div>
            <div class="tooltip-close" onclick="hideTooltip()">✕</div>
        </div>
        <div class="tooltip-summary">${doc.summary}</div>
        ${doc.details ? `
            <div class="tooltip-section">
                <div class="tooltip-section-title" style="color: var(--text-dim);">상세 동작 및 원리</div>
                <div class="tooltip-details">${doc.details.replace(/\n/g, '<br>')}</div>
            </div>
        ` : ''}
        ${doc.rec ? `
            <div class="tooltip-rec">
                <span>🎯</span>
                <div><strong>권장 가이드:</strong> ${doc.rec}</div>
            </div>
        ` : ''}
        ${doc.warn ? `
            <div class="tooltip-warn">
                <span>⚠️</span>
                <div><strong>주의사항:</strong> ${doc.warn}</div>
            </div>
        ` : ''}
    `;

    popover.style.display = 'block';

    const rect = targetElement.getBoundingClientRect();
    const popRect = popover.getBoundingClientRect();

    let top = rect.bottom + window.scrollY + 8;
    let left = rect.left + window.scrollX - 20;

    if (left + popRect.width > window.innerWidth - 20) {
        left = window.innerWidth - popRect.width - 20;
    }
    if (left < 10) left = 10;

    if (top + popRect.height > window.innerHeight + window.scrollY - 10) {
        top = rect.top + window.scrollY - popRect.height - 8;
    }

    popover.style.top = `${top}px`;
    popover.style.left = `${left}px`;
}

function hideTooltip() {
    const popover = document.getElementById('globalTooltipPopover');
    if (popover) popover.style.display = 'none';
}

// ==========================================
// 🧭 NAVIGATION & VIEW SWITCHING
// ==========================================
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const viewPanes = document.querySelectorAll('.view-pane');
    const breadcrumb = document.getElementById('breadcrumbTitle');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetViewId = item.getAttribute('data-view');
            
            navItems.forEach(n => n.classList.remove('active'));
            viewPanes.forEach(v => v.classList.remove('active'));

            item.classList.add('active');
            const targetPane = document.getElementById(targetViewId);
            if (targetPane) targetPane.classList.add('active');

            state.selectedView = targetViewId;
            breadcrumb.textContent = item.querySelector('.nav-label').textContent;

            // Refresh view-specific content
            if (targetViewId === 'view-runner') {
                updateCompiledYamlPreview();
            } else if (targetViewId === 'view-results') {
                loadRuns();
            }
        });
    });

    // Top Quick Run button shortcuts
    document.getElementById('btnQuickRunTop').addEventListener('click', () => {
        document.querySelector('[data-view="view-runner"]').click();
        document.getElementById('btnStartPipelineRun').scrollIntoView({ behavior: 'smooth' });
    });
}

// ==========================================
// 📡 CONFIG LOAD & SYNC
// ==========================================
async function loadInitialConfig() {
    try {
        const res = await fetch(`${API_BASE}/api/config`);
        const data = await res.json();
        state.defaultConfig = data.default || {};
        state.config = data.active || data.default || {};
        
        populateFormFromConfig(state.config);
        updateSidebarSummary();
        updateCompiledYamlPreview();
    } catch (e) {
        showToast("Failed to load initial configuration: " + e.message, "error");
    }
}

function populateFormFromConfig(cfg) {
    // Data Ingestion
    const dataSection = cfg.data || {};
    document.getElementById('cfgDataDir').value = dataSection.data_dir || 'data';
    document.getElementById('cfgOutputDir').value = dataSection.output_dir || 'outputs';

    // Splitting
    const splitSection = dataSection.split || {};
    const splitMethod = splitSection.method || 'train_test_split';
    document.getElementById('cfgSplitMethod').value = splitMethod;
    if (splitSection.test_size) document.getElementById('splitParamValTestSize').value = splitSection.test_size;
    if (splitSection.n_splits) document.getElementById('splitParamValNFolds').value = splitSection.n_splits;
    handleSplitMethodChange(splitMethod);

    // Active Models
    const fwSection = cfg.framework || {};
    const activeModels = fwSection.active_models || ['XGBoost', 'CatBoost', 'RandomForest', 'MLP'];
    renderModelPool(cfg.models || {}, activeModels);

    // HPO
    const hpoSection = cfg.hpo || {};
    document.getElementById('cfgHpoEnabled').checked = !!hpoSection.enabled;
    document.getElementById('cfgHpoTrials').value = hpoSection.n_trials || 10;

    // SHAP
    const shapSection = cfg.shap || {};
    document.getElementById('cfgShapEnabled').checked = !!shapSection.enabled;
    document.getElementById('cfgShapModel').value = shapSection.model || 'Champion';
    document.getElementById('cfgShapMaxSamples').value = shapSection.max_samples || 100;
    handleShapModelChange(document.getElementById('cfgShapModel').value);

    // Custom YAML
    const customObj = {};
    const standardKeys = ['logging', 'framework', 'data', 'models', 'hpo', 'shap'];
    for (const [k, v] of Object.entries(cfg)) {
        if (!standardKeys.includes(k)) customObj[k] = v;
    }
    document.getElementById('customYamlTextarea').value = JSON.stringify(customObj, null, 2);
}

const DEFAULT_MODEL_TEMPLATES = {
    'XGBoost': { n_estimators: 100, learning_rate: 0.1, max_depth: 6, n_jobs: -1 },
    'CatBoost': { iterations: 100, learning_rate: 0.1, depth: 6, verbose: 0 },
    'RandomForest': { n_estimators: 100, max_depth: 'null', n_jobs: -1 },
    'MLP': { hidden_layer_sizes: '[128, 64]', activation: 'relu', solver: 'adam', max_iter: 500 },
    'TabPFN': { N_ensemble_configurations: 32 },
    'TabICL': { n_estimators: 8, device: 'cpu', batch_size: 4 },
    'Transformer': { epochs: 20, d_model: 32, nhead: 4, num_layers: 2, batch_size: 32 }
};

function renderModelPool(allModels, activeList) {
    const grid = document.getElementById('modelSelectionGrid');
    const accordion = document.getElementById('modelParamsAccordion');
    grid.innerHTML = '';
    accordion.innerHTML = '';

    const modelNames = ['XGBoost', 'CatBoost', 'RandomForest', 'MLP', 'TabPFN', 'TabICL', 'Transformer'];

    modelNames.forEach(m => {
        const isChecked = activeList.includes(m);
        
        // 1. Grid Checkbox Card
        const card = document.createElement('label');
        card.className = `model-card-checkbox ${isChecked ? 'checked' : ''}`;
        card.id = `model_card_${m}`;
        card.innerHTML = `
            <input type="checkbox" data-model="${m}" ${isChecked ? 'checked' : ''}>
            <strong>${m}</strong>
        `;
        grid.appendChild(card);

        // 2. Merge user params with default templates
        const defaultParams = DEFAULT_MODEL_TEMPLATES[m] || {};
        const userParams = allModels[m] || {};
        const params = { ...defaultParams, ...userParams };

        // 3. Accordion for params
        const accItem = document.createElement('div');
        accItem.className = `accordion-item ${isChecked ? 'active-model open' : 'inactive-model'}`;
        accItem.id = `accordion_${m}`;
        accItem.innerHTML = `
            <div class="accordion-header">
                <div class="accordion-header-left">
                    <span>🔧 <strong>${m}</strong> Hyperparameters</span>
                </div>
                <div class="accordion-header-right">
                    <span class="badge ${isChecked ? 'badge-success' : 'text-muted'}" id="acc_badge_${m}">
                        ${isChecked ? '● Active' : '○ Inactive'}
                    </span>
                    <span class="acc-arrow">▼</span>
                </div>
            </div>
            <div class="accordion-body">
                <div class="grid grid-2" id="paramsGroup_${m}"></div>
            </div>
        `;
        const accBodyGrid = accItem.querySelector(`#paramsGroup_${m}`);
        for (const [pk, pv] of Object.entries(params)) {
            const formG = document.createElement('div');
            formG.className = 'form-group';
            const displayVal = (typeof pv === 'object' && pv !== null) ? JSON.stringify(pv) : (pv === null ? 'null' : pv);
            const tipKey = `${m}.${pk}`;
            formG.innerHTML = `
                <label>${pk} <span class="info-tip" data-tip="${tipKey}">ⓘ</span></label>
                <input type="text" class="form-control model-param-input" data-model="${m}" data-param="${pk}" value="${displayVal}">
            `;
            formG.querySelector('input').addEventListener('input', updateCompiledYamlPreview);
            accBodyGrid.appendChild(formG);
        }

        // Toggle accordion on header click
        accItem.querySelector('.accordion-header').addEventListener('click', () => {
            accItem.classList.toggle('open');
        });

        // Checkbox change event handler: synchronize accordion and view
        card.querySelector('input').addEventListener('change', (e) => {
            const checked = e.target.checked;
            card.classList.toggle('checked', checked);
            
            const badge = document.getElementById(`acc_badge_${m}`);
            if (checked) {
                accItem.className = 'accordion-item active-model open';
                if (badge) {
                    badge.className = 'badge badge-success';
                    badge.textContent = '● Active';
                }
            } else {
                accItem.className = 'accordion-item inactive-model';
                if (badge) {
                    badge.className = 'badge text-muted';
                    badge.textContent = '○ Inactive';
                }
            }

            updateSidebarSummary();
            updateCompiledYamlPreview();
        });

        accordion.appendChild(accItem);
    });
}

// ==========================================
// 📂 DATASET DISCOVERY & INSPECTION
// ==========================================
async function loadDatasets() {
    const dataDir = document.getElementById('cfgDataDir').value;
    try {
        const res = await fetch(`${API_BASE}/api/datasets?data_dir=${encodeURIComponent(dataDir)}`);
        const json = await res.json();
        state.datasets = json.datasets || [];
        
        const select = document.getElementById('cfgDatasetPathSelect');
        select.innerHTML = '';
        state.datasets.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.path;
            opt.textContent = `${d.name} (${d.size_kb} KB)`;
            select.appendChild(opt);
        });

        if (state.datasets.length > 0) {
            inspectSelectedDataset(state.datasets[0].path);
        } else {
            document.getElementById('sampleTableWrapper').innerHTML = '<div class="text-muted">No dataset files found.</div>';
        }
    } catch (e) {
        showToast("Error listing datasets: " + e.message, "error");
    }
}

async function inspectSelectedDataset(filePath) {
    const wrapper = document.getElementById('sampleTableWrapper');
    wrapper.innerHTML = '<div class="loading-state">Inspecting dataset schema & preview...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/datasets/inspect?path=${encodeURIComponent(filePath)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        state.currentDatasetCols = data.columns || [];

        // Populate Target & Ignored dropdowns
        const targetSelect = document.getElementById('cfgTargetCol');
        const ignoredSelect = document.getElementById('cfgIgnoredCols');
        targetSelect.innerHTML = '';
        ignoredSelect.innerHTML = '';

        state.currentDatasetCols.forEach(col => {
            const tOpt = document.createElement('option');
            tOpt.value = col;
            tOpt.textContent = col;
            targetSelect.appendChild(tOpt);

            const iOpt = document.createElement('option');
            iOpt.value = col;
            iOpt.textContent = col;
            ignoredSelect.appendChild(iOpt);
        });

        // Set default target if found
        if (state.currentDatasetCols.includes('Target_Y')) {
            targetSelect.value = 'Target_Y';
        } else if (state.currentDatasetCols.length > 0) {
            targetSelect.value = state.currentDatasetCols[state.currentDatasetCols.length - 1];
        }

        // Render Sample Table
        renderSampleTable(data.columns, data.preview);
        updateSidebarSummary();
        updateCompiledYamlPreview();
    } catch (e) {
        wrapper.innerHTML = `<div class="text-muted" style="color: var(--danger);">Failed to load sample: ${e.message}</div>`;
    }
}

function renderSampleTable(cols, rows) {
    const wrapper = document.getElementById('sampleTableWrapper');
    if (!rows || rows.length === 0) {
        wrapper.innerHTML = '<div class="text-muted">No rows to preview.</div>';
        return;
    }

    let html = '<table class="data-table"><thead><tr>';
    cols.forEach(c => html += `<th>${c}</th>`);
    html += '</tr></thead><tbody>';

    rows.forEach(r => {
        html += '<tr>';
        cols.forEach(c => {
            html += `<td>${r[c] !== undefined ? r[c] : ''}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

// ==========================================
// 🧩 COMPILE CURRENT CONFIG
// ==========================================
function getCompiledConfig() {
    const activeModels = [];
    document.querySelectorAll('.model-card-checkbox input:checked').forEach(cb => {
        activeModels.push(cb.getAttribute('data-model'));
    });

    const splitMethod = document.getElementById('cfgSplitMethod').value;
    const splitParams = { method: splitMethod };
    if (splitMethod === 'train_test_split') {
        splitParams.test_size = parseFloat(document.getElementById('splitParamValTestSize').value) || 0.2;
    } else if (splitMethod === 'kfold') {
        splitParams.n_splits = parseInt(document.getElementById('splitParamValNFolds').value) || 5;
    }

    // Collect updated model parameters
    const modelsParams = {};
    document.querySelectorAll('.model-param-input').forEach(inp => {
        const m = inp.getAttribute('data-model');
        const p = inp.getAttribute('data-param');
        let val = inp.value.trim();
        
        if (val.startsWith('[') && val.endsWith(']')) {
            try {
                val = JSON.parse(val);
            } catch (e) {
                val = val.slice(1, -1).split(',').map(x => {
                    const num = Number(x.trim());
                    return isNaN(num) ? x.trim() : num;
                }).filter(x => x !== '');
            }
        } else if (!isNaN(val) && val !== '') {
            val = val.includes('.') ? parseFloat(val) : parseInt(val);
        } else if (val.toLowerCase() === 'true') {
            val = true;
        } else if (val.toLowerCase() === 'false') {
            val = false;
        } else if (val.toLowerCase() === 'none' || val.toLowerCase() === 'null' || val === '') {
            val = null;
        }

        if (!modelsParams[m]) modelsParams[m] = {};
        modelsParams[m][p] = val;
    });

    const targetCol = document.getElementById('cfgTargetCol').value || 'Target_Y';
    const ignoredOpts = Array.from(document.getElementById('cfgIgnoredCols').selectedOptions).map(o => o.value);

    const compiled = {
        logging: { log_dir: "logs", console_level: "INFO" },
        framework: {
            random_state: 42,
            test_size: 0.2,
            active_models: activeModels
        },
        data: {
            data_dir: document.getElementById('cfgDataDir').value,
            output_dir: document.getElementById('cfgOutputDir').value,
            target_column: targetCol,
            ignored_columns: ignoredOpts.length > 0 ? ignoredOpts : null,
            split: splitParams
        },
        hpo: {
            enabled: document.getElementById('cfgHpoEnabled').checked,
            n_trials: parseInt(document.getElementById('cfgHpoTrials').value) || 10
        },
        shap: {
            enabled: document.getElementById('cfgShapEnabled').checked,
            model: document.getElementById('cfgShapModel').value,
            max_samples: parseInt(document.getElementById('cfgShapMaxSamples').value) || 100
        },
        models: modelsParams
    };

    return compiled;
}

function updateCompiledYamlPreview() {
    const cfg = getCompiledConfig();
    const yamlStr = jsyamlDump(cfg);
    const block = document.getElementById('compiledYamlBlock');
    if (block) block.textContent = yamlStr;
}

function jsyamlDump(obj, indent = 0) {
    let result = '';
    const spaces = '  '.repeat(indent);
    for (const [key, value] of Object.entries(obj)) {
        if (value === null || value === undefined) {
            result += `${spaces}${key}: null\n`;
        } else if (typeof value === 'object' && !Array.isArray(value)) {
            result += `${spaces}${key}:\n` + jsyamlDump(value, indent + 1);
        } else if (Array.isArray(value)) {
            result += `${spaces}${key}: [${value.map(v => typeof v === 'string' ? `'${v}'` : v).join(', ')}]\n`;
        } else if (typeof value === 'string') {
            result += `${spaces}${key}: '${value}'\n`;
        } else {
            result += `${spaces}${key}: ${value}\n`;
        }
    }
    return result;
}

// ==========================================
// 🔄 SIDEBAR STATUS SYNCHRONIZATION
// ==========================================
function updateSidebarSummary() {
    const targetCol = document.getElementById('cfgTargetCol').value || 'Target_Y';
    const splitMethod = document.getElementById('cfgSplitMethod').value || 'train_test_split';
    const activeCount = document.querySelectorAll('.model-card-checkbox input:checked').length;
    const hpoEnabled = document.getElementById('cfgHpoEnabled').checked;
    const shapEnabled = document.getElementById('cfgShapEnabled').checked;
    const shapModel = document.getElementById('cfgShapModel').value;

    document.getElementById('sumTarget').textContent = targetCol;
    document.getElementById('sumSplit').textContent = splitMethod;
    document.getElementById('sumModels').textContent = activeCount;
    document.getElementById('sumHpo').textContent = hpoEnabled ? 'Enabled' : 'Disabled';
    document.getElementById('sumShap').textContent = shapEnabled ? `Enabled (${shapModel})` : 'Disabled';
    document.getElementById('activeModelsCountPill').textContent = `${activeCount} Models Active`;
}

// ==========================================
// 🚀 PIPELINE EXECUTION & TERMINAL LOGS
// ==========================================
async function startPipelineRun() {
    if (state.isRunning) return;

    const btn = document.getElementById('btnStartPipelineRun');
    const runId = document.getElementById('runIdInput').value.trim();
    const overwrite = document.getElementById('overwriteRunCheck').checked;
    const datasetPath = document.getElementById('cfgDatasetPathSelect').value;
    const compiledCfg = getCompiledConfig();

    btn.disabled = true;
    btn.textContent = '⏳ Running AutoML Benchmark...';
    setEngineStatus(true);

    const payload = {
        config: compiledCfg,
        run_id: runId || null,
        overwrite_run: overwrite,
        dataset_path: datasetPath || null,
        target_column: compiledCfg.data.target_column,
        enable_shap: compiledCfg.shap.enabled,
        shap_model: compiledCfg.shap.model
    };

    try {
        const res = await fetch(`${API_BASE}/api/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Execution trigger failed');
        }

        showToast("AutoML Benchmark Run initiated!", "success");
    } catch (e) {
        showToast(e.message, "error");
        btn.disabled = false;
        btn.textContent = '🚀 Start AutoML Benchmark Run';
        setEngineStatus(false);
    }
}

function setupWebSocket() {
    state.ws = new WebSocket(`${WS_BASE}/ws/logs`);
    const terminal = document.getElementById('terminalLogs');

    state.ws.onopen = () => {
        setInterval(() => {
            if (state.ws.readyState === WebSocket.OPEN) {
                state.ws.send('ping');
            }
        }, 10000);
    };

    state.ws.onmessage = (event) => {
        if (event.data === 'pong') return;

        const line = document.createElement('div');
        line.className = 'log-line';
        line.textContent = event.data;

        // Color highlights
        if (event.data.includes('❌') || event.data.includes('Error')) {
            line.style.color = '#ef4444';
        } else if (event.data.includes('✨') || event.data.includes('completed')) {
            line.style.color = '#10b981';
            setEngineStatus(false);
            document.getElementById('btnStartPipelineRun').disabled = false;
            document.getElementById('btnStartPipelineRun').textContent = '🚀 Start AutoML Benchmark Run';
            showToast("Pipeline Run Completed! Explore in Results & Metrics.", "success");
        } else if (event.data.includes('🚀')) {
            line.style.color = '#818cf8';
        }

        terminal.appendChild(line);
        terminal.scrollTop = terminal.scrollHeight;
    };

    state.ws.onclose = () => {
        setTimeout(setupWebSocket, 3000);
    };
}

function setEngineStatus(running) {
    state.isRunning = running;
    const badge = document.getElementById('engineStatusBadge');
    if (running) {
        badge.className = 'status-indicator running';
        badge.textContent = '● Running';
    } else {
        badge.className = 'status-indicator ready';
        badge.textContent = '● Ready';
    }
}

// ==========================================
// 📊 RESULTS & METRICS DASHBOARD
// ==========================================
async function loadRuns() {
    try {
        const res = await fetch(`${API_BASE}/api/runs`);
        const data = await res.json();
        state.runs = data.runs || [];

        const select = document.getElementById('resultsRunSelect');
        const alert = document.getElementById('noRunsAlert');
        const content = document.getElementById('resultsDashboardContent');

        select.innerHTML = '';
        if (state.runs.length === 0) {
            alert.classList.remove('hidden');
            content.classList.add('hidden');
            return;
        }

        alert.classList.add('hidden');
        content.classList.remove('hidden');

        state.runs.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.run_id;
            opt.textContent = `${r.run_id} (${r.created_at}) - Top: ${r.champion}`;
            select.appendChild(opt);
        });

        state.selectedRunId = state.runs[0].run_id;
        loadRunDetail(state.selectedRunId);
    } catch (e) {
        showToast("Failed to load results runs: " + e.message, "error");
    }
}

async function loadRunDetail(runId) {
    try {
        const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}`);
        if (!res.ok) throw new Error("Run not found");
        const data = await res.json();

        const report = data.report || {};
        const champion = report.champion || 'N/A';
        const metrics = report.metrics || {};
        const champMetrics = metrics[champion] || {};

        // 1. Champion Banner
        document.getElementById('championModelName').textContent = champion;
        document.getElementById('metricR2').textContent = typeof champMetrics.R2 === 'number' ? champMetrics.R2.toFixed(4) : (champMetrics.R2 || 'N/A');
        document.getElementById('metricRMSE').textContent = typeof champMetrics.RMSE === 'number' ? champMetrics.RMSE.toFixed(4) : (champMetrics.RMSE || 'N/A');
        document.getElementById('metricMAE').textContent = typeof champMetrics.MAE === 'number' ? champMetrics.MAE.toFixed(4) : (champMetrics.MAE || 'N/A');
        document.getElementById('metricRunId').textContent = runId;

        // 2. Leaderboard Table
        const tbody = document.getElementById('leaderboardTableBody');
        tbody.innerHTML = '';
        
        const sortedEntries = Object.entries(metrics).sort((a, b) => {
            const r2A = a[1].R2 !== undefined ? a[1].R2 : -999;
            const r2B = b[1].R2 !== undefined ? b[1].R2 : -999;
            return r2B - r2A;
        });

        sortedEntries.forEach(([mName, mObj], idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td>${mName === champion ? `🏆 <strong>${mName}</strong>` : mName}</td>
                <td>${mObj.R2 !== undefined ? mObj.R2.toFixed(4) : 'N/A'}</td>
                <td>${mObj.RMSE !== undefined ? mObj.RMSE.toFixed(4) : 'N/A'}</td>
                <td>${mObj.MAE !== undefined ? mObj.MAE.toFixed(4) : 'N/A'}</td>
            `;
            tbody.appendChild(tr);
        });

        // 3. Comparison Images
        const imgR2 = document.getElementById('imgComparisonR2');
        const imgRMSE = document.getElementById('imgComparisonRMSE');
        if (data.images['model_comparison_r2.png']) {
            imgR2.src = data.images['model_comparison_r2.png'];
        }
        if (data.images['model_comparison_rmse.png']) {
            imgRMSE.src = data.images['model_comparison_rmse.png'];
        }

        // 4. Individual Model Diagnostics
        const diagSelect = document.getElementById('diagModelSelect');
        diagSelect.innerHTML = '';
        Object.keys(metrics).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            diagSelect.appendChild(opt);
        });

        diagSelect.onchange = () => updateDiagnosticPlots(runId, diagSelect.value);
        if (Object.keys(metrics).length > 0) {
            diagSelect.value = champion;
            updateDiagnosticPlots(runId, champion);
        }

        // 5. SHAP Assets
        const shapCard = document.getElementById('shapResultCard');
        if (data.shap) {
            shapCard.classList.remove('hidden');
            const shapSummary = data.images[`${data.shap.model}_shap_summary.png`];
            const shapBar = data.images[`${data.shap.model}_shap_bar.png`];
            if (shapSummary) document.getElementById('imgShapSummary').src = shapSummary;
            if (shapBar) document.getElementById('imgShapBar').src = shapBar;
        } else {
            shapCard.classList.add('hidden');
        }

    } catch (e) {
        showToast("Error loading run details: " + e.message, "error");
    }
}

function updateDiagnosticPlots(runId, modelName) {
    document.getElementById('imgActualVsPred').src = `/outputs/${runId}/${modelName}_actual_vs_pred.png`;
    document.getElementById('imgResiduals').src = `/outputs/${runId}/${modelName}_residuals.png`;
    document.getElementById('imgLearningCurve').src = `/outputs/${runId}/${modelName}_learning_curve.png`;
}

// ==========================================
// 🛠️ UI EVENT HANDLERS & HELPERS
// ==========================================
function setupEventListeners() {
    // Dataset Path Change
    document.getElementById('cfgDatasetPathSelect').addEventListener('change', (e) => {
        inspectSelectedDataset(e.target.value);
    });

    // Target Column Change
    document.getElementById('cfgTargetCol').addEventListener('change', () => {
        updateSidebarSummary();
        updateCompiledYamlPreview();
    });

    // Split Method Change
    document.getElementById('cfgSplitMethod').addEventListener('change', (e) => {
        handleSplitMethodChange(e.target.value);
        updateSidebarSummary();
        updateCompiledYamlPreview();
    });

    // HPO & SHAP toggles
    document.getElementById('cfgHpoEnabled').addEventListener('change', () => {
        updateSidebarSummary();
        updateCompiledYamlPreview();
    });
    document.getElementById('cfgShapEnabled').addEventListener('change', () => {
        updateSidebarSummary();
        updateCompiledYamlPreview();
    });
    document.getElementById('cfgShapModel').addEventListener('change', (e) => {
        handleShapModelChange(e.target.value);
        updateSidebarSummary();
        updateCompiledYamlPreview();
    });

    // Execution Trigger
    document.getElementById('btnStartPipelineRun').addEventListener('click', startPipelineRun);

    // Save YAML Only
    document.getElementById('btnSaveConfigOnly').addEventListener('click', async () => {
        const cfg = getCompiledConfig();
        try {
            await fetch(`${API_BASE}/api/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: cfg })
            });
            showToast("Saved configs/web_config.yml successfully!", "success");
        } catch (e) {
            showToast("Failed to save config: " + e.message, "error");
        }
    });

    // Results Run Change
    document.getElementById('resultsRunSelect').addEventListener('change', (e) => {
        loadRunDetail(e.target.value);
    });

    // Clear Terminal
    document.getElementById('btnClearTerminal').addEventListener('click', () => {
        document.getElementById('terminalLogs').innerHTML = '<div class="log-line text-muted">> Logs cleared.</div>';
    });

    // Reset Default Button
    document.getElementById('btnResetConfig').addEventListener('click', () => {
        populateFormFromConfig(state.defaultConfig);
        updateSidebarSummary();
        updateCompiledYamlPreview();
        showToast("Reset to default configuration profile.", "info");
    });
}

function handleSplitMethodChange(method) {
    const testSizeG = document.getElementById('splitParamTestSize');
    const nFoldsG = document.getElementById('splitParamNFolds');
    const desc = document.getElementById('splitStrategyDesc');

    if (method === 'train_test_split') {
        testSizeG.classList.remove('hidden');
        nFoldsG.classList.add('hidden');
        desc.innerHTML = '<strong>Holdout Split</strong>: Dataset is partitioned randomly into Training (80%) and Testing (20%) subsets with fixed random seeding.';
    } else if (method === 'kfold') {
        testSizeG.classList.add('hidden');
        nFoldsG.classList.remove('hidden');
        desc.innerHTML = '<strong>K-Fold Cross Validation</strong>: Dataset is partitioned into K equal folds to minimize evaluation bias and maximize sample efficiency.';
    } else {
        testSizeG.classList.add('hidden');
        nFoldsG.classList.add('hidden');
        desc.innerHTML = '<strong>Time Series Sequential Split</strong>: Dataset is ordered temporally without lookahead leakage.';
    }
}

function handleShapModelChange(model) {
    const desc = document.getElementById('shapEngineDesc');
    if (model === 'TabICL') {
        desc.innerHTML = '⚡ <strong>TabICL Dedicated In-Context Explainer</strong>: Uses prompt dataset background sampling to compute exact feature attributions.';
    } else if (['XGBoost', 'CatBoost', 'RandomForest'].includes(model)) {
        desc.innerHTML = `🌲 <strong>TreeExplainer</strong>: Fast and exact tree traversal feature attributions for <b>${model}</b>.`;
    } else if (model === 'Champion') {
        desc.innerHTML = '🏆 <strong>Dynamic Champion Explainer</strong>: Inspects winning architecture and automatically routes to optimal explainer.';
    } else {
        desc.innerHTML = `🧠 <strong>ModelExplainer / KernelExplainer</strong>: Model-agnostic background sampling for <b>${model}</b>.`;
    }
}

function showToast(msg, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.style.borderColor = type === 'error' ? 'var(--danger)' : (type === 'success' ? 'var(--success)' : 'var(--primary)');
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3500);
}
