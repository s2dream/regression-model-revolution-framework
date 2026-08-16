/**
 * =====================================================================
 * 🚀 AutoML Regression Studio - Frontend Application Engine
 * =====================================================================
 */

const API_BASE = window.location.origin;
const WS_BASE = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host;

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
    setupNavigation();
    setupEventListeners();
    await loadInitialConfig();
    await loadDatasets();
    await loadRuns();
    setupWebSocket();
});

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
            formG.innerHTML = `
                <label>${pk}</label>
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
