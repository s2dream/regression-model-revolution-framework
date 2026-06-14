import streamlit as st
import yaml
import os
import sys
import glob
import pandas as pd
import json
import ast
import subprocess
import time

# ==========================================
# 🎨 PREMIUM AESTHETIC CONFIGURATIONS
# ==========================================
st.set_page_config(
    page_title="AutoML Regression Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Neon accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title-container {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
        color: white;
    }
    
    .main-title-container h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .main-title-container p {
        font-size: 1.15rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Modern status indicator */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-ready {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(10, 185, 129, 0.3);
    }
    
    .status-running {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# ⚙️ HELPER FUNCTIONS
# ==========================================
def load_config(path="configs/default.yml"):
    if not os.path.exists(path):
        st.error(f"Configuration file not found at: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(config_data, path="configs/web_config.yml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)

def is_valid_image(filepath):
    if not filepath or not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_dataset_columns(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    try:
        if file_path.endswith('.csv'):
            return list(pd.read_csv(file_path, nrows=1).columns)
        elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
            return list(pd.read_csv(file_path, sep='\t', nrows=1).columns)
        elif file_path.endswith('.parquet'):
            return list(pd.read_parquet(file_path).columns)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return list(json.loads(line).keys())
    except Exception as e:
        st.sidebar.warning(f"Failed to parse columns from dataset: {e}")
    return []

def render_dynamic_params(params_dict, key_prefix):
    """
    Dynamically renders widgets based on type of values in dictionary.
    Ensures easy extensibility when new keys/params are added to default.yml.
    """
    updated = {}
    if not params_dict:
        return updated
        
    cols = st.columns(2)
    for i, (k, v) in enumerate(params_dict.items()):
        col = cols[i % 2]
        with col:
            widget_key = f"{key_prefix}_{k}"
            if isinstance(v, bool):
                updated[k] = st.checkbox(k, value=v, key=widget_key)
            elif isinstance(v, int):
                updated[k] = st.number_input(k, value=v, step=1, key=widget_key)
            elif isinstance(v, float):
                updated[k] = st.number_input(k, value=v, format="%.4f", step=0.01, key=widget_key)
            elif isinstance(v, list):
                val_str = str(v)
                res_str = st.text_input(f"{k} (list, e.g. [128, 64])", value=val_str, key=widget_key)
                try:
                    updated[k] = ast.literal_eval(res_str)
                except Exception:
                    updated[k] = v
            elif v is None:
                # Variable toggle for Null values
                col1, col2 = st.columns([1, 2])
                with col1:
                    is_null = st.checkbox("Null", value=True, key=f"{widget_key}_null_chk")
                with col2:
                    if is_null:
                        st.text_input(k, value="null (disabled)", disabled=True, key=f"{widget_key}_null_val")
                        updated[k] = None
                    else:
                        raw_val = st.text_input(k, value="", key=f"{widget_key}_null_val")
                        if raw_val.isdigit():
                            updated[k] = int(raw_val)
                        else:
                            try:
                                updated[k] = float(raw_val)
                            except ValueError:
                                updated[k] = raw_val if raw_val else None
            else:
                updated[k] = st.text_input(k, value=str(v), key=widget_key)
    return updated


# ==========================================
# 🏁 INITIAL STATE & BASE CONFIG LOADING
# ==========================================
default_config = load_config("configs/default.yml")
available_models = list(default_config.get("models", {}).keys())

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "run_logs" not in st.session_state:
    st.session_state.run_logs = ""
if "current_turn" not in st.session_state:
    st.session_state.current_turn = 1

# Initialize Planner variables in session state if not present
if "target_column" not in st.session_state:
    st.session_state.target_column = default_config.get("data", {}).get("target_column", "Target_Y")
if "ignored_columns" not in st.session_state:
    st.session_state.ignored_columns = default_config.get("data", {}).get("ignored_columns", []) or []
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = default_config.get("data", {}).get("feature_columns", []) or []
if "split_method" not in st.session_state:
    st.session_state.split_method = default_config.get("data", {}).get("split", {}).get("method", "train_test_split")
if "test_size" not in st.session_state:
    st.session_state.test_size = default_config.get("data", {}).get("split", {}).get("test_size", 0.2)
if "n_splits" not in st.session_state:
    st.session_state.n_splits = default_config.get("data", {}).get("split", {}).get("n_splits", 5)

for model in available_models:
    session_key = f"active_model_chk_{model}"
    if session_key not in st.session_state:
        st.session_state[session_key] = model in default_config.get("framework", {}).get("active_models", [])

if "batch_jobs" not in st.session_state:
    st.session_state.batch_jobs = []
if "batch_selected_dataset" not in st.session_state:
    st.session_state.batch_selected_dataset = ""
if "run_mode" not in st.session_state:
    st.session_state.run_mode = "Single Dataset Mode"


# ==========================================
# 🚀 HEADER SECTION
# ==========================================
st.markdown("""
<div class="main-title-container">
    <h1>🚀 AutoML Regression Studio</h1>
    <p>Configure data preprocessing, model selection, and hyperparameters dynamically using default.yml as schema.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 👈 SIDEBAR: DATA & SPLIT CONFIGURATIONS
# ==========================================
st.sidebar.markdown("### 📁 Dataset & Splitting")

# Scan local data directory
data_dir = st.sidebar.text_input("Data Directory", value=default_config.get("data", {}).get("data_dir", "data"))
output_dir = st.sidebar.text_input("Output Directory", value=default_config.get("data", {}).get("output_dir", "outputs"))

local_files = glob.glob(os.path.join(data_dir, "*"))
local_files = [f for f in local_files if f.endswith(('.csv', '.tsv', '.parquet', '.jsonl'))]
if local_files:
    dataset_path = st.sidebar.selectbox("Select Local File", local_files)
else:
    dataset_path = st.sidebar.text_input("Local File Path", value="")
    st.sidebar.info("No matching dataset files found in data directory. Enter path manually.")

# Dynamic Column configurations based on selected file (if exists)
columns = []
if dataset_path and os.path.exists(dataset_path):
    columns = get_dataset_columns(dataset_path)

# Target column selection
target_col_current = st.session_state.target_column
if columns:
    target_column = st.sidebar.selectbox("Target Column (y)", columns, index=columns.index(target_col_current) if target_col_current in columns else 0, key="target_column_select")
    st.session_state.target_column = target_column
else:
    target_column = st.sidebar.text_input("Target Column (y)", value=target_col_current, key="target_column_input")
    st.session_state.target_column = target_column

# Ignored & Feature Columns
ignored_cols_current = st.session_state.ignored_columns
feature_cols_current = st.session_state.feature_columns

if columns:
    # Filter out target column from choice lists
    remaining_cols = [c for c in columns if c != target_column]
    ignored_columns = st.sidebar.multiselect("Ignored Columns (dropped first)", remaining_cols, default=[c for c in ignored_cols_current if c in remaining_cols], key="ignored_columns_select")
    st.session_state.ignored_columns = ignored_columns
    
    # Filter out ignored columns from feature choice list
    feature_choices = [c for c in remaining_cols if c not in ignored_columns]
    feature_columns = st.sidebar.multiselect("Feature Columns (X) [Leave empty to use all remaining]", feature_choices, default=[c for c in feature_cols_current if c in feature_choices], key="feature_columns_select")
    st.session_state.feature_columns = feature_columns
else:
    ignored_cols_str = st.sidebar.text_input("Ignored Columns (comma separated)", value=",".join(ignored_cols_current), key="ignored_columns_input")
    ignored_columns = [c.strip() for c in ignored_cols_str.split(",") if c.strip()]
    st.session_state.ignored_columns = ignored_columns
    
    feature_cols_str = st.sidebar.text_input("Feature Columns (comma separated) [Leave empty for all]", value=",".join(feature_cols_current), key="feature_columns_input")
    feature_columns = [c.strip() for c in feature_cols_str.split(",") if c.strip()]
    st.session_state.feature_columns = feature_columns

# Split Options (dynamically rendered from default_config data.split)
st.sidebar.markdown("---")
st.sidebar.markdown("### ✂️ Data Splitting Strategy")

current_method = st.session_state.split_method
split_methods_list = ["train_test_split", "kfold", "timeseries"]
split_method = st.sidebar.selectbox(
    "Split Method", 
    split_methods_list, 
    index=split_methods_list.index(current_method) if current_method in split_methods_list else 0,
    key="split_method_select"
)
st.session_state.split_method = split_method

split_params_to_render = {
    "test_size": st.session_state.test_size,
    "n_splits": st.session_state.n_splits,
    "val_size": default_config.get("data", {}).get("split", {}).get("val_size", None),
    "shuffle": default_config.get("data", {}).get("split", {}).get("shuffle", True)
}

st.sidebar.markdown("##### Split Parameters")
updated_split_params = render_dynamic_params(split_params_to_render, "split")
updated_split_params["method"] = split_method

if "test_size" in updated_split_params:
    st.session_state.test_size = updated_split_params["test_size"]
if "n_splits" in updated_split_params:
    st.session_state.n_splits = updated_split_params["n_splits"]


# ==========================================
# 📊 CENTRAL APPLICATION CONTENT: TABS
# ==========================================
tab_planner, tab_models, tab_custom, tab_runner, tab_results = st.tabs([
    "📋 Data & Experiment Planner",
    "🛠️ Models & Active Pool", 
    "🧩 Custom Configurations",
    "⚙️ Runner Console", 
    "📈 Results & Metrics"
])

# ------------------------------------------
# TAB 0: DATA & EXPERIMENT PLANNER
# ------------------------------------------
# ------------------------------------------
# TAB 0: DATA & EXPERIMENT PLANNER
# ------------------------------------------
with tab_planner:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("📋 Multi-Dataset Batch Planner")
    st.write("지정된 데이터 디렉토리에서 여러 데이터 파일들을 감지하여 일괄 실험(Batch)을 구성합니다. 각 데이터셋의 실험 포함 여부 및 예측 대상(Target) 변수를 개별 설정할 수 있습니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # scan execution section
    col_scan1, col_scan2 = st.columns([1, 2])
    with col_scan1:
        scan_clicked = st.button("🔍 Scan Data Directory", use_container_width=True, type="primary")
    with col_scan2:
        st.write(f"스캔 대상 디렉토리: `{data_dir}` (출력 경로: `{output_dir}`)")

    # Perform scanning logic
    if scan_clicked or (not st.session_state.batch_jobs and os.path.exists(data_dir)):
        try:
            from automl_framework.planner import DataPlanner
            planner = DataPlanner()
            scanned = planner.scan_directory(data_dir)
            
            # Map scanned items to session state, preserving user modifications if existing
            new_jobs = []
            for item in scanned:
                existing = next((j for j in st.session_state.batch_jobs if j["file_path"] == item["file_path"]), None)
                if existing:
                    new_jobs.append({
                        "file_path": item["file_path"],
                        "file_name": item["file_name"],
                        "file_size": item["file_size"],
                        "columns": item["columns"],
                        "target_column": existing["target_column"],
                        "include": existing["include"]
                    })
                else:
                    new_jobs.append({
                        "file_path": item["file_path"],
                        "file_name": item["file_name"],
                        "file_size": item["file_size"],
                        "columns": item["columns"],
                        "target_column": item["target_column"],
                        "include": True
                    })
            st.session_state.batch_jobs = new_jobs
            if scan_clicked:
                st.success(f"데이터 디렉토리에서 총 {len(new_jobs)}개의 데이터셋을 찾았습니다!")
        except Exception as e:
            st.error(f"디렉토리 스캔 오류: {e}")

    # Render configurations
    if st.session_state.batch_jobs:
        st.markdown("---")
        st.markdown("##### 📋 배치 실험 스캔 결과 및 개별 타겟 설정")
        
        # Table Header
        h_inc, h_name, h_size, h_target = st.columns([1, 3, 2, 4])
        h_inc.markdown("**Include**")
        h_name.markdown("**File Name**")
        h_size.markdown("**File Size**")
        h_target.markdown("**Target Column (y)**")
        
        # Table Content
        for idx, job in enumerate(st.session_state.batch_jobs):
            c_inc, c_name, c_size, c_target = st.columns([1, 3, 2, 4])
            with c_inc:
                new_inc = st.checkbox("", value=job["include"], key=f"batch_inc_{idx}", label_visibility="collapsed")
                st.session_state.batch_jobs[idx]["include"] = new_inc
            with c_name:
                st.markdown(f"`{job['file_name']}`")
            with c_size:
                st.write(job["file_size"])
            with c_target:
                cols_list = job["columns"]
                if cols_list:
                    # In case target_column is not in the list, fallback index to 0
                    try:
                        default_idx = cols_list.index(job["target_column"])
                    except ValueError:
                        default_idx = 0
                    new_target = st.selectbox(
                        "", 
                        cols_list, 
                        index=default_idx, 
                        key=f"batch_target_{idx}", 
                        label_visibility="collapsed"
                    )
                    st.session_state.batch_jobs[idx]["target_column"] = new_target
                else:
                    new_target = st.text_input(
                        "", 
                        value=job["target_column"], 
                        key=f"batch_target_text_{idx}", 
                        label_visibility="collapsed"
                    )
                    st.session_state.batch_jobs[idx]["target_column"] = new_target
                    
        # Help warning if nothing is selected
        active_count = sum(1 for j in st.session_state.batch_jobs if j["include"])
        if active_count == 0:
            st.warning("⚠️ 선택된 실험 대상 데이터셋이 없습니다. 최소 1개 이상의 데이터셋을 선택해주세요.")
    else:
        st.info("데이터 디렉토리에 유효한 데이터 파일(.csv, .tsv, .parquet, .jsonl)이 없습니다. 파일을 배치해두거나 경로를 확인해주세요.")

# ------------------------------------------
# TAB 1: MODELS & ACTIVE POOL
# ------------------------------------------
with tab_models:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🤖 Configure Model Pool")
    st.write("Enable models to include them in the AutoML experiment pool, and modify their hyperparameters below.")
    st.markdown('</div>', unsafe_allow_html=True)

    default_active_models = default_config.get("framework", {}).get("active_models", [])
    available_models = list(default_config.get("models", {}).keys())

    # Active model checkboxes
    st.markdown("##### Select Active Models")
    active_cols = st.columns(len(available_models) if available_models else 1)
    active_models = []
    for i, model in enumerate(available_models):
        col = active_cols[i % len(active_cols)]
        with col:
            session_key = f"active_model_chk_{model}"
            is_checked = st.checkbox(model, value=st.session_state[session_key], key=session_key)
            if is_checked:
                active_models.append(model)

    st.markdown("---")

    # Hyperparameters per model (Dynamic rendering)
    updated_models_params = {}
    for model_name in available_models:
        model_params = default_config.get("models", {}).get(model_name, {})
        with st.expander(f"⚙️ {model_name} Hyperparameters", expanded=(model_name in active_models)):
            if model_name not in active_models:
                st.warning("This model is currently inactive. Check the box above to activate it.")
            updated_models_params[model_name] = render_dynamic_params(model_params, f"model_{model_name}")

# ------------------------------------------
# TAB 2: CUSTOM CONFIGURATIONS (Future Extensiblity)
# ------------------------------------------
with tab_custom:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🧩 Custom & Extra configurations")
    st.write("Any extra top-level keys in default.yml that aren't parsed by data or models sections are displayed here. You can easily add sections to default.yml and they will automatically show up below.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Find custom/extra config blocks
    known_keys = ["logging", "framework", "data", "models"]
    custom_keys = {k: v for k, v in default_config.items() if k not in known_keys}
    
    updated_custom_config = {}
    if custom_keys:
        for section_name, section_val in custom_keys.items():
            st.markdown(f"##### Section: `{section_name}`")
            if isinstance(section_val, dict):
                updated_custom_config[section_name] = render_dynamic_params(section_val, f"custom_{section_name}")
            else:
                updated_custom_config[section_name] = st.text_input(section_name, value=str(section_val), key=f"custom_{section_name}")
    else:
        st.info("No custom/extra top-level keys found in default.yml. You can add sections like 'preprocessing:' or 'evaluation_metrics:' to the default config file, and they will render here dynamically.")

# ------------------------------------------
# TAB 3: RUNNER CONSOLE
# ------------------------------------------
with tab_runner:
    st.subheader("⚙️ Experiment Execution Console")
    
    # Compile YAML Configuration
    compiled_config = {
        "logging": default_config.get("logging", {
            "log_dir": "logs",
            "console_level": "INFO"
        }),
        "framework": {
            "random_state": default_config.get("framework", {}).get("random_state", 42),
            "test_size": updated_split_params.get("test_size", default_config.get("framework", {}).get("test_size", 0.2)),
            "active_models": active_models
        },
        "data": {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "target_column": target_column,
            "feature_columns": feature_columns if feature_columns else None,
            "ignored_columns": ignored_columns if ignored_columns else None,
            "split": updated_split_params
        },
        "models": updated_models_params
    }
    
    # Merge custom config sections
    for k, v in updated_custom_config.items():
        compiled_config[k] = v

    col_ctrl, col_status = st.columns([1, 1])
    with col_ctrl:
        st.markdown("##### Configuration Preview")
        # If batch mode is active, display which datasets are scheduled to run
        if st.session_state.run_mode == "Batch Experiments Mode" and st.session_state.batch_jobs:
            active_jobs = [j["file_name"] for j in st.session_state.batch_jobs if j["include"]]
            st.warning(f"⚠️ **배치 모드 활성화됨**: 다음 {len(active_jobs)}개 데이터셋에 대해 독립적인 실험이 순차 진행됩니다:")
            for j in active_jobs:
                st.write(f"- `{j}`")
        st.code(yaml.dump(compiled_config, default_flow_style=False), language="yaml")
    
    with col_status:
        st.markdown("##### Status Controls")
        
        # Display current status badge
        if st.session_state.pipeline_running:
            st.markdown('Status: <span class="status-badge status-running">Running Experiment</span>', unsafe_allow_html=True)
        else:
            st.markdown('Status: <span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
            
        st.markdown("")
        
        # Select Execution Mode if batch_jobs are configured
        active_batch_count = sum(1 for j in st.session_state.batch_jobs if j["include"])
        if active_batch_count > 0:
            run_mode_select = st.radio(
                "Execution Mode", 
                ["Single Dataset Mode", "Batch Experiments Mode"], 
                index=0 if st.session_state.run_mode == "Single Dataset Mode" else 1,
                key="run_mode_radio"
            )
            st.session_state.run_mode = run_mode_select
        else:
            st.session_state.run_mode = "Single Dataset Mode"
            
        st.markdown("")
        turn_index = st.number_input("Execution Turn Index", value=st.session_state.current_turn, min_value=1, step=1)
        st.session_state.current_turn = turn_index

        # Run Button
        run_btn = st.button("🚀 Execute AutoML Pipeline", disabled=st.session_state.pipeline_running)

    if run_btn:
        st.session_state.pipeline_running = True
        st.session_state.run_logs = ""
        
        # Determine jobs to execute
        if st.session_state.run_mode == "Batch Experiments Mode" and st.session_state.batch_jobs:
            jobs_to_run = [j for j in st.session_state.batch_jobs if j["include"]]
        else:
            # Single dataset mode
            jobs_to_run = [{
                "file_path": dataset_path,
                "file_name": os.path.basename(dataset_path) if dataset_path else "",
                "target_column": target_column
            }]

        if not jobs_to_run or (st.session_state.run_mode == "Single Dataset Mode" and not dataset_path):
            st.error("실행할 대상 데이터셋이 존재하지 않습니다.")
            st.session_state.pipeline_running = False
        else:
            total_jobs = len(jobs_to_run)
            success_count = 0
            
            for idx, job in enumerate(jobs_to_run):
                job_file = job["file_path"]
                job_name = job["file_name"]
                job_target = job["target_column"]
                job_name_clean = os.path.splitext(job_name)[0]
                
                # Suffix output directory per dataset to avoid overwriting
                if st.session_state.run_mode == "Batch Experiments Mode":
                    job_output_dir = os.path.join(output_dir, job_name_clean)
                else:
                    job_output_dir = output_dir
                
                # 1. Update compiled_config with dataset-specific output directory and target column
                compiled_config["data"]["output_dir"] = job_output_dir
                compiled_config["data"]["target_column"] = job_target
                compiled_config["data"]["dataset_path"] = job_file
                
                # Update framework.test_size
                compiled_config["framework"]["test_size"] = updated_split_params.get("test_size", default_config.get("framework", {}).get("test_size", 0.2))
                
                save_config(compiled_config, "configs/web_config.yml")
                
                log_header = f"\n========================================\n🚀 [{idx+1}/{total_jobs}] Running dataset: {job_name}\nTarget column: {job_target}\nOutput directory: {job_output_dir}\n========================================\n"
                st.session_state.run_logs += log_header
                
                # Stream log
                log_container = st.empty()
                log_container.code("\n".join(st.session_state.run_logs.split("\n")[-200:]), language="text")
                
                cmd = [sys.executable, "main.py", "--config", "configs/web_config.yml", "--turn", str(turn_index), "--dataset-path", job_file, "--target", job_target]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        st.session_state.run_logs += line
                        # Display only last 200 lines to preserve browser memory
                        log_container.code("\n".join(st.session_state.run_logs.split("\n")[-200:]), language="text")
                        
                rc = process.poll()
                if rc == 0:
                    success_count += 1
                    st.session_state.run_logs += f"\n✅ Successfully finished: {job_name}\n"
                    if st.session_state.run_mode == "Batch Experiments Mode":
                        st.session_state.batch_selected_dataset = job_name
                else:
                    st.session_state.run_logs += f"\n❌ Failed (Return code {rc}): {job_name}\n"
            
            st.session_state.pipeline_running = False
            
            if success_count == total_jobs:
                st.success(f"All {total_jobs} experiments completed successfully!")
                st.rerun()
            else:
                st.warning(f"Batch completed: {success_count}/{total_jobs} succeeded, {total_jobs - success_count} failed. Check console logs.")

    # Always show logs if they exist
    if st.session_state.run_logs:
        st.markdown("##### Full Console Logs")
        st.code(st.session_state.run_logs, language="text")

# ------------------------------------------
# TAB 4: RESULTS & METRICS
# ------------------------------------------
with tab_results:
    st.subheader("📊 Performance Dashboards & Charts")
    
    # 🔍 Scan output directory for any subdirectories (indicating different datasets in Batch mode)
    sub_dirs = []
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                sub_dirs.append(item)
    sub_dirs.sort()
    
    # Selection of output folder (Allows switching between single mode and different batch datasets)
    if sub_dirs:
        # Determine default selected directory based on last run dataset
        default_idx = 0
        if st.session_state.batch_selected_dataset:
            clean_name = os.path.splitext(st.session_state.batch_selected_dataset)[0]
            if clean_name in sub_dirs:
                default_idx = sub_dirs.index(clean_name) + 1
                
        selected_target = st.selectbox(
            "Select Experiment Dataset to View",
            ["Single Mode (Root Output)"] + sub_dirs,
            index=default_idx
        )
        
        if selected_target == "Single Mode (Root Output)":
            current_output_dir = output_dir
        else:
            current_output_dir = os.path.join(output_dir, selected_target)
    else:
        current_output_dir = output_dir
        
    report_filename = f"turn_{st.session_state.current_turn}_report.json"
    report_path = os.path.join(current_output_dir, report_filename)
    
    if os.path.exists(report_path):
        # Load Report JSON
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        best_model = report_data.get("best_model", "N/A")
        metrics_dict = report_data.get("metrics", {})
        
        # Display summary cards
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🏆 Champion Model", best_model)
        with c2:
            if best_model in metrics_dict:
                st.metric("🎯 Best R2 Score", f"{metrics_dict[best_model].get('R2', 0.0):.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Convert metrics to DataFrame and display
        if metrics_dict:
            df_metrics = pd.DataFrame(metrics_dict).T
            st.markdown("##### Model Metrics Comparison Table")
            st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['R2'], color='#0f3d24').highlight_min(axis=0, subset=['RMSE', 'MAE'], color='#0f3d24'))
            
            # Model Comparison Charts
            st.markdown("---")
            st.markdown("##### 📈 Benchmark Comparisons")
            col_chart1, col_chart2 = st.columns(2)
            
            comp_r2_img = os.path.join(current_output_dir, f"turn_{st.session_state.current_turn}_model_comparison_r2.png")
            comp_rmse_img = os.path.join(current_output_dir, f"turn_{st.session_state.current_turn}_model_comparison_rmse.png")
            
            with col_chart1:
                if is_valid_image(comp_r2_img):
                    st.image(comp_r2_img, caption="R2 Comparison Chart")
                else:
                    st.info("R2 Comparison Chart not found or corrupted.")
            with col_chart2:
                if is_valid_image(comp_rmse_img):
                    st.image(comp_rmse_img, caption="RMSE Comparison Chart")
                else:
                    st.info("RMSE Comparison Chart not found or corrupted.")
                    
            # Individual Model Diagnostic Charts
            st.markdown("---")
            st.markdown("##### 🔍 Model Diagnostic Plots")
            selected_model = st.selectbox("Select Model for Diagnostic Plots", list(metrics_dict.keys()))
            
            if selected_model:
                col_diag1, col_diag2 = st.columns(2)
                pred_vs_act_img = os.path.join(current_output_dir, f"turn_{st.session_state.current_turn}_{selected_model}_actual_vs_pred.png")
                residuals_img = os.path.join(current_output_dir, f"turn_{st.session_state.current_turn}_{selected_model}_residuals.png")
                
                with col_diag1:
                    if is_valid_image(pred_vs_act_img):
                        st.image(pred_vs_act_img, caption=f"{selected_model}: Actual vs Predicted")
                    else:
                        st.info("Diagnostic plot not found or corrupted.")
                with col_diag2:
                    if is_valid_image(residuals_img):
                        st.image(residuals_img, caption=f"{selected_model}: Residuals Plot")
                    else:
                        st.info("Residuals plot not found or corrupted.")
        else:
            st.warning("No metrics data found in report JSON.")
    else:
        st.info(f"No execution report found for Turn {st.session_state.current_turn} at `{report_path}`. Run an experiment in the 'Runner Console' tab first to generate results!")
