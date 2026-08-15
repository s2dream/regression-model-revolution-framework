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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Header Banner */
    .main-title-container {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #7c3aed 100%);
        padding: 2.2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        color: white;
    }
    
    .main-title-container h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .main-title-container p {
        font-size: 1.1rem;
        opacity: 0.92;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.12);
    }
    
    .card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
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
        border: 1px solid rgba(16, 185, 129, 0.3);
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
    
    /* Sidebar Navigation Enhancements */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        transition: all 0.2s ease;
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
        st.warning(f"Failed to parse columns from dataset: {e}")
    return []

def preview_dataset_sample(file_path, nrows=5):
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, nrows=nrows)
        elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
            return pd.read_csv(file_path, sep='\t', nrows=nrows)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path).head(nrows)
    except Exception:
        return None
    return None

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
base_default = load_config("configs/default.yml")

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "run_logs" not in st.session_state:
    st.session_state.run_logs = ""
if "current_turn" not in st.session_state:
    st.session_state.current_turn = 1

# Persistent app config states across navigation switches
if "cfg_data_dir" not in st.session_state:
    st.session_state.cfg_data_dir = base_default.get("data", {}).get("data_dir", "data")
if "cfg_output_dir" not in st.session_state:
    st.session_state.cfg_output_dir = base_default.get("data", {}).get("output_dir", "outputs")
if "cfg_data_source" not in st.session_state:
    st.session_state.cfg_data_source = "Local Directory"
if "cfg_dataset_path" not in st.session_state:
    st.session_state.cfg_dataset_path = ""
if "cfg_kaggle_dataset" not in st.session_state:
    st.session_state.cfg_kaggle_dataset = ""
if "cfg_url" not in st.session_state:
    st.session_state.cfg_url = ""
if "cfg_target_col" not in st.session_state:
    st.session_state.cfg_target_col = base_default.get("data", {}).get("target_column", "Target_Y")
if "cfg_ignored_cols" not in st.session_state:
    st.session_state.cfg_ignored_cols = base_default.get("data", {}).get("ignored_columns", []) or []
if "cfg_feature_cols" not in st.session_state:
    st.session_state.cfg_feature_cols = base_default.get("data", {}).get("feature_columns", []) or []
if "cfg_split_method" not in st.session_state:
    st.session_state.cfg_split_method = base_default.get("data", {}).get("split", {}).get("method", "train_test_split")
if "cfg_split_params" not in st.session_state:
    st.session_state.cfg_split_params = {k: v for k, v in base_default.get("data", {}).get("split", {}).items() if k != "method"}
if "cfg_hpo_enabled" not in st.session_state:
    st.session_state.cfg_hpo_enabled = base_default.get("hpo", {}).get("enabled", False)
if "cfg_hpo_trials" not in st.session_state:
    st.session_state.cfg_hpo_trials = base_default.get("hpo", {}).get("n_trials", 10)
if "cfg_active_models" not in st.session_state:
    st.session_state.cfg_active_models = base_default.get("framework", {}).get("active_models", list(base_default.get("models", {}).keys()))
if "cfg_models_params" not in st.session_state:
    st.session_state.cfg_models_params = base_default.get("models", {})
if "cfg_custom_sections" not in st.session_state:
    known_keys = ["logging", "framework", "data", "models", "hpo"]
    st.session_state.cfg_custom_sections = {k: v for k, v in base_default.items() if k not in known_keys}


# ==========================================
# 👈 LEFT SIDEBAR: MENU NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; text-align: center;">
        <h2 style="font-family: 'Outfit', sans-serif; margin: 0; color: #818cf8; font-weight: 800;">
            🚀 Studio Menu
        </h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0.3rem 0 1rem 0;">AutoML Regression Framework</p>
    </div>
    """, unsafe_allow_html=True)

    NAV_DATASET = "📁 Dataset & Splitting"
    NAV_MODELS = "🛠️ Models & Active Pool"
    NAV_CUSTOM = "🧩 Custom Configurations"
    NAV_RUNNER = "⚙️ Runner Console"
    NAV_RESULTS = "📈 Results & Metrics"

    menu_options = [
        NAV_DATASET,
        NAV_MODELS,
        NAV_CUSTOM,
        NAV_RUNNER,
        NAV_RESULTS
    ]

    selected_menu = st.radio(
        "Navigation",
        options=menu_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Sidebar quick status card
    st.markdown("""
    <div style="background: rgba(255,255,255,0.04); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; margin-bottom: 0.5rem;">Pipeline Status</div>
    """, unsafe_allow_html=True)
    if st.session_state.pipeline_running:
        st.markdown('<span class="status-badge status-running">● Running Experiment</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-ready">● Ready</span>', unsafe_allow_html=True)
    
    num_active = len(st.session_state.cfg_active_models)
    hpo_str = "Enabled" if st.session_state.cfg_hpo_enabled else "Disabled"
    st.markdown(f"""
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            <div>🤖 <b>Active Models:</b> {num_active}</div>
            <div style="margin-top: 0.3rem;">🎯 <b>HPO:</b> {hpo_str}</div>
            <div style="margin-top: 0.3rem;">🔄 <b>Turn:</b> {st.session_state.current_turn}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 🚀 HEADER SECTION (TOP OF MAIN VIEW)
# ==========================================
st.markdown(f"""
<div class="main-title-container">
    <h1>🚀 AutoML Regression Studio</h1>
    <p>Current View: <b>{selected_menu}</b> — Configure, optimize, execute and analyze regression models seamlessly.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 📁 MENU 1: DATASET & SPLITTING
# ==========================================
if selected_menu == NAV_DATASET:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📁 Data Source & Path Configurations</div>', unsafe_allow_html=True)
    st.write("Configure where your dataset is loaded from, specify output artifacts destination, and configure column roles.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.cfg_data_dir = st.text_input("Data Directory", value=st.session_state.cfg_data_dir)
    with col2:
        st.session_state.cfg_output_dir = st.text_input("Output Directory", value=st.session_state.cfg_output_dir)

    source_options = ["Local Directory", "Kaggle Dataset", "UCI URL / Direct Link"]
    st.session_state.cfg_data_source = st.selectbox(
        "Data Source Mode", 
        source_options, 
        index=source_options.index(st.session_state.cfg_data_source) if st.session_state.cfg_data_source in source_options else 0
    )

    if st.session_state.cfg_data_source == "Local Directory":
        local_files = glob.glob(os.path.join(st.session_state.cfg_data_dir, "*"))
        local_files = [f for f in local_files if f.endswith(('.csv', '.tsv', '.parquet', '.jsonl'))]
        if local_files:
            default_file_idx = 0
            if st.session_state.cfg_dataset_path in local_files:
                default_file_idx = local_files.index(st.session_state.cfg_dataset_path)
            st.session_state.cfg_dataset_path = st.selectbox("Select Local File", local_files, index=default_file_idx)
        else:
            st.session_state.cfg_dataset_path = st.text_input("Local File Path", value=st.session_state.cfg_dataset_path)
            st.info("No matching dataset files (.csv, .tsv, .parquet, .jsonl) found in data directory. You can enter path manually.")
    elif st.session_state.cfg_data_source == "Kaggle Dataset":
        st.session_state.cfg_kaggle_dataset = st.text_input("Kaggle Dataset Identifier (e.g. 'crawford/80-cereals')", value=st.session_state.cfg_kaggle_dataset)
    else:
        st.session_state.cfg_url = st.text_input("Direct URL to CSV/TSV/JSONL", value=st.session_state.cfg_url)

    # Dynamic Column configurations based on selected file
    columns = []
    if st.session_state.cfg_data_source == "Local Directory" and st.session_state.cfg_dataset_path and os.path.exists(st.session_state.cfg_dataset_path):
        columns = get_dataset_columns(st.session_state.cfg_dataset_path)
        
        # Dataset Sample Preview
        sample_df = preview_dataset_sample(st.session_state.cfg_dataset_path)
        if sample_df is not None:
            with st.expander("👁️ Dataset Quick Preview (Top 5 rows)", expanded=False):
                st.dataframe(sample_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="card-header">🏷️ Column Mapping & Roles</div>', unsafe_allow_html=True)

    col_target, col_cols = st.columns([1, 2])
    
    with col_target:
        if columns:
            target_default_idx = columns.index(st.session_state.cfg_target_col) if st.session_state.cfg_target_col in columns else 0
            st.session_state.cfg_target_col = st.selectbox("Target Column (y)", columns, index=target_default_idx)
        else:
            st.session_state.cfg_target_col = st.text_input("Target Column (y)", value=st.session_state.cfg_target_col)

    with col_cols:
        if columns:
            remaining_cols = [c for c in columns if c != st.session_state.cfg_target_col]
            valid_ignored = [c for c in st.session_state.cfg_ignored_cols if c in remaining_cols]
            st.session_state.cfg_ignored_cols = st.multiselect("Ignored Columns (dropped first)", remaining_cols, default=valid_ignored)
            
            feature_choices = [c for c in remaining_cols if c not in st.session_state.cfg_ignored_cols]
            valid_features = [c for c in st.session_state.cfg_feature_cols if c in feature_choices]
            st.session_state.cfg_feature_cols = st.multiselect("Feature Columns (X) [Empty = Use All Remaining]", feature_choices, default=valid_features)
        else:
            ignored_str = st.text_input("Ignored Columns (comma separated)", value=",".join(st.session_state.cfg_ignored_cols))
            st.session_state.cfg_ignored_cols = [c.strip() for c in ignored_str.split(",") if c.strip()]
            
            feature_str = st.text_input("Feature Columns (comma separated) [Empty = All]", value=",".join(st.session_state.cfg_feature_cols))
            st.session_state.cfg_feature_cols = [c.strip() for c in feature_str.split(",") if c.strip()]

    # Data Splitting Strategy
    st.markdown("---")
    st.markdown('<div class="card-header">✂️ Data Splitting Strategy</div>', unsafe_allow_html=True)
    
    split_methods = ["train_test_split", "kfold", "timeseries"]
    default_split_idx = split_methods.index(st.session_state.cfg_split_method) if st.session_state.cfg_split_method in split_methods else 0
    st.session_state.cfg_split_method = st.selectbox("Split Method", split_methods, index=default_split_idx)
    
    rendered_split_params = render_dynamic_params(st.session_state.cfg_split_params, "split")
    st.session_state.cfg_split_params = rendered_split_params

    # Hyperparameter Optimization (Optuna)
    st.markdown("---")
    st.markdown('<div class="card-header">🎯 Hyperparameter Optimization (Optuna)</div>', unsafe_allow_html=True)
    hpo_c1, hpo_c2 = st.columns([1, 2])
    with hpo_c1:
        st.session_state.cfg_hpo_enabled = st.checkbox("Enable HPO (Optuna)", value=st.session_state.cfg_hpo_enabled)
    with hpo_c2:
        st.session_state.cfg_hpo_trials = st.number_input("HPO Trials per Model", min_value=2, max_value=200, value=st.session_state.cfg_hpo_trials, step=1)


# ==========================================
# 🛠️ MENU 2: MODELS & ACTIVE POOL
# ==========================================
elif selected_menu == NAV_MODELS:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🤖 Configure Regression Model Pool</div>', unsafe_allow_html=True)
    st.write("Select the models you want to include in the active competition pool, and customize individual model hyperparameters below.")
    st.markdown('</div>', unsafe_allow_html=True)

    available_models = list(base_default.get("models", {}).keys())

    # Active model checkboxes
    st.markdown("##### ⚡ Active Model Selection")
    active_cols = st.columns(len(available_models) if available_models else 1)
    new_active_models = []
    for i, model in enumerate(available_models):
        col = active_cols[i % len(active_cols)]
        with col:
            is_active_default = model in st.session_state.cfg_active_models
            if st.checkbox(model, value=is_active_default, key=f"active_model_chk_{model}"):
                new_active_models.append(model)
    st.session_state.cfg_active_models = new_active_models

    st.markdown("---")
    st.markdown("##### ⚙️ Model Hyperparameters & Architectures")
    
    updated_models = {}
    for model_name in available_models:
        model_params = st.session_state.cfg_models_params.get(model_name, base_default.get("models", {}).get(model_name, {}))
        is_model_active = model_name in st.session_state.cfg_active_models
        
        with st.expander(f"{'🟢' if is_model_active else '⚪'} {model_name} Parameters", expanded=is_model_active):
            if not is_model_active:
                st.warning(f"'{model_name}' is currently inactive. Activate it above if you want it included in the AutoML pool.")
            updated_models[model_name] = render_dynamic_params(model_params, f"model_param_{model_name}")
            
    st.session_state.cfg_models_params = updated_models


# ==========================================
# 🧩 MENU 3: CUSTOM CONFIGURATIONS
# ==========================================
elif selected_menu == NAV_CUSTOM:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🧩 Custom & Extended Configurations</div>', unsafe_allow_html=True)
    st.write("Any extra top-level keys defined in `configs/default.yml` (e.g. logging, preprocessing, evaluation metrics) are automatically parsed and rendered below.")
    st.markdown('</div>', unsafe_allow_html=True)

    updated_custom = {}
    if st.session_state.cfg_custom_sections:
        for section_name, section_val in st.session_state.cfg_custom_sections.items():
            st.markdown(f"##### Section: `{section_name}`")
            if isinstance(section_val, dict):
                updated_custom[section_name] = render_dynamic_params(section_val, f"custom_sec_{section_name}")
            else:
                updated_custom[section_name] = st.text_input(section_name, value=str(section_val), key=f"custom_sec_{section_name}")
        st.session_state.cfg_custom_sections = updated_custom
    else:
        st.info("No custom/extra top-level keys found in `configs/default.yml`. You can add sections like `preprocessing:` or `evaluation_metrics:` to your default YAML config file, and they will render here dynamically.")


# ==========================================
# ⚙️ MENU 4: RUNNER CONSOLE
# ==========================================
elif selected_menu == NAV_RUNNER:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⚙️ Experiment Execution Console</div>', unsafe_allow_html=True)
    st.write("Review the compiled configuration, select your experiment turn index, and launch the AutoML regression training pipeline.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Compile YAML Configuration
    split_payload = dict(st.session_state.cfg_split_params)
    split_payload["method"] = st.session_state.cfg_split_method

    compiled_config = {
        "logging": base_default.get("logging", {
            "log_dir": "logs",
            "console_level": "INFO"
        }),
        "framework": {
            "random_state": base_default.get("framework", {}).get("random_state", 42),
            "test_size": base_default.get("framework", {}).get("test_size", 0.2),
            "active_models": st.session_state.cfg_active_models
        },
        "data": {
            "data_dir": st.session_state.cfg_data_dir,
            "output_dir": st.session_state.cfg_output_dir,
            "target_column": st.session_state.cfg_target_col,
            "feature_columns": st.session_state.cfg_feature_cols if st.session_state.cfg_feature_cols else None,
            "ignored_columns": st.session_state.cfg_ignored_cols if st.session_state.cfg_ignored_cols else None,
            "split": split_payload
        },
        "hpo": {
            "enabled": st.session_state.cfg_hpo_enabled,
            "n_trials": st.session_state.cfg_hpo_trials
        },
        "models": st.session_state.cfg_models_params
    }
    
    # Merge custom config sections
    for k, v in st.session_state.cfg_custom_sections.items():
        compiled_config[k] = v

    col_ctrl, col_status = st.columns([1.1, 0.9])
    
    with col_ctrl:
        st.markdown("##### 📄 Compiled YAML Config Preview")
        st.code(yaml.dump(compiled_config, default_flow_style=False, allow_unicode=True), language="yaml")
    
    with col_status:
        st.markdown("##### 🎛️ Execution Controls")
        
        # Display current status badge
        if st.session_state.pipeline_running:
            st.markdown('Current Status: <span class="status-badge status-running">Running Experiment</span>', unsafe_allow_html=True)
        else:
            st.markdown('Current Status: <span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
            
        st.markdown("")
        turn_index = st.number_input("Execution Turn Index", value=st.session_state.current_turn, min_value=1, step=1)
        st.session_state.current_turn = turn_index

        st.markdown("")
        run_btn = st.button("🚀 Execute AutoML Pipeline", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running)

    if run_btn:
        st.session_state.pipeline_running = True
        st.session_state.run_logs = ""
        
        # 1. Save YAML config to web_config.yml
        save_config(compiled_config, "configs/web_config.yml")
        
        st.info("Configuration saved to `configs/web_config.yml`. Initializing subprocess execution...")
        
        # 2. Setup running command
        cmd = [sys.executable, "main.py", "--config", "configs/web_config.yml", "--turn", str(turn_index)]
        if st.session_state.cfg_data_source == "Local Directory" and st.session_state.cfg_dataset_path:
            cmd += ["--dataset-path", st.session_state.cfg_dataset_path]
        elif st.session_state.cfg_data_source == "Kaggle Dataset" and st.session_state.cfg_kaggle_dataset:
            cmd += ["--kaggle-dataset", st.session_state.cfg_kaggle_dataset]
        elif st.session_state.cfg_data_source == "UCI URL / Direct Link" and st.session_state.cfg_url:
            cmd += ["--url", st.session_state.cfg_url]
            
        st.write(f"Executing: `{' '.join(cmd)}`")
        
        # Execute subprocess and stream stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        log_container = st.empty()
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                st.session_state.run_logs += line
                lines = st.session_state.run_logs.split("\n")
                log_container.code("\n".join(lines[-200:]), language="text")
                
        rc = process.poll()
        st.session_state.pipeline_running = False
        
        if rc == 0:
            st.success(f"🎉 Pipeline Execution Turn {turn_index} completed successfully! Check outputs in '{st.session_state.cfg_output_dir}'.")
            st.rerun()
        else:
            st.error(f"Execution failed with return code {rc}. Review the console logs below.")

    # Always show logs if they exist
    if st.session_state.run_logs:
        st.markdown("---")
        st.markdown("##### 📜 Live Console Stream Output")
        st.code(st.session_state.run_logs, language="text")


# ==========================================
# 📈 MENU 5: RESULTS & METRICS
# ==========================================
elif selected_menu == NAV_RESULTS:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📈 Performance Dashboards & Visual Diagnostics</div>', unsafe_allow_html=True)
    st.write("Inspect champion model rankings, comparative regression metrics (R2, RMSE, MAE), benchmark charts, and residual diagnostics.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Turn selector
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        selected_turn = st.number_input("Inspect Turn", value=st.session_state.current_turn, min_value=1, step=1)
        st.session_state.current_turn = selected_turn

    report_filename = f"turn_{selected_turn}_report.json"
    report_path = os.path.join(st.session_state.cfg_output_dir, report_filename)
    
    if os.path.exists(report_path):
        # Load Report JSON
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        best_model = report_data.get("best_model", "N/A")
        metrics_dict = report_data.get("metrics", {})
        
        # Display summary cards
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.2, 1, 1.8])
        with c1:
            st.metric("🏆 Champion Model", best_model)
        with c2:
            if best_model in metrics_dict:
                st.metric("🎯 Best R2 Score", f"{metrics_dict[best_model].get('R2', 0.0):.4f}")
        with c3:
            st.markdown("##### 📥 Export Reports")
            btn_col1, btn_col2 = st.columns(2)
            html_report_file = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_report.html")
            md_summary_file = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_summary.md")
            
            with btn_col1:
                if os.path.exists(html_report_file):
                    with open(html_report_file, "r", encoding="utf-8") as f_html:
                        st.download_button(
                            label="🌐 HTML Report",
                            data=f_html.read(),
                            file_name=f"turn_{selected_turn}_report.html",
                            mime="text/html",
                            use_container_width=True
                        )
            with btn_col2:
                if os.path.exists(md_summary_file):
                    with open(md_summary_file, "r", encoding="utf-8") as f_md:
                        st.download_button(
                            label="📝 Markdown",
                            data=f_md.read(),
                            file_name=f"turn_{selected_turn}_summary.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Convert metrics to DataFrame and display
        if metrics_dict:
            df_metrics = pd.DataFrame(metrics_dict).T
            st.markdown("##### 📊 Model Metrics Comparison Table")
            st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['R2'], color='#0f3d24').highlight_min(axis=0, subset=['RMSE', 'MAE'], color='#0f3d24'), use_container_width=True)
            
            # Model Comparison Charts
            st.markdown("---")
            st.markdown("##### 📈 Benchmark Comparison Visualizations")
            col_chart1, col_chart2 = st.columns(2)
            
            comp_r2_img = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_model_comparison_r2.png")
            comp_rmse_img = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_model_comparison_rmse.png")
            
            with col_chart1:
                if is_valid_image(comp_r2_img):
                    st.image(comp_r2_img, caption="R2 Comparison Chart", use_container_width=True)
                else:
                    st.info("R2 Comparison Chart not found or not generated for this turn.")
            with col_chart2:
                if is_valid_image(comp_rmse_img):
                    st.image(comp_rmse_img, caption="RMSE Comparison Chart", use_container_width=True)
                else:
                    st.info("RMSE Comparison Chart not found or not generated for this turn.")
                    
            # Individual Model Diagnostic Charts
            st.markdown("---")
            st.markdown("##### 🔍 Model Diagnostics & Residual Analysis")
            selected_model = st.selectbox("Select Model for Diagnostic Plots", list(metrics_dict.keys()))
            
            if selected_model:
                col_diag1, col_diag2 = st.columns(2)
                pred_vs_act_img = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_{selected_model}_actual_vs_pred.png")
                residuals_img = os.path.join(st.session_state.cfg_output_dir, f"turn_{selected_turn}_{selected_model}_residuals.png")
                
                with col_diag1:
                    if is_valid_image(pred_vs_act_img):
                        st.image(pred_vs_act_img, caption=f"{selected_model}: Actual vs Predicted", use_container_width=True)
                    else:
                        st.info(f"Actual vs Predicted plot not found for {selected_model}.")
                with col_diag2:
                    if is_valid_image(residuals_img):
                        st.image(residuals_img, caption=f"{selected_model}: Residuals Plot", use_container_width=True)
                    else:
                        st.info(f"Residuals plot not found for {selected_model}.")
        else:
            st.warning("No metrics data found in report JSON.")
    else:
        st.info(f"No execution report found for Turn {selected_turn} at `{report_path}`. Run an experiment in the '⚙️ Runner Console' menu first to generate results!")
