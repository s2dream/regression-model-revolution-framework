import streamlit as st
import yaml
import os
import sys
# Cross-platform OpenMP duplicate library protection (Linux, Windows, macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import pandas as pd
import json
import ast
import subprocess
import time
from PIL import Image

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

    /* Minimize Streamlit Default Top & Bottom Padding */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 96% !important;
    }
    
    /* Sleek, Compact Top Header Bar */
    .main-title-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25), 0 0 15px rgba(99, 102, 241, 0.1);
    }
    
    .brand-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .brand-title h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    .brand-subtext {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-left: 0.4rem;
        border-left: 1px solid rgba(255, 255, 255, 0.15);
        padding-left: 0.6rem;
    }
    
    .view-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #c7d2fe;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Compact Section Header */
    .compact-section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #6366f1;
        padding: 0.45rem 0.9rem;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 0.85rem;
    }

    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }

    .section-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.025);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.1);
    }
    
    .card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Modern status indicator */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 50px;
        font-size: 0.78rem;
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

    .engine-badge {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def load_config(path="configs/default.yml"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(config_dict, path="configs/web_config.yml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

def is_valid_image(filepath):
    """Checks if a file exists, has a non-zero size, and is a valid readable image."""
    if not filepath or not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False
    try:
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
    """Dynamically renders widgets based on value types in dictionary."""
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
if "custom_run_id" not in st.session_state:
    st.session_state.custom_run_id = ""

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
if "cfg_shap_enabled" not in st.session_state:
    st.session_state.cfg_shap_enabled = base_default.get("shap", {}).get("enabled", False)
if "cfg_shap_model" not in st.session_state:
    st.session_state.cfg_shap_model = base_default.get("shap", {}).get("model", "Champion")
if "cfg_shap_max_samples" not in st.session_state:
    st.session_state.cfg_shap_max_samples = base_default.get("shap", {}).get("max_samples", 100)
if "cfg_active_models" not in st.session_state:
    st.session_state.cfg_active_models = base_default.get("framework", {}).get("active_models", list(base_default.get("models", {}).keys()))
if "cfg_models_params" not in st.session_state:
    st.session_state.cfg_models_params = base_default.get("models", {})
if "cfg_custom_sections" not in st.session_state:
    known_keys = ["logging", "framework", "data", "models", "hpo", "shap"]
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

    NAV_DATASET = "📁 Dataset Selection"
    NAV_SPLIT = "✂️ Data Splitting"
    NAV_MODELS = "🛠️ Models & Active Pool"
    NAV_SHAP = "🔍 SHAP Interpretability"
    NAV_CUSTOM = "🧩 Custom Configurations"
    NAV_RUNNER = "⚙️ Runner Console"
    NAV_RESULTS = "📈 Results & Metrics"

    menu_options = [
        NAV_DATASET,
        NAV_SPLIT,
        NAV_MODELS,
        NAV_SHAP,
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
    shap_str = f"Enabled ({st.session_state.cfg_shap_model})" if st.session_state.cfg_shap_enabled else "Disabled"
    st.markdown(f"""
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            <div>📁 <b>Target Col:</b> {st.session_state.cfg_target_col}</div>
            <div style="margin-top: 0.3rem;">✂️ <b>Split Strategy:</b> {st.session_state.cfg_split_method}</div>
            <div style="margin-top: 0.3rem;">🤖 <b>Active Models:</b> {num_active}</div>
            <div style="margin-top: 0.3rem;">🎯 <b>HPO:</b> {hpo_str}</div>
            <div style="margin-top: 0.3rem;">🔍 <b>SHAP:</b> {shap_str}</div>
            <div style="margin-top: 0.3rem;">📂 <b>Outputs Dir:</b> {st.session_state.cfg_output_dir}/&lt;run_id&gt;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 🚀 HEADER SECTION (TOP OF MAIN VIEW)
# ==========================================
status_pill = '<span class="status-badge status-running">● Running</span>' if st.session_state.pipeline_running else '<span class="status-badge status-ready">● Ready</span>'

st.markdown(f"""
<div class="main-title-container">
    <div class="brand-title">
        <span>🚀</span>
        <h1>AutoML Regression Studio</h1>
        <span class="brand-subtext">Enterprise ML Benchmark & Interpretability</span>
    </div>
    <div style="display: flex; align-items: center; gap: 0.6rem;">
        <span class="view-pill">{selected_menu}</span>
        {status_pill}
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 📁 MENU 1: DATASET SELECTION
# ==========================================
if selected_menu == NAV_DATASET:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">📁 Data Source & Path Configurations</span>
        <span class="section-desc">Select dataset source, explore samples & map column roles</span>
    </div>
    """, unsafe_allow_html=True)

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
            with st.expander("👁️ Dataset Quick Preview (Top 5 rows)", expanded=True):
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


# ==========================================
# ✂️ MENU 2: DATA SPLITTING
# ==========================================
elif selected_menu == NAV_SPLIT:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">✂️ Data Splitting & Validation Strategy</span>
        <span class="section-desc">Configure data partitioning to validate generalization</span>
    </div>
    """, unsafe_allow_html=True)

    split_methods = ["train_test_split", "kfold", "timeseries"]
    default_split_idx = split_methods.index(st.session_state.cfg_split_method) if st.session_state.cfg_split_method in split_methods else 0
    st.session_state.cfg_split_method = st.selectbox(
        "Split Method", 
        split_methods, 
        index=default_split_idx,
        help="train_test_split: Simple holdout split\nkfold: K-Fold cross validation\ntimeseries: Sequential time-ordered split"
    )
    
    st.markdown("##### ⚙️ Splitting Parameters")
    rendered_split_params = render_dynamic_params(st.session_state.cfg_split_params, "split")
    st.session_state.cfg_split_params = rendered_split_params

    # Visual explanation card for the chosen split strategy
    st.markdown("---")
    st.markdown("##### 💡 Strategy Overview")
    if st.session_state.cfg_split_method == "train_test_split":
        test_sz = st.session_state.cfg_split_params.get("test_size", 0.2)
        st.info(f"📊 **Holdout Split**: Dataset is partitioned randomly into Training ({100 - int(float(test_sz)*100)}%) and Testing ({int(float(test_sz)*100)}%) sets.")
    elif st.session_state.cfg_split_method == "kfold":
        n_splits = st.session_state.cfg_split_params.get("n_splits", 5)
        st.info(f"🔄 **K-Fold Cross Validation**: Data is divided into {n_splits} equal folds to minimize evaluation bias.")
    else:
        st.info("⏱️ **Time Series Split**: Data is partitioned along the temporal dimension without future-data lookahead.")


# ==========================================
# 🛠️ MENU 3: MODELS & ACTIVE POOL
# ==========================================
elif selected_menu == NAV_MODELS:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">🤖 Active Regression Models & Tuning</span>
        <span class="section-desc">Select models for benchmark competition & customize hyperparameters</span>
    </div>
    """, unsafe_allow_html=True)

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
        with st.expander(f"🔧 {model_name} Parameters", expanded=(model_name in st.session_state.cfg_active_models)):
            current_model_params = st.session_state.cfg_models_params.get(model_name, {})
            updated_params = render_dynamic_params(current_model_params, f"model_{model_name}")
            updated_models[model_name] = updated_params
    st.session_state.cfg_models_params = updated_models

    # Hyperparameter Optimization (Optuna)
    st.markdown("---")
    st.markdown('<div class="card-header">🎯 Hyperparameter Optimization (Optuna)</div>', unsafe_allow_html=True)
    hpo_c1, hpo_c2 = st.columns([1, 2])
    with hpo_c1:
        st.session_state.cfg_hpo_enabled = st.checkbox("Enable HPO (Optuna)", value=st.session_state.cfg_hpo_enabled)
    with hpo_c2:
        st.session_state.cfg_hpo_trials = st.number_input("HPO Trials per Model", min_value=2, max_value=200, value=st.session_state.cfg_hpo_trials, step=1)


# ==========================================
# 🔍 MENU 4: SHAP INTERPRETABILITY
# ==========================================
elif selected_menu == NAV_SHAP:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">🔍 SHAP Model Explainability</span>
        <span class="section-desc">Feature attribution, importance rankings & Beeswarm distributions</span>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.cfg_shap_enabled = st.checkbox(
        "Enable SHAP Analysis after Training", 
        value=st.session_state.cfg_shap_enabled,
        help="When enabled, SHAP feature importance and Beeswarm summary plots are automatically generated."
    )

    available_models = list(base_default.get("models", {}).keys())
    shap_model_options = ["Champion"] + available_models
    
    curr_idx = 0
    if st.session_state.cfg_shap_model in shap_model_options:
        curr_idx = shap_model_options.index(st.session_state.cfg_shap_model)

    col_m, col_s = st.columns(2)
    with col_m:
        st.session_state.cfg_shap_model = st.selectbox(
            "Target Model for SHAP Analysis", 
            shap_model_options, 
            index=curr_idx,
            help="Select 'Champion' to automatically explain the best-performing model, or pick a specific model."
        )
    with col_s:
        st.session_state.cfg_shap_max_samples = st.number_input(
            "Max Test Samples to Analyze", 
            min_value=10, 
            max_value=500, 
            value=st.session_state.cfg_shap_max_samples, 
            step=10,
            help="Limits test samples evaluated with SHAP to speed up analysis."
        )

    # Display indicator badge for explainer engine
    chosen = st.session_state.cfg_shap_model
    st.markdown("---")
    st.markdown("##### ⚡ Explainer Engine Routing Indicator")
    if chosen == "TabICL":
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem;">
            <span class="engine-badge">⚡ Engine: TabICL Dedicated In-Context Explainer</span>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #cbd5e1;">
                TabICL utilizes its specialized In-Context explainer pipeline, sampling background context from the prompt dataset to compute exact feature attributions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif chosen in ["XGBoost", "CatBoost", "RandomForest"]:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 1rem;">
            <span class="engine-badge" style="color: #6ee7b7; border-color: #10b981;">🌲 Engine: TreeExplainer</span>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #cbd5e1;">
                <b>{chosen}</b> uses fast TreeExplainer for exact and rapid tree traversal feature attributions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif chosen == "Champion":
        st.markdown("""
        <div style="background: rgba(227, 179, 65, 0.1); border: 1px solid rgba(227, 179, 65, 0.3); border-radius: 12px; padding: 1rem;">
            <span class="engine-badge" style="color: #fde047; border-color: #eab308;">🏆 Engine: Dynamic Champion Explainer</span>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #cbd5e1;">
                The framework will automatically inspect the winning Champion model and apply the optimal explainer engine (TreeExplainer, TabICL Dedicated, or KernelExplainer).
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 12px; padding: 1rem;">
            <span class="engine-badge" style="color: #cbd5e1; border-color: #94a3b8;">🧠 Engine: ModelExplainer / KernelExplainer</span>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #cbd5e1;">
                <b>{chosen}</b> utilizes model-agnostic kernel explainer sampling.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 🧩 MENU 5: CUSTOM CONFIGURATIONS
# ==========================================
elif selected_menu == NAV_CUSTOM:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">🧩 Custom Configurations</span>
        <span class="section-desc">Review or inject non-standard parameters in YAML</span>
    </div>
    """, unsafe_allow_html=True)

    custom_yaml_str = yaml.dump(st.session_state.cfg_custom_sections, default_flow_style=False, allow_unicode=True)
    edited_custom_yaml = st.text_area("Custom YAML Dictionary", value=custom_yaml_str, height=250)
    try:
        parsed_custom = yaml.safe_load(edited_custom_yaml) or {}
        st.session_state.cfg_custom_sections = parsed_custom
        st.success("Valid YAML format.")
    except Exception as e:
        st.error(f"Invalid YAML format: {e}")


# ==========================================
# ⚙️ MENU 6: RUNNER CONSOLE
# ==========================================
elif selected_menu == NAV_RUNNER:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">⚙️ Execution Runner & Console</span>
        <span class="section-desc">Compile config, trigger subprocess benchmark & stream logs live</span>
    </div>
    """, unsafe_allow_html=True)

    # Assemble complete configuration dictionary
    split_payload = {"method": st.session_state.cfg_split_method}
    split_payload.update(st.session_state.cfg_split_params)

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
        "shap": {
            "enabled": st.session_state.cfg_shap_enabled,
            "model": st.session_state.cfg_shap_model,
            "max_samples": st.session_state.cfg_shap_max_samples
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
        st.markdown("##### 🚀 Execution Controls")
        
        custom_run_id = st.text_input("Custom Run ID (Optional)", value=st.session_state.custom_run_id, placeholder="e.g. experiment_v1 (Leave blank for auto-timestamp)")
        st.session_state.custom_run_id = custom_run_id
        overwrite_choice = st.checkbox("Overwrite existing output directory if Run ID already exists (--overwrite-run)", value=False)

        # Save config button
        if st.button("💾 Save Config to configs/web_config.yml", use_container_width=True):
            save_config(compiled_config, "configs/web_config.yml")
            st.success("Saved configs/web_config.yml successfully!")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Start execution button
        if st.button("🚀 Start AutoML Benchmark Run", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running):
            save_config(compiled_config, "configs/web_config.yml")
            st.session_state.pipeline_running = True
            st.session_state.run_logs = ""
            
            # Construct CLI command arguments
            cmd = [
                sys.executable, "main.py",
                "--config", "configs/web_config.yml"
            ]
            if custom_run_id.strip():
                cmd.extend(["--run-id", custom_run_id.strip()])
            if overwrite_choice:
                cmd.append("--overwrite-run")
            
            if st.session_state.cfg_data_source == "Local Directory" and st.session_state.cfg_dataset_path:
                cmd.extend(["--dataset-path", st.session_state.cfg_dataset_path])
            elif st.session_state.cfg_data_source == "Kaggle Dataset" and st.session_state.cfg_kaggle_dataset:
                cmd.extend(["--kaggle-dataset", st.session_state.cfg_kaggle_dataset])
            elif st.session_state.cfg_data_source == "UCI URL / Direct Link" and st.session_state.cfg_url:
                cmd.extend(["--url", st.session_state.cfg_url])

            if st.session_state.cfg_target_col:
                cmd.extend(["--target", st.session_state.cfg_target_col])

            if st.session_state.cfg_shap_enabled:
                cmd.append("--enable-shap")
                cmd.extend(["--shap-model", st.session_state.cfg_shap_model])

            # Stream subprocess execution
            st.info(f"Executing: `{' '.join(cmd)}`")
            log_placeholder = st.empty()
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                logs_buffer = []
                for line in process.stdout:
                    logs_buffer.append(line)
                    log_placeholder.code("".join(logs_buffer[-25:]), language="bash")
                
                process.wait()
                st.session_state.run_logs = "".join(logs_buffer)
                
                if process.returncode == 0:
                    st.success("🎉 Pipeline executed successfully! Navigate to '📈 Results & Metrics' to explore diagnostic charts and SHAP reports.")
                else:
                    st.error(f"❌ Pipeline execution terminated with exit code: {process.returncode}")
            except Exception as e:
                st.error(f"Failed to start subprocess: {e}")
            finally:
                st.session_state.pipeline_running = False

    st.markdown("---")
    st.markdown("##### 📜 Full Process Log Stream")
    st.code(st.session_state.run_logs if st.session_state.run_logs else "No logs yet. Run an experiment above to view stdout/stderr stream.", language="bash")


# ==========================================
# 📈 MENU 7: RESULTS & METRICS
# ==========================================
elif selected_menu == NAV_RESULTS:
    st.markdown("""
    <div class="compact-section-header">
        <span class="section-title">📈 Performance Scorecard & Visual Diagnostics</span>
        <span class="section-desc">Leaderboard, error distributions, loss curves & SHAP interpretability</span>
    </div>
    """, unsafe_allow_html=True)

    # Scan available run directories inside outputs/
    available_runs = []
    output_base = st.session_state.cfg_output_dir
    if os.path.exists(output_base):
        for entry in os.listdir(output_base):
            entry_path = os.path.join(output_base, entry)
            if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "report.json")):
                available_runs.append(entry)
                
    # Sort runs newest first by folder creation/modification time
    available_runs.sort(
        key=lambda r: os.path.getmtime(os.path.join(output_base, r)) if os.path.exists(os.path.join(output_base, r)) else 0,
        reverse=True
    )

    if not available_runs:
        st.info(f"No execution runs found under `{output_base}/`. Run an experiment in the '⚙️ Runner Console' menu first to generate results!")
    else:
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            selected_run = st.selectbox("Select Execution Run", available_runs, index=0)

        run_dir = os.path.join(output_base, selected_run)
        report_path = os.path.join(run_dir, "report.json")
        
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                
            metrics_dict = report_data.get("metrics", {})
            champion_model = report_data.get("champion_model", report_data.get("best_model", "N/A"))
            
            if metrics_dict:
                # 1. Champion Highlight Card
                champ_r2 = metrics_dict.get(champion_model, {}).get("R2", 0.0)
                champ_rmse = metrics_dict.get(champion_model, {}).get("RMSE", 0.0)
                champ_mae = metrics_dict.get(champion_model, {}).get("MAE", 0.0)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 0.9rem; text-transform: uppercase; color: #e3b341; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.4rem;">🏆 Winning Champion Model (Run: {selected_run})</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; color: #ffffff;">{champion_model}</div>
                    <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                        <div><span style="color: #94a3b8; font-size: 0.85rem;">R² SCORE:</span> <b style="color: #58a6ff; font-size: 1.2rem;">{champ_r2:.4f}</b></div>
                        <div><span style="color: #94a3b8; font-size: 0.85rem;">RMSE:</span> <b style="color: #cbd5e1; font-size: 1.2rem;">{champ_rmse:.4f}</b></div>
                        <div><span style="color: #94a3b8; font-size: 0.85rem;">MAE:</span> <b style="color: #aff5b4; font-size: 1.2rem;">{champ_mae:.4f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. Performance Summary Table
                st.markdown("##### 📊 Model Leaderboard")
                df_metrics = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={"index": "Model Name"})
                if "R2" in df_metrics.columns:
                    df_metrics = df_metrics.sort_values(by="R2", ascending=False)
                st.dataframe(df_metrics.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}), use_container_width=True)
                
                # 3. Model Comparison Charts
                st.markdown("---")
                st.markdown("##### 📈 Model Comparison Charts")
                col_c1, col_c2 = st.columns(2)
                r2_img = os.path.join(run_dir, "model_comparison_r2.png")
                rmse_img = os.path.join(run_dir, "model_comparison_rmse.png")
                
                with col_c1:
                    if is_valid_image(r2_img):
                        st.image(r2_img, caption=f"R2 Comparison ({selected_run})", use_container_width=True)
                    else:
                        st.info("R2 Comparison chart not found.")
                with col_c2:
                    if is_valid_image(rmse_img):
                        st.image(rmse_img, caption=f"RMSE Comparison ({selected_run})", use_container_width=True)
                    else:
                        st.info("RMSE Comparison chart not found.")

                # 4. Diagnostic Plots
                st.markdown("---")
                st.markdown("##### 🔍 Model Diagnostics & Residual Analysis")
                selected_model = st.selectbox("Select Model for Diagnostic Plots", list(metrics_dict.keys()))
                
                if selected_model:
                    col_diag1, col_diag2 = st.columns(2)
                    pred_vs_act_img = os.path.join(run_dir, f"{selected_model}_actual_vs_pred.png")
                    residuals_img = os.path.join(run_dir, f"{selected_model}_residuals.png")
                    
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

                    # If learning curve exists, render it below
                    learning_curves = report_data.get("learning_curves", {})
                    if selected_model in learning_curves:
                        curve_img = learning_curves[selected_model]
                        if is_valid_image(curve_img):
                            st.markdown("###### 📈 Loss / Learning Curve")
                            st.image(curve_img, caption=f"{selected_model}: Loss Curve", use_container_width=True)

                # ==========================================
                # 🔍 SHAP FEATURE ATTRIBUTION SECTION
                # ==========================================
                st.markdown("---")
                st.markdown('<div class="card-header">🔍 SHAP Feature Attribution & Model Interpretation</div>', unsafe_allow_html=True)
                
                # Find any SHAP report for this run
                shap_report_files = glob.glob(os.path.join(run_dir, "*_shap_report.json"))
                if shap_report_files:
                    for shap_file in shap_report_files:
                        try:
                            with open(shap_file, "r", encoding="utf-8") as f_s:
                                s_data = json.load(f_s)
                            
                            s_model = s_data.get("model_name", "Unknown")
                            s_engine = s_data.get("explainer_engine", "Default Explainer")
                            s_samples = s_data.get("num_samples_analyzed", 0)
                            s_importance = s_data.get("mean_abs_shap", {})
                            
                            st.markdown(f"""
                            <div class="premium-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                                    <h4 style="margin: 0; color: #818cf8;">Model: <b>{s_model}</b></h4>
                                    <span class="engine-badge">⚡ Engine: {s_engine}</span>
                                </div>
                                <p style="font-size: 0.9rem; color: #94a3b8; margin: 0;">Evaluated Samples: <b>{s_samples}</b> | Top Features: <b>{', '.join(s_data.get('top_features', []))}</b></p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Render SHAP bar and summary plots
                            col_s1, col_s2 = st.columns(2)
                            s_bar_img = os.path.join(run_dir, f"{s_model}_shap_bar.png")
                            s_summary_img = os.path.join(run_dir, f"{s_model}_shap_summary.png")
                            
                            with col_s1:
                                if is_valid_image(s_bar_img):
                                    st.image(s_bar_img, caption=f"{s_model}: SHAP Feature Importance", use_container_width=True)
                                else:
                                    st.info("SHAP Bar plot not available.")
                            with col_s2:
                                if is_valid_image(s_summary_img):
                                    st.image(s_summary_img, caption=f"{s_model}: SHAP Beeswarm Summary", use_container_width=True)
                                else:
                                    st.info("SHAP Summary plot not available.")

                            # Show feature importance table
                            if s_importance:
                                with st.expander(f"📋 View Full SHAP Importance Scores ({s_model})", expanded=False):
                                    df_shap = pd.DataFrame(list(s_importance.items()), columns=["Feature", "Mean Absolute SHAP"])
                                    st.dataframe(df_shap, use_container_width=True)
                                    
                                    with open(shap_file, "r", encoding="utf-8") as f_dl:
                                        st.download_button(
                                            label=f"📥 Download {s_model} SHAP JSON Report",
                                            data=f_dl.read(),
                                            file_name=os.path.basename(shap_file),
                                            mime="application/json"
                                        )
                        except Exception as e:
                            st.warning(f"Error loading SHAP report file `{shap_file}`: {e}")
                else:
                    st.info(f"No SHAP explanation report generated for Run `{selected_run}`. Enable SHAP Analysis in '🔍 SHAP Interpretability' and re-run.")
            else:
                st.warning("No metrics data found in report JSON.")
        else:
            st.info(f"No report found at `{report_path}`.")
