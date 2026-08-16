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
# 🎨 PAGE & WORKSPACE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AutoML Regression Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling that seamlessly integrates with Streamlit native layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        padding-bottom: 4px;
        margin-bottom: 1.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.45rem 0.9rem;
        border-radius: 6px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.12) !important;
        color: #818cf8 !important;
    }

    /* Modern status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-ready {
        background-color: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    
    .status-running {
        background-color: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .engine-badge {
        background-color: rgba(99, 102, 241, 0.1);
        color: #c7d2fe;
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.78rem;
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
                new_str = st.text_input(f"{k} (list)", value=val_str, key=widget_key)
                try:
                    updated[k] = ast.literal_eval(new_str)
                except Exception:
                    updated[k] = [x.strip() for x in new_str.strip("[]").split(",") if x.strip()]
            elif isinstance(v, dict):
                dict_str = json.dumps(v)
                new_dict_str = st.text_input(f"{k} (dict/JSON)", value=dict_str, key=widget_key)
                try:
                    updated[k] = json.loads(new_dict_str)
                except Exception:
                    updated[k] = v
            elif v is None:
                raw_val = st.text_input(f"{k} (optional)", value="", key=widget_key)
                if raw_val.lower() in ["none", "null", ""]:
                    updated[k] = None
                elif raw_val.lower() == "true":
                    updated[k] = True
                elif raw_val.lower() == "false":
                    updated[k] = False
                else:
                    try:
                        updated[k] = int(raw_val)
                    except ValueError:
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

# Persistent app config states
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
# 👈 SIDEBAR: CONTROL & ENGINE STATUS
# ==========================================
with st.sidebar:
    st.title("AutoML Platform")
    st.caption("Tabular Regression Benchmark Suite")
    st.markdown("---")
    
    st.markdown("##### Engine Status")
    if st.session_state.pipeline_running:
        st.markdown('<span class="status-badge status-running">● Running Benchmark</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-ready">● Ready</span>', unsafe_allow_html=True)
    
    num_active = len(st.session_state.cfg_active_models)
    hpo_str = "Enabled" if st.session_state.cfg_hpo_enabled else "Disabled"
    shap_str = f"Enabled ({st.session_state.cfg_shap_model})" if st.session_state.cfg_shap_enabled else "Disabled"
    
    st.markdown(f"""
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.7; margin-top: 0.8rem; background: rgba(30, 41, 59, 0.3); padding: 0.8rem; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.1);">
            <div>Target: <b style="color: #f1f5f9;">{st.session_state.cfg_target_col}</b></div>
            <div>Split: <b style="color: #f1f5f9;">{st.session_state.cfg_split_method}</b></div>
            <div>Active Models: <b style="color: #f1f5f9;">{num_active}</b></div>
            <div>HPO: <b style="color: #f1f5f9;">{hpo_str}</b></div>
            <div>SHAP: <b style="color: #f1f5f9;">{shap_str}</b></div>
            <div>Outputs: <code style="font-size: 0.75rem;">{st.session_state.cfg_output_dir}/&lt;run_id&gt;</code></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Reset to Default Config", use_container_width=True):
        st.session_state.cfg_data_dir = base_default.get("data", {}).get("data_dir", "data")
        st.session_state.cfg_output_dir = base_default.get("data", {}).get("output_dir", "outputs")
        st.session_state.cfg_active_models = base_default.get("framework", {}).get("active_models", list(base_default.get("models", {}).keys()))
        st.session_state.cfg_models_params = base_default.get("models", {})
        st.rerun()


# ==========================================
# 📑 MAIN WORKSPACE: NATIVE TABBED INTERFACE
# ==========================================
tab_data, tab_split, tab_models, tab_shap, tab_custom, tab_runner, tab_results = st.tabs([
    "📁 Data Ingestion",
    "✂️ Partitioning",
    "🤖 Model Pool",
    "🔍 SHAP Attribution",
    "🧩 Advanced Config",
    "⚙️ Runner Console",
    "📈 Results & Metrics"
])


# ==========================================
# 📁 TAB 1: DATA INGESTION
# ==========================================
with tab_data:
    st.subheader("Data Source & Role Mapping")
    st.caption("Select dataset source, inspect data samples, and assign feature and target column roles.")

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
        
        sample_df = preview_dataset_sample(st.session_state.cfg_dataset_path)
        if sample_df is not None:
            with st.expander("👁️ Quick Dataset Sample Preview (Top 5 rows)", expanded=True):
                st.dataframe(sample_df, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Column Role Mapping")

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
# ✂️ TAB 2: DATA PARTITIONING
# ==========================================
with tab_split:
    st.subheader("Data Partitioning & Validation Strategy")
    st.caption("Configure dataset splitting method to reliably validate model generalization.")

    split_methods = ["train_test_split", "kfold", "timeseries"]
    default_split_idx = split_methods.index(st.session_state.cfg_split_method) if st.session_state.cfg_split_method in split_methods else 0
    st.session_state.cfg_split_method = st.selectbox(
        "Split Method", 
        split_methods, 
        index=default_split_idx,
        help="train_test_split: Simple holdout split\nkfold: K-Fold cross validation\ntimeseries: Sequential time-ordered split"
    )
    
    st.markdown("##### Splitting Parameters")
    rendered_split_params = render_dynamic_params(st.session_state.cfg_split_params, "split")
    st.session_state.cfg_split_params = rendered_split_params

    st.markdown("---")
    st.markdown("##### Strategy Overview")
    if st.session_state.cfg_split_method == "train_test_split":
        test_sz = st.session_state.cfg_split_params.get("test_size", 0.2)
        st.info(f"📊 **Holdout Split**: Dataset is partitioned into Training ({100 - int(float(test_sz)*100)}%) and Testing ({int(float(test_sz)*100)}%) sets.")
    elif st.session_state.cfg_split_method == "kfold":
        n_splits = st.session_state.cfg_split_params.get("n_splits", 5)
        st.info(f"🔄 **K-Fold Cross Validation**: Data is divided into {n_splits} equal folds to minimize evaluation bias.")
    else:
        st.info("⏱️ **Time Series Split**: Data is partitioned along the temporal dimension without future lookahead.")


# ==========================================
# 🤖 TAB 3: MODEL POOL & TUNING
# ==========================================
with tab_models:
    st.subheader("Model Pool & Hyperparameter Tuning")
    st.caption("Select active regressor architectures and configure baseline hyperparameters.")

    available_models = list(base_default.get("models", {}).keys())

    st.markdown("##### Active Models")
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
    st.markdown("##### Model Hyperparameters")
    
    updated_models = {}
    for model_name in available_models:
        with st.expander(f"🔧 {model_name} Parameters", expanded=(model_name in st.session_state.cfg_active_models)):
            current_model_params = st.session_state.cfg_models_params.get(model_name, {})
            updated_params = render_dynamic_params(current_model_params, f"model_{model_name}")
            updated_models[model_name] = updated_params
    st.session_state.cfg_models_params = updated_models

    # Hyperparameter Optimization (Optuna)
    st.markdown("---")
    st.markdown("##### Hyperparameter Optimization (Optuna)")
    hpo_c1, hpo_c2 = st.columns([1, 2])
    with hpo_c1:
        st.session_state.cfg_hpo_enabled = st.checkbox("Enable HPO (Optuna)", value=st.session_state.cfg_hpo_enabled)
    with hpo_c2:
        st.session_state.cfg_hpo_trials = st.number_input("HPO Trials per Model", min_value=2, max_value=200, value=st.session_state.cfg_hpo_trials, step=1)


# ==========================================
# 🔍 TAB 4: SHAP EXPLAINABILITY
# ==========================================
with tab_shap:
    st.subheader("SHAP Model Interpretability")
    st.caption("Generate feature attribution rankings, Beeswarm distributions, and explainer dashboards.")

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
    st.markdown("##### Explainer Engine Routing")
    if chosen == "TabICL":
        st.info("⚡ **TabICL Dedicated Explainer**: Utilizes in-context prompt dataset sampling to compute exact feature attributions.")
    elif chosen in ["XGBoost", "CatBoost", "RandomForest"]:
        st.info(f"🌲 **TreeExplainer**: Fast and exact tree traversal feature attributions for **{chosen}**.")
    elif chosen == "Champion":
        st.info("🏆 **Dynamic Champion Explainer**: Automatically inspects winning model architecture and routes to TreeExplainer, TabICL Dedicated, or KernelExplainer.")
    else:
        st.info(f"🧠 **ModelExplainer / KernelExplainer**: Model-agnostic background sampling for **{chosen}**.")


# ==========================================
# 🧩 TAB 5: ADVANCED CONFIG
# ==========================================
with tab_custom:
    st.subheader("Custom YAML Profile")
    st.caption("Review or inject additional non-standard sections in YAML format.")

    custom_yaml_str = yaml.dump(st.session_state.cfg_custom_sections, default_flow_style=False, allow_unicode=True)
    edited_custom_yaml = st.text_area("Custom YAML Dictionary", value=custom_yaml_str, height=250)
    try:
        parsed_custom = yaml.safe_load(edited_custom_yaml) or {}
        st.session_state.cfg_custom_sections = parsed_custom
        st.success("Valid YAML format.")
    except Exception as e:
        st.error(f"Invalid YAML format: {e}")


# ==========================================
# ⚙️ TAB 6: RUNNER CONSOLE
# ==========================================
with tab_runner:
    st.subheader("Pipeline Execution Runner")
    st.caption("Compile your active configuration profile and launch the AutoML execution pipeline.")

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
        st.markdown("##### Compiled Config Preview")
        st.code(yaml.dump(compiled_config, default_flow_style=False, allow_unicode=True), language="yaml")
    
    with col_status:
        st.markdown("##### Execution Controls")
        
        custom_run_id = st.text_input("Custom Run ID (Optional)", value=st.session_state.custom_run_id, placeholder="e.g. run_exp_1 (Blank = auto timestamp)")
        st.session_state.custom_run_id = custom_run_id
        overwrite_choice = st.checkbox("Overwrite existing output directory (--overwrite-run)", value=False)

        if st.button("💾 Save Config to configs/web_config.yml", use_container_width=True):
            save_config(compiled_config, "configs/web_config.yml")
            st.success("Saved configs/web_config.yml successfully!")

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Start AutoML Benchmark Run", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running):
            save_config(compiled_config, "configs/web_config.yml")
            st.session_state.pipeline_running = True
            st.session_state.run_logs = ""
            
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
                    st.success("🎉 Pipeline executed successfully! Go to '📈 Results & Metrics' to inspect visual diagnostics.")
                else:
                    st.error(f"❌ Pipeline execution terminated with exit code: {process.returncode}")
            except Exception as e:
                st.error(f"Failed to start subprocess: {e}")
            finally:
                st.session_state.pipeline_running = False

    st.markdown("---")
    st.markdown("##### Live Console Stream")
    st.code(st.session_state.run_logs if st.session_state.run_logs else "No active logs. Start a benchmark run above.", language="bash")


# ==========================================
# 📈 TAB 7: RESULTS & METRICS
# ==========================================
with tab_results:
    st.subheader("Benchmark Results & Model Diagnostics")
    st.caption("Inspect performance scorecards, error distributions, loss curves, and SHAP explainability.")

    available_runs = []
    output_base = st.session_state.cfg_output_dir
    if os.path.exists(output_base):
        for entry in os.listdir(output_base):
            entry_path = os.path.join(output_base, entry)
            if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "report.json")):
                available_runs.append(entry)
                
    available_runs.sort(
        key=lambda r: os.path.getmtime(os.path.join(output_base, r)) if os.path.exists(os.path.join(output_base, r)) else 0,
        reverse=True
    )

    if not available_runs:
        st.info(f"No execution runs found in `{output_base}/`. Run an experiment in the '⚙️ Runner Console' tab first!")
    else:
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            selected_run = st.selectbox("Select Execution Run", available_runs, index=0)

        run_dir = os.path.join(output_base, selected_run)
        report_path = os.path.join(run_dir, "report.json")
        
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                
            champion_model = report_data.get("champion", "N/A")
            metrics = report_data.get("metrics", {})
            
            # Champion Metric Cards
            st.markdown(f"#### 🏆 Champion Architecture: **{champion_model}**")
            
            champ_metrics = metrics.get(champion_model, {})
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                r2_val = champ_metrics.get("R2", champ_metrics.get("r2", "N/A"))
                st.metric("R² Score (Higher is better)", f"{r2_val:.4f}" if isinstance(r2_val, float) else str(r2_val))
            with m2:
                rmse_val = champ_metrics.get("RMSE", champ_metrics.get("rmse", "N/A"))
                st.metric("RMSE (Lower is better)", f"{rmse_val:.4f}" if isinstance(rmse_val, float) else str(rmse_val))
            with m3:
                mae_val = champ_metrics.get("MAE", champ_metrics.get("mae", "N/A"))
                st.metric("MAE (Lower is better)", f"{mae_val:.4f}" if isinstance(mae_val, float) else str(mae_val))
            with m4:
                run_id_val = report_data.get("run_id", selected_run)
                st.metric("Run Identifier", str(run_id_val))

            st.markdown("---")

            # Leaderboard Table
            st.markdown("##### 📊 Full Evaluation Leaderboard")
            leaderboard_rows = []
            for model_name, model_metrics in metrics.items():
                row = {"Model": model_name}
                row.update(model_metrics)
                leaderboard_rows.append(row)
            
            if leaderboard_rows:
                df_leaderboard = pd.DataFrame(leaderboard_rows)
                if "R2" in df_leaderboard.columns:
                    df_leaderboard = df_leaderboard.sort_values(by="R2", ascending=False)
                elif "r2" in df_leaderboard.columns:
                    df_leaderboard = df_leaderboard.sort_values(by="r2", ascending=False)
                st.dataframe(df_leaderboard, use_container_width=True)

            st.markdown("---")

            # Benchmark Plots
            st.markdown("##### 📈 Model Comparison Charts")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                r2_plot = os.path.join(run_dir, "model_comparison_r2.png")
                if is_valid_image(r2_plot):
                    st.image(r2_plot, caption="R² Comparison Across Models", use_container_width=True)
            with c_p2:
                rmse_plot = os.path.join(run_dir, "model_comparison_rmse.png")
                if is_valid_image(rmse_plot):
                    st.image(rmse_plot, caption="RMSE Comparison Across Models", use_container_width=True)

            st.markdown("---")

            # Detailed Diagnostics
            st.markdown("##### 🔬 Individual Model Diagnostic Plots")
            evaluated_models = list(metrics.keys())
            if evaluated_models:
                selected_diag_model = st.selectbox("Inspect Diagnostic Curves for Model:", evaluated_models, index=0)
                
                d1, d2, d3 = st.columns(3)
                with d1:
                    act_pred = os.path.join(run_dir, f"{selected_diag_model}_actual_vs_pred.png")
                    if is_valid_image(act_pred):
                        st.image(act_pred, caption=f"{selected_diag_model} Actual vs Predicted", use_container_width=True)
                    else:
                        st.info("Actual vs Pred plot not generated.")
                with d2:
                    residuals_plot = os.path.join(run_dir, f"{selected_diag_model}_residuals.png")
                    if is_valid_image(residuals_plot):
                        st.image(residuals_plot, caption=f"{selected_diag_model} Residual Errors", use_container_width=True)
                    else:
                        st.info("Residuals plot not generated.")
                with d3:
                    lc_plot = os.path.join(run_dir, f"{selected_diag_model}_learning_curve.png")
                    if is_valid_image(lc_plot):
                        st.image(lc_plot, caption=f"{selected_diag_model} Learning Curve", use_container_width=True)
                    else:
                        st.info("Learning curve not generated for this model type.")

            st.markdown("---")

            # SHAP Explainability Reports
            st.markdown("##### 🔍 SHAP Explainability & Feature Contribution")
            shap_report_files = glob.glob(os.path.join(run_dir, "*_shap_report.json"))
            if shap_report_files:
                shap_report_path = shap_report_files[0]
                with open(shap_report_path, "r", encoding="utf-8") as sf:
                    shap_data = json.load(sf)
                
                st.write(f"**Target Model Explained:** `{shap_data.get('model', 'N/A')}` | **Samples Evaluated:** `{shap_data.get('sample_count', 'N/A')}`")
                
                sh1, sh2 = st.columns(2)
                with sh1:
                    summary_plot = os.path.join(run_dir, f"{shap_data.get('model')}_shap_summary.png")
                    if is_valid_image(summary_plot):
                        st.image(summary_plot, caption="SHAP Beeswarm Summary Plot", use_container_width=True)
                with sh2:
                    bar_plot = os.path.join(run_dir, f"{shap_data.get('model')}_shap_bar.png")
                    if is_valid_image(bar_plot):
                        st.image(bar_plot, caption="SHAP Global Feature Importance", use_container_width=True)
                
                st.markdown("##### Top Feature Attribution Rankings")
                feat_ranking = shap_data.get("feature_importance_ranking", {})
                if feat_ranking:
                    df_shap = pd.DataFrame(list(feat_ranking.items()), columns=["Feature", "Mean |SHAP Value|"])
                    st.dataframe(df_shap, use_container_width=True)
            else:
                st.info("SHAP analysis was not enabled for this run.")
