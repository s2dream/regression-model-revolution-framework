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
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "run_logs" not in st.session_state:
    st.session_state.run_logs = ""
if "current_turn" not in st.session_state:
    st.session_state.current_turn = 1


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

data_source = st.sidebar.selectbox("Data Source Mode", ["Local Directory", "Kaggle Dataset", "UCI URL / Direct Link"])

dataset_path = None
kaggle_dataset = None
url = None

if data_source == "Local Directory":
    local_files = glob.glob(os.path.join(data_dir, "*"))
    local_files = [f for f in local_files if f.endswith(('.csv', '.tsv', '.parquet', '.jsonl'))]
    if local_files:
        dataset_path = st.sidebar.selectbox("Select Local File", local_files)
    else:
        dataset_path = st.sidebar.text_input("Local File Path", value="")
        st.sidebar.info("No matching dataset files found in data directory. Enter path manually.")
elif data_source == "Kaggle Dataset":
    kaggle_dataset = st.sidebar.text_input("Kaggle Dataset (e.g. 'crawford/80-cereals')", value="")
else:
    url = st.sidebar.text_input("Direct URL to CSV/TSV/JSONL", value="")

# Dynamic Column configurations based on selected file (if exists)
columns = []
if data_source == "Local Directory" and dataset_path and os.path.exists(dataset_path):
    columns = get_dataset_columns(dataset_path)

# Target column selection
target_col_default = default_config.get("data", {}).get("target_column", "Target_Y")
if columns:
    target_column = st.sidebar.selectbox("Target Column (y)", columns, index=columns.index(target_col_default) if target_col_default in columns else 0)
else:
    target_column = st.sidebar.text_input("Target Column (y)", value=target_col_default)

# Ignored & Feature Columns
ignored_cols_default = default_config.get("data", {}).get("ignored_columns", []) or []
feature_cols_default = default_config.get("data", {}).get("feature_columns", []) or []

if columns:
    # Filter out target column from choice lists
    remaining_cols = [c for c in columns if c != target_column]
    ignored_columns = st.sidebar.multiselect("Ignored Columns (dropped first)", remaining_cols, default=[c for c in ignored_cols_default if c in remaining_cols])
    
    # Filter out ignored columns from feature choice list
    feature_choices = [c for c in remaining_cols if c not in ignored_columns]
    feature_columns = st.sidebar.multiselect("Feature Columns (X) [Leave empty to use all remaining]", feature_choices, default=[c for c in feature_cols_default if c in feature_choices])
else:
    ignored_cols_str = st.sidebar.text_input("Ignored Columns (comma separated)", value=",".join(ignored_cols_default))
    ignored_columns = [c.strip() for c in ignored_cols_str.split(",") if c.strip()]
    
    feature_cols_str = st.sidebar.text_input("Feature Columns (comma separated) [Leave empty for all]", value=",".join(feature_cols_default))
    feature_columns = [c.strip() for c in feature_cols_str.split(",") if c.strip()]

# Split Options (dynamically rendered from default_config data.split)
st.sidebar.markdown("---")
st.sidebar.markdown("### ✂️ Data Splitting Strategy")
split_config = default_config.get("data", {}).get("split", {})

split_method = st.sidebar.selectbox(
    "Split Method", 
    ["train_test_split", "kfold", "timeseries"], 
    index=["train_test_split", "kfold", "timeseries"].index(split_config.get("method", "train_test_split"))
)

split_params_to_render = {k: v for k, v in split_config.items() if k != "method"}
updated_split_params = render_dynamic_params(split_params_to_render, "split")
updated_split_params["method"] = split_method


# ==========================================
# 🎯 HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Hyperparameter Optimization")
hpo_config = default_config.get("hpo", {})
hpo_enabled = st.sidebar.checkbox("Enable HPO (Optuna)", value=hpo_config.get("enabled", False))
hpo_trials = st.sidebar.number_input("HPO Trials per Model", min_value=2, max_value=100, value=hpo_config.get("n_trials", 10), step=1)
hpo_metric = st.sidebar.selectbox(
    "HPO Optimization Metric",
    options=["RMSE", "MAE", "R2"],
    index=["RMSE", "MAE", "R2"].index(hpo_config.get("metric", "RMSE").upper() if hpo_config.get("metric") else "RMSE")
)


# ==========================================
# 📊 CENTRAL APPLICATION CONTENT: TABS
# ==========================================
tab_models, tab_custom, tab_runner, tab_results = st.tabs([
    "🛠️ Models & Active Pool", 
    "🧩 Custom Configurations",
    "⚙️ Runner Console", 
    "📈 Results & Metrics"
])

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
            is_active_default = model in default_active_models
            if st.checkbox(model, value=is_active_default, key=f"active_{model}"):
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
            "test_size": default_config.get("framework", {}).get("test_size", 0.2),
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
        "hpo": {
            "enabled": hpo_enabled,
            "n_trials": hpo_trials,
            "metric": hpo_metric
        },
        "models": updated_models_params
    }
    
    # Merge custom config sections
    for k, v in updated_custom_config.items():
        compiled_config[k] = v

    col_ctrl, col_status = st.columns([1, 1])
    with col_ctrl:
        st.markdown("##### Configuration Preview")
        st.code(yaml.dump(compiled_config, default_flow_style=False), language="yaml")
    
    with col_status:
        st.markdown("##### Status Controls")
        
        # Display current status badge
        if st.session_state.pipeline_running:
            st.markdown('Status: <span class="status-badge status-running">Running Experiment</span>', unsafe_allow_html=True)
        else:
            st.markdown('Status: <span class="status-badge status-ready">Ready</span>', unsafe_allow_html=True)
            
        st.markdown("")
        turn_index = st.number_input("Execution Turn Index", value=st.session_state.current_turn, min_value=1, step=1)
        st.session_state.current_turn = turn_index

        # Run Button
        run_btn = st.button("🚀 Execute AutoML Pipeline", disabled=st.session_state.pipeline_running)

    if run_btn:
        st.session_state.pipeline_running = True
        st.session_state.run_logs = ""
        
        # 1. Save YAML config to web_config.yml
        save_config(compiled_config, "configs/web_config.yml")
        
        st.info("Configuration saved to `configs/web_config.yml`. Initializing subprocess run...")
        
        # 2. Setup running commands
        cmd = [sys.executable, "main.py", "--config", "configs/web_config.yml", "--turn", str(turn_index)]
        if data_source == "Local Directory" and dataset_path:
            cmd += ["--dataset-path", dataset_path]
        elif data_source == "Kaggle Dataset" and kaggle_dataset:
            cmd += ["--kaggle-dataset", kaggle_dataset]
        elif data_source == "UCI URL / Direct Link" and url:
            cmd += ["--url", url]
            
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
                # Display only last 200 lines to preserve browser memory
                lines = st.session_state.run_logs.split("\n")
                log_container.code("\n".join(lines[-200:]), language="text")
                
        rc = process.poll()
        st.session_state.pipeline_running = False
        
        if rc == 0:
            st.success(f"Execution completed successfully! Check outputs in '{output_dir}'.")
            # Automatically switch to Results Tab if possible (we can rerun Streamlit)
            st.rerun()
        else:
            st.error(f"Execution failed with return code {rc}. Review log output.")

    # Always show logs if they exist
    if st.session_state.run_logs:
        st.markdown("##### Full Console Logs")
        st.code(st.session_state.run_logs, language="text")

# ------------------------------------------
# TAB 4: RESULTS & METRICS
# ------------------------------------------
with tab_results:
    st.subheader("📊 Performance Dashboards & Charts")
    
    report_filename = f"turn_{st.session_state.current_turn}_report.json"
    report_path = os.path.join(output_dir, report_filename)
    
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
            
            comp_r2_img = os.path.join(output_dir, f"turn_{st.session_state.current_turn}_model_comparison_r2.png")
            comp_rmse_img = os.path.join(output_dir, f"turn_{st.session_state.current_turn}_model_comparison_rmse.png")
            
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
                pred_vs_act_img = os.path.join(output_dir, f"turn_{st.session_state.current_turn}_{selected_model}_actual_vs_pred.png")
                residuals_img = os.path.join(output_dir, f"turn_{st.session_state.current_turn}_{selected_model}_residuals.png")
                
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
