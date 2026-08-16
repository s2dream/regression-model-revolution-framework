import os
import sys

# Cross-platform OpenMP duplicate library protection (Linux, Windows, macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import json
import time
import yaml
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="AutoML Regression Studio API",
    description="Enterprise API Server for AutoML Tabular Regression Framework",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)


# ==========================================
# 🔄 PIPELINE RUNNER & STATE MANAGEMENT
# ==========================================
class RunnerState:
    def __init__(self):
        self.is_running: bool = False
        self.current_process: Optional[subprocess.Popen] = None
        self.log_history: List[str] = []
        self.active_websockets: List[WebSocket] = []
        self.last_run_id: Optional[str] = None
        self.last_exit_code: Optional[int] = None

    async def broadcast_log(self, line: str):
        self.log_history.append(line)
        dead_sockets = []
        for ws in self.active_websockets:
            try:
                await ws.send_text(line)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            if dead in self.active_websockets:
                self.active_websockets.remove(dead)

runner_state = RunnerState()


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(data: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ==========================================
# 📡 REST API ENDPOINTS
# ==========================================
@app.get("/api/config")
async def get_config():
    """Returns default and web config definitions."""
    default_cfg = load_yaml(os.path.join(CONFIGS_DIR, "default.yml"))
    web_cfg_path = os.path.join(CONFIGS_DIR, "web_config.yml")
    active_cfg = load_yaml(web_cfg_path) if os.path.exists(web_cfg_path) else default_cfg
    return {
        "default": default_cfg,
        "active": active_cfg
    }

class ConfigPayload(BaseModel):
    config: Dict[str, Any]

@app.post("/api/config")
async def save_config_endpoint(payload: ConfigPayload):
    """Saves compiled configuration to configs/web_config.yml."""
    web_cfg_path = os.path.join(CONFIGS_DIR, "web_config.yml")
    save_yaml(payload.config, web_cfg_path)
    return {"status": "success", "message": "Saved configs/web_config.yml successfully."}


@app.get("/api/datasets")
async def list_datasets(data_dir: Optional[str] = None):
    """Lists available tabular dataset files in data directory."""
    target_dir = data_dir if data_dir and os.path.exists(data_dir) else DATA_DIR
    files = []
    for ext in ["*.csv", "*.tsv", "*.parquet", "*.jsonl"]:
        files.extend(glob.glob(os.path.join(target_dir, ext)))
    
    file_items = []
    for f in sorted(files):
        file_items.append({
            "name": os.path.basename(f),
            "path": f,
            "size_kb": round(os.path.getsize(f) / 1024, 2)
        })
    return {"datasets": file_items}


@app.get("/api/datasets/inspect")
async def inspect_dataset(path: str = Query(..., description="Absolute or relative path to dataset")):
    """Parses column names and returns top 5 preview rows."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at: {path}")
    
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path, nrows=5)
        elif path.endswith((".tsv", ".txt")):
            df = pd.read_csv(path, sep="\t", nrows=5)
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path).head(5)
        elif path.endswith(".jsonl"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
                    if len(records) >= 5:
                        break
            df = pd.DataFrame(records)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # Sanitize NaN/Inf values for strict JSON compliance
        clean_records = []
        for record in df.to_dict(orient="records"):
            clean_rec = {}
            for k, v in record.items():
                if pd.isna(v) or v is np.nan:
                    clean_rec[k] = None
                else:
                    clean_rec[k] = v
            clean_records.append(clean_rec)

        return {
            "columns": list(df.columns),
            "preview": clean_records,
            "row_count_sample": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect dataset: {str(e)}")


class RunPayload(BaseModel):
    config: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    overwrite_run: bool = False
    dataset_path: Optional[str] = None
    target_column: Optional[str] = None
    enable_shap: bool = False
    shap_model: Optional[str] = None

@app.post("/api/run")
async def start_pipeline_run(payload: RunPayload):
    """Triggers AutoML benchmark pipeline execution subprocess."""
    if runner_state.is_running:
        raise HTTPException(status_code=409, detail="A benchmark pipeline is already running.")

    # Save active config if supplied
    web_cfg_path = os.path.join(CONFIGS_DIR, "web_config.yml")
    if payload.config:
        save_yaml(payload.config, web_cfg_path)

    cmd = [sys.executable, "main.py", "--config", "configs/web_config.yml"]
    if payload.run_id and payload.run_id.strip():
        cmd.extend(["--run-id", payload.run_id.strip()])
    if payload.overwrite_run:
        cmd.append("--overwrite-run")
    if payload.dataset_path and payload.dataset_path.strip():
        cmd.extend(["--dataset-path", payload.dataset_path.strip()])
    if payload.target_column and payload.target_column.strip():
        cmd.extend(["--target", payload.target_column.strip()])
    if payload.enable_shap:
        cmd.append("--enable-shap")
        if payload.shap_model:
            cmd.extend(["--shap-model", payload.shap_model])

    runner_state.is_running = True
    runner_state.log_history = []
    runner_state.last_exit_code = None

    async def execute_task():
        header_msg = f"🚀 [AutoML Studio] Executing: {' '.join(cmd)}\n"
        await runner_state.broadcast_log(header_msg)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=PROJECT_ROOT
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode("utf-8", errors="replace")
                await runner_state.broadcast_log(decoded_line)
            
            await proc.wait()
            runner_state.last_exit_code = proc.returncode
            if proc.returncode == 0:
                await runner_state.broadcast_log("\n✨ [AutoML Studio] Pipeline completed successfully!\n")
            else:
                await runner_state.broadcast_log(f"\n❌ [AutoML Studio] Pipeline terminated with exit code: {proc.returncode}\n")
        except Exception as e:
            await runner_state.broadcast_log(f"\n❌ [AutoML Studio] Execution error: {str(e)}\n")
            runner_state.last_exit_code = -1
        finally:
            runner_state.is_running = False

    asyncio.create_task(execute_task())
    return {"status": "started", "command": " ".join(cmd)}


@app.get("/api/status")
async def get_status():
    """Returns real-time pipeline execution state."""
    return {
        "is_running": runner_state.is_running,
        "last_exit_code": runner_state.last_exit_code,
        "log_count": len(runner_state.log_history)
    }


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """Real-time WebSocket connection for streaming process logs."""
    await websocket.accept()
    runner_state.active_websockets.append(websocket)
    
    # Send historical logs first
    if runner_state.log_history:
        for line in runner_state.log_history:
            await websocket.send_text(line)
            
    try:
        while True:
            # Keep-alive heartbeat
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in runner_state.active_websockets:
            runner_state.active_websockets.remove(websocket)


@app.get("/api/runs")
async def list_runs(output_dir: Optional[str] = None):
    """Scans and lists all execution runs inside outputs directory."""
    base_dir = output_dir if output_dir and os.path.exists(output_dir) else OUTPUTS_DIR
    runs = []
    if os.path.exists(base_dir):
        for entry in os.listdir(base_dir):
            run_path = os.path.join(base_dir, entry)
            report_path = os.path.join(run_path, "report.json")
            if os.path.isdir(run_path) and os.path.exists(report_path):
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        rep_data = json.load(f)
                except Exception:
                    rep_data = {}

                mtime = os.path.getmtime(run_path)
                runs.append({
                    "run_id": entry,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)),
                    "mtime": mtime,
                    "champion": rep_data.get("champion", "N/A"),
                    "metrics": rep_data.get("metrics", {})
                })

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str, output_dir: Optional[str] = None):
    """Returns complete report data, metric rankings, and plot image paths for a specific run."""
    base_dir = output_dir if output_dir and os.path.exists(output_dir) else OUTPUTS_DIR
    run_path = os.path.join(base_dir, run_id)
    report_path = os.path.join(run_path, "report.json")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or report.json missing.")
    
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Collect available image assets
    images = {}
    for img_file in glob.glob(os.path.join(run_path, "*.png")):
        base_name = os.path.basename(img_file)
        images[base_name] = f"/outputs/{run_id}/{base_name}"

    # Collect SHAP report if present
    shap_data = None
    shap_reports = glob.glob(os.path.join(run_path, "*_shap_report.json"))
    if shap_reports:
        try:
            with open(shap_reports[0], "r", encoding="utf-8") as sf:
                shap_data = json.load(sf)
        except Exception:
            shap_data = None

    return {
        "run_id": run_id,
        "report": report_data,
        "images": images,
        "shap": shap_data
    }


# ==========================================
# 📦 STATIC ASSET MOUNTING
# ==========================================
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
async def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "AutoML Studio Backend Ready", "docs": "/docs"})

if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print("========================================================")
    print("🚀 Starting Enterprise AutoML Studio Web Server...")
    print("🌐 Open http://localhost:8501 in your browser")
    print("========================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=True)
