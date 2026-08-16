import pytest
import os
import json
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_api_config_endpoints():
    """Verifies GET and POST /api/config functionality."""
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "default" in data
    assert "active" in data

    # Test saving config
    sample_cfg = {
        "framework": {"random_state": 42, "active_models": ["XGBoost"]},
        "data": {"target_column": "Target_Y"}
    }
    save_res = client.post("/api/config", json={"config": sample_cfg})
    assert save_res.status_code == 200
    assert save_res.json()["status"] == "success"


def test_api_datasets_endpoints():
    """Verifies GET /api/datasets and inspect functionality."""
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    assert len(data["datasets"]) > 0

    target_csv = data["datasets"][0]["path"]
    inspect_res = client.get(f"/api/datasets/inspect?path={target_csv}")
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert "columns" in inspect_data
    assert "preview" in inspect_data
    assert len(inspect_data["columns"]) > 0


def test_api_status_and_runs_endpoints():
    """Verifies GET /api/status and GET /api/runs."""
    status_res = client.get("/api/status")
    assert status_res.status_code == 200
    assert "is_running" in status_res.json()

    runs_res = client.get("/api/runs")
    assert runs_res.status_code == 200
    assert "runs" in runs_res.json()
    
    if len(runs_res.json()["runs"]) > 0:
        sample_run_id = runs_res.json()["runs"][0]["run_id"]
        detail_res = client.get(f"/api/runs/{sample_run_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["run_id"] == sample_run_id
        assert "report" in detail_data
