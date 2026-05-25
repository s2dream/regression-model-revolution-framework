import os
import logging
import pytest
from automl_framework.util.logger import setup_logger

@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture supplying a temporary logs directory."""
    return str(tmp_path / "logs")

def test_logger_setup(temp_log_dir):
    # Setup standard config mock
    config = {
        "logging": {
            "log_dir": temp_log_dir,
            "console_level": "DEBUG"
        },
        "framework": {
            "random_state": 42
        }
    }
    
    # Call logging configuration setup
    logger = setup_logger(turn=3, config=config)
    
    # Issue test messages
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message")
    
    # Assert logs directory was created
    assert os.path.exists(temp_log_dir)
    
    # Assert latest.log was created
    latest_path = os.path.join(temp_log_dir, "latest.log")
    assert os.path.exists(latest_path)
    
    # Verify the dynamic turn specific log file was created
    files = os.listdir(temp_log_dir)
    turn_files = [f for f in files if "turn_3" in f]
    assert len(turn_files) == 1
    turn_file_path = os.path.join(temp_log_dir, turn_files[0])
    
    # Check correct logging header, yaml configuration output, and logging content
    with open(turn_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "🚀 AutoML Regression Framework Run Configuration" in content
    assert "random_state: 42" in content
    assert "Test INFO message" in content
    assert "Test DEBUG message" in content

    # Verify latest.log contains identical contents
    with open(latest_path, "r", encoding="utf-8") as f:
        latest_content = f.read()
        
    assert "🚀 AutoML Regression Framework Run Configuration" in latest_content
    assert "Test INFO message" in latest_content
