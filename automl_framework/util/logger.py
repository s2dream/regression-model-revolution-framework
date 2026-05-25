import os
import yaml
import logging
from datetime import datetime

def setup_logger(turn: int = 1, config: dict = None) -> logging.Logger:
    """
    Sets up a centralized logger that writes to both console and a dynamic log file.
    Logs are stored in a 'logs' directory, named with the turn and timestamp.
    Prints the loaded config at the start of the log file for easy reference.
    """
    # 1. Determine log directory
    log_dir = "logs"
    if config:
        log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2. Unique log filename based on turn and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"automl_turn_{turn}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    latest_log_path = os.path.join(log_dir, "latest.log")

    # 3. Format configuration dictionary as a YAML block for the log header
    config_header = ""
    if config:
        try:
            config_str = yaml.dump(config, default_flow_style=False)
            config_header = (
                "====================================================================\n"
                f"🚀 AutoML Regression Framework Run Configuration (Turn {turn} - {timestamp})\n"
                "====================================================================\n"
                f"{config_str}"
                "====================================================================\n\n"
            )
        except Exception as e:
            config_header = f"# Error dumping configuration: {e}\n\n"

    # 4. Initialize log files and write the configuration header
    for path in [log_path, latest_log_path]:
        try:
            with open(path, "w", encoding="utf-8") as f:
                if config_header:
                    f.write(config_header)
        except Exception as e:
            # Fallback if writing file fails (e.g. permission or lock issues)
            print(f"[Logger Setup] WARNING: Could not write header to {path}: {e}")

    # 5. Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # 6. Formatting style
    log_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
    formatter = logging.Formatter(log_format)

    # 7. File Handlers (capturing detailed DEBUG logs)
    try:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[Logger Setup] ERROR setting up file handler for {log_path}: {e}")

    try:
        latest_handler = logging.FileHandler(latest_log_path, mode="a", encoding="utf-8")
        latest_handler.setLevel(logging.DEBUG)
        latest_handler.setFormatter(formatter)
        logger.addHandler(latest_handler)
    except Exception as e:
        print(f"[Logger Setup] ERROR setting up file handler for {latest_log_path}: {e}")

    # 8. Console Handler (active monitoring, defaults to INFO level)
    console_handler = logging.StreamHandler()
    console_level = logging.INFO
    if config:
        console_level_str = config.get("logging", {}).get("console_level", "INFO").upper()
        console_level = getattr(logging, console_level_str, logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Output saved to: {log_path}")
    return logger
