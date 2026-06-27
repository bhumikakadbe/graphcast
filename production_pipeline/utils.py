# utils.py
import logging
import time
import functools
import os
import subprocess
import shutil

# Configure basic colored logging output for the console
class ColorFormatter(logging.Formatter):
    """Custom formatter providing ANSI colors for logs."""
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def format(self, record):
        orig_msg = record.msg
        if record.levelno == logging.INFO:
            record.msg = f"{self.GREEN}{orig_msg}{self.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.YELLOW}{orig_msg}{self.RESET}"
        elif record.levelno in (logging.ERROR, logging.CRITICAL):
            record.msg = f"{self.RED}{orig_msg}{self.RESET}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{self.BLUE}{orig_msg}{self.RESET}"
        
        result = super().format(record)
        record.msg = orig_msg  # Restore original
        return result

def get_logger(name: str = "GraphCastPipeline") -> logging.Logger:
    """Returns a highly styled logger with file and console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_formatter = ColorFormatter("[%(asctime)s - %(levelname)s - %(name)s] %(message)s", datefmt="%H:%M:%S")
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)
    
    # File Handler
    os.makedirs("logs", exist_ok=True)
    f_handler = logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8")
    f_handler.setLevel(logging.DEBUG)
    f_formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d] %(message)s")
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)
    
    return logger

logger = get_logger()

def retry(max_retries: int = 5, initial_backoff: float = 2.0, backoff_factor: float = 2.0):
    """Decorator to retry a function call with exponential backoff on exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.critical(f"Function {func.__name__} failed after {max_retries} attempts.")
                        raise e
                    logger.warning(
                        f"Attempt {retries} for {func.__name__} failed with error: {e}. "
                        f"Retrying in {backoff:.2f} seconds..."
                    )
                    time.sleep(backoff)
                    backoff *= backoff_factor
            return None
        return wrapper
    return decorator

def get_ram_usage() -> str:
    """Returns host RAM usage as a string using platform-agnostic fallbacks."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return f"Host RAM: {used_gb:.2f}GB / {total_gb:.2f}GB ({mem.percent}%)"
    except ImportError:
        # Fallback if psutil is not installed yet
        return "Host RAM: psutil not installed"

def get_gpu_usage() -> str:
    """Queries nvidia-smi to obtain GPU memory details for OOM prevention."""
    if not shutil.which("nvidia-smi"):
        return "GPU Info: nvidia-smi not available (CPU Mode)"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split("\n")
        gpu_reports = []
        for i, line in enumerate(lines):
            used, total, util = line.split(",")
            gpu_reports.append(f"GPU {i}: VRAM {used.strip()}MB / {total.strip()}MB (Util: {util.strip()}%)")
        return " | ".join(gpu_reports)
    except Exception as e:
        return f"GPU Info: Query failed ({e})"

def log_system_resources(tag: str = "Resource Tracker"):
    """Logs current host and GPU usage details."""
    ram = get_ram_usage()
    gpu = get_gpu_usage()
    logger.debug(f"[{tag}] {ram} | {gpu}")
