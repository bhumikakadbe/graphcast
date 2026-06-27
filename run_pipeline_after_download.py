"""
run_pipeline_after_download.py
==============================
Monitors the background ERA5 download and automatically runs the compression
pipeline and validation experiment once all 12 months of 2015 data are ready.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/master_pipeline_run.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("MasterPipelineOrchestrator")


def check_downloads_complete(year: int) -> bool:
    """Verifies that all 12 months (pressure + surface) have finished downloading."""
    raw_dir = Path("data") / "ERA5" / "raw" / str(year)
    if not raw_dir.exists():
        return False

    completed_months = 0
    for month in range(1, 13):
        p_file = raw_dir / f"era5_pressure_{year}_{month:02d}.nc"
        s_file = raw_dir / f"era5_surface_{year}_{month:02d}.nc"
        
        # Verify both files exist and have reasonable sizes
        if p_file.exists() and p_file.stat().st_size > 10 * 1024 * 1024: # > 10MB
            if s_file.exists() and s_file.stat().st_size > 500 * 1024:  # > 500KB
                completed_months += 1

    log.info(f"Checking download directory: {completed_months}/12 months fully downloaded.")
    return completed_months == 12


def run_command(command: list) -> bool:
    """Runs a shell command and streams output to log."""
    log.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            log.info(f"  [SUBPROCESS] {line.strip()}")
        process.wait()
        return process.returncode == 0
    except Exception as e:
        log.error(f"Command failed with exception: {e}")
        return False


def main():
    year = 2015
    python_path = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_path):
        python_path = "python" # fallback

    log.info("=" * 60)
    log.info(f"Starting master pipeline orchestrator for Year {year}...")
    log.info("=" * 60)

    # 1. Wait for downloader to finish
    log.info("Monitoring download loop...")
    while not check_downloads_complete(year):
        # Sleep for 2 minutes before checking again
        time.sleep(120)

    log.info("All raw ERA5 data for 2015 successfully downloaded and verified!")

    # 2. Run compression pipeline
    log.info("Triggering Stage 2: Compression Pipeline...")
    compression_cmd = [
        python_path,
        "compression/pipeline.py",
        f"data/ERA5/raw/{year}/",
        "--year", str(year)
    ]
    success = run_command(compression_cmd)
    if not success:
        log.error("Compression pipeline failed! Aborting master workflow.")
        sys.exit(1)

    log.info("Compression pipeline finished successfully!")

    # 3. Run validation experiment
    log.info("Triggering Stage 3: Compression Validation Experiment...")
    validation_cmd = [
        python_path,
        "compression/validation_experiment.py",
        "--raw-dir", f"data/ERA5/raw/{year}/"
    ]
    success = run_command(validation_cmd)
    if not success:
        log.error("Validation experiment failed! Degradation criteria might have been violated.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("🎉 SUCCESS: Year 2015 pipeline and validation completed successfully!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
