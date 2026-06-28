# progressive_upgrade.py
import os
import sys
import time
import logging
import dataclasses
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import xarray as xr
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graphcast import data_utils
from graphcast import graphcast
from production_pipeline import preprocessing
from production_pipeline import normalization
from production_pipeline import training
from production_pipeline.utils import logger

# Paths to checkpoints and stats
DATASET_TEMPLATE_PATH = "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
DIFFS_STD_PATH = "checkpoints/diffs_stddev_by_level.nc"
MEAN_PATH = "checkpoints/mean_by_level.nc"
STDDEV_PATH = "checkpoints/stddev_by_level.nc"

def generate_simulated_dataset(year: int, template_path: str = DATASET_TEMPLATE_PATH) -> xr.Dataset:
    """Loads local template dataset and adds minor noise/shifts to simulate data for the selected year."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template dataset missing at {template_path}")
        
    logger.info(f"Generating simulated training dataset for year {year} using template...")
    ds = xr.open_dataset(template_path, engine="scipy")
    
    # Generate variations
    np.random.seed(year)
    ds_simulated = ds.copy(deep=True)
    for var in ds_simulated.data_vars:
        if not np.issubdtype(ds_simulated[var].dtype, np.number):
            continue
        std_val = float(ds_simulated[var].std())
        if std_val > 0:
            # Add up to 2% noise to simulate seasonal variation
            noise = np.random.normal(0, 0.02 * std_val, size=ds_simulated[var].shape)
            ds_simulated[var] = ds_simulated[var] + noise
            
    # Shift time coordinate year if absolute datetimes are present
    if "time" in ds_simulated.coords:
        t_vals = ds_simulated.coords["time"].values
        if np.issubdtype(t_vals.dtype, np.datetime64):
            # Convert to pandas series to easily replace year
            pd_time = pd.to_datetime(t_vals)
            shifted_time = pd_time.map(lambda t: t.replace(year=year) if hasattr(t, 'replace') else t)
            ds_simulated.coords["time"] = shifted_time
            
    return ds_simulated

def check_downloads_complete(year: int) -> bool:
    """Verifies if the raw monthly netcdf files exist for a year."""
    raw_dir = Path("data") / "ERA5" / "raw" / str(year)
    if not raw_dir.exists():
        return False

    completed_months = 0
    for month in range(1, 13):
        p_file = raw_dir / f"era5_pressure_{year}_{month:02d}.nc"
        s_file = raw_dir / f"era5_surface_{year}_{month:02d}.nc"
        
        if p_file.exists() and p_file.stat().st_size > 5 * 1024 * 1024:
            if s_file.exists() and s_file.stat().st_size > 200 * 1024:
                completed_months += 1
                
    return completed_months == 12

def run_progressive_upgrade_flow(
    start_year: int,
    end_year: int,
    epochs_per_year: int = 1,
    use_simulation: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None, # (year, epoch, state, loss)
    stop_check: Optional[Callable[[], bool]] = None
) -> list:
    """Orchestrates yearly progressive model training (e.g. 2015 -> 2016 -> 2017)."""
    
    def log_msg(msg: str, level=logging.INFO):
        logger.log(level, msg)
        if log_callback:
            log_callback(f"[{time.strftime('%H:%M:%S')}] {msg}")
            
    log_msg(f"Initializing progressive upgrade pipeline: {start_year} -> {end_year} ({epochs_per_year} epochs/year)")
    
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year.")
        
    os.makedirs("checkpoints", exist_ok=True)
    
    # Verify standard stats
    if not (os.path.exists(DIFFS_STD_PATH) and os.path.exists(MEAN_PATH) and os.path.exists(STDDEV_PATH)):
        raise FileNotFoundError("Google statistics files missing in checkpoints/ folder!")
        
    norm_stats = normalization.load_google_stats(DIFFS_STD_PATH, MEAN_PATH, STDDEV_PATH)
    
    # Model configuration
    model_config = graphcast.ModelConfig(
        resolution=0,
        mesh_size=4,
        latent_size=32,
        gnn_msg_steps=4,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )
    
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )
    
    generated_checkpoints = []
    
    for year in range(start_year, end_year + 1):
        if stop_check and stop_check():
            log_msg("Progressive training cancellation requested. Aborting.")
            break
            
        log_msg(f"\n--- Starting Upgrades for Year {year} ---")
        
        # 1. Source parameters (weights)
        checkpoint_in = None
        if year == start_year:
            # First year: check if there's a base checkpoint already present
            if os.path.exists("checkpoints/fine_tuned_model.nc"):
                checkpoint_in = "checkpoints/fine_tuned_model.nc"
                log_msg("First year will load weights from: checkpoints/fine_tuned_model.nc")
            else:
                log_msg("First year will start from random parameter weights initialization.")
        else:
            # Subsequent years: load the model from previous year
            prev_model = f"checkpoints/model_{year-1}.nc"
            if os.path.exists(prev_model):
                checkpoint_in = prev_model
                log_msg(f"Resuming training from previous year's upgraded model: {prev_model}")
            else:
                log_msg(f"Warning: Previous year checkpoint {prev_model} missing. Starting from scratch or base.")
                if os.path.exists("checkpoints/fine_tuned_model.nc"):
                    checkpoint_in = "checkpoints/fine_tuned_model.nc"
                    
        # 2. Acquire dataset
        ds = None
        if use_simulation:
            if progress_callback:
                progress_callback(year, 0, "Generating Synthetic Data", 0.0)
            ds = generate_simulated_dataset(year, DATASET_TEMPLATE_PATH)
        else:
            # Try to load downloaded data
            if not check_downloads_complete(year):
                log_msg(f"Raw data for year {year} not found locally. Initiating CDS API download...")
                if progress_callback:
                    progress_callback(year, 0, "Downloading Data from CDS", 0.0)
                
                # Check for CDS api credentials
                cdsapirc = Path.home() / ".cdsapirc"
                if not cdsapirc.exists() and not os.environ.get("CDSAPI_KEY"):
                    log_msg("CDS API credentials not found! Fallback to simulated data mode.", logging.ERROR)
                    log_msg("To download real ERA5 data, please configure ~/.cdsapirc first.")
                    log_msg("Simulating dataset instead...")
                    ds = generate_simulated_dataset(year, DATASET_TEMPLATE_PATH)
                else:
                    try:
                        from data_collection.era5_downloader import ERA5Downloader
                        downloader = ERA5Downloader(region=[25.0, 74.0, 17.0, 85.0]) # Nagpur
                        downloader.download_year(year)
                    except Exception as e:
                        log_msg(f"CDS download failed: {e}. Fallback to simulated data mode.", logging.ERROR)
                        ds = generate_simulated_dataset(year, DATASET_TEMPLATE_PATH)
                        
            if ds is None:
                # Run compression pipeline on raw data
                log_msg(f"Running compression and validation pipeline for year {year}...")
                if progress_callback:
                    progress_callback(year, 0, "Running Data Compression", 0.0)
                try:
                    from compression.pipeline import run_compression_pipeline
                    pipeline_results = run_compression_pipeline(
                        data_path=f"data/ERA5/raw/{year}",
                        year=year,
                        use_pca=True,
                        validate=True
                    )
                    
                    # Reconstruct compressed dataset
                    from compression.pca_compressor import ERA5PCACompressor
                    compressor = ERA5PCACompressor.load(f"data/ERA5/processed/{year}/pca_compressor_{year}.pkl")
                    compressed_ds = xr.open_dataset(f"data/ERA5/processed/{year}/era5_pca_{year}.nc")
                    ds = compressor.reconstruct(compressed_ds)
                    log_msg(f"Reconstructed dataset loaded successfully. Time steps: {ds.dims.get('time', 0)}")
                except Exception as e:
                    log_msg(f"Compression pipeline failed: {e}. Using simulated dataset as fallback.", logging.ERROR)
                    ds = generate_simulated_dataset(year, DATASET_TEMPLATE_PATH)
                    
        # Apply coordinate alignments and add forcings
        ds = preprocessing.align_coordinates(ds)
        ds = preprocessing.add_graphcast_forcings(ds)
        
        # 3. Extract inputs, targets, forcings
        log_msg("Extracting training tensor slices...")
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=slice("6h", "12h"),
            **dataclasses.asdict(task_config)
        )
        
        # 4. Define callbacks
        def on_epoch(epoch_num, loss_val):
            log_msg(f"  Year {year} | Epoch {epoch_num}/{epochs_per_year} | Loss: {loss_val:.6f}")
            if progress_callback:
                progress_callback(year, epoch_num, "Training", loss_val)
                
        # 5. Run JAX training loop
        checkpoint_out = f"checkpoints/model_{year}.nc"
        log_msg(f"Starting training loop for Year {year} (resuming weights: {checkpoint_in is not None})...")
        if progress_callback:
            progress_callback(year, 0, "Initializing JAX training", 0.0)
            
        training.run_fine_tuning_loop(
            train_inputs=inputs,
            train_targets=targets,
            train_forcings=forcings,
            norm_stats=norm_stats,
            model_config=model_config,
            task_config=task_config,
            epochs=epochs_per_year,
            checkpoint_out_path=checkpoint_out,
            checkpoint_in_path=checkpoint_in,
            on_epoch_end=on_epoch
        )
        
        log_msg(f"Finished training for year {year}. Upgraded checkpoint saved to {checkpoint_out}")
        if progress_callback:
            progress_callback(year, epochs_per_year, "Completed", 0.0)
            
        generated_checkpoints.append(checkpoint_out)
        
    log_msg(f"Progressive upgrades complete! Models generated: {generated_checkpoints}")
    return generated_checkpoints
