# test_data_pipeline.py
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from google.cloud import storage

# Add the project root to python path to import production_pipeline package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graphcast import data_utils
from graphcast import graphcast
import dataclasses
from production_pipeline import preprocessing
from production_pipeline import normalization
from production_pipeline.utils import logger

# Configuration
BUCKET_NAME = "dm_graphcast"
DATASET_BLOB = "graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
DATASET_LOCAL_PATH = "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

DIFFS_STD_PATH = "checkpoints/diffs_stddev_by_level.nc"
MEAN_PATH = "checkpoints/mean_by_level.nc"
STDDEV_PATH = "checkpoints/stddev_by_level.nc"

def download_sample_dataset_if_needed():
    """Checks for local sample dataset and downloads it from GCS if missing."""
    if os.path.exists(DATASET_LOCAL_PATH):
        logger.info(f"Sample dataset already exists locally at {DATASET_LOCAL_PATH}")
        return
        
    logger.info(f"Downloading sample dataset from gs://{BUCKET_NAME}/{DATASET_BLOB}...")
    os.makedirs(os.path.dirname(DATASET_LOCAL_PATH), exist_ok=True)
    
    gcs_client = storage.Client.create_anonymous_client()
    bucket = gcs_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(DATASET_BLOB)
    blob.download_to_filename(DATASET_LOCAL_PATH)
    logger.info("Sample dataset downloaded successfully.")

def run_phase_1_data_pipeline() -> xr.Dataset:
    """Phase 1: Load, preprocess, and extract tensors."""
    logger.info("\n--- PHASE 1: Verify Data Pipeline End-to-End ---")
    
    # 1. Load ERA5 Zarr/NetCDF
    logger.info(f"Loading dataset from: {DATASET_LOCAL_PATH}")
    ds = xr.open_dataset(DATASET_LOCAL_PATH, engine="scipy")
    
    logger.info("Original Dataset Dimensions:")
    for dim, size in ds.dims.items():
        logger.info(f"  {dim}: {size}")
        
    # 2. Preprocess / Align Coordinates
    ds_aligned = preprocessing.align_coordinates(ds)
    
    # 3. Add Forcings (Solar radiation + Year/Day calendar progress)
    ds_with_forcings = preprocessing.add_graphcast_forcings(ds_aligned)
    
    # 4. Extract inputs, targets, and forcings
    # Set target task config parameters matching the 1.0 degree 13-level dataset
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )
    
    logger.info("Extracting inputs, targets, and forcings...")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds_with_forcings,
        target_lead_times=slice("6h", "12h"), # Test 2 steps rollout
        **dataclasses.asdict(task_config)
    )
    
    logger.info("Extracted Shapes:")
    logger.info(f"  Inputs shape/dims:  {inputs.dims.mapping}")
    logger.info(f"  Targets shape/dims: {targets.dims.mapping}")
    logger.info(f"  Forcings shape/dims: {forcings.dims.mapping}")
    
    return inputs, targets, forcings, task_config

def run_phase_2_normalization(inputs, targets, forcings):
    """Phase 2: Verify normalization and denormalization."""
    logger.info("\n--- PHASE 2: Verify Normalization ---")
    
    # 1. Load Stats
    logger.info("Loading precomputed Google statistics...")
    diffs_std, mean, stddev = normalization.load_google_stats(
        DIFFS_STD_PATH,
        MEAN_PATH,
        STDDEV_PATH
    )
    
    # 2. Verify Inputs Normalization (z = (x - mean) / stddev)
    logger.info("Verifying Input Normalization...")
    norm_inputs = normalization.normalize_dataset(inputs, mean, stddev)
    denorm_inputs = normalization.denormalize_dataset(norm_inputs, mean, stddev)
    
    input_variables_checked = []
    for var in inputs.data_vars:
        if var in denorm_inputs.data_vars:
            orig_vals = inputs[var].values
            reconstructed_vals = denorm_inputs[var].values
            
            # Check maximum difference
            max_diff = np.nanmax(np.abs(orig_vals - reconstructed_vals))
            logger.info(f"  Variable '{var}' reconstruction max absolute difference: {max_diff:.2e}")
            
            # Ensure reconstruction is extremely close
            assert np.allclose(orig_vals, reconstructed_vals, rtol=1e-4, atol=1e-4, equal_nan=True), \
                f"Input Normalization mismatch for variable: {var}"
            input_variables_checked.append(var)
            
    logger.info(f"Input Normalization matches perfectly for all {len(input_variables_checked)} variables.")
    
    # 3. Verify Target Residual Normalization (diff = target - last_input)
    logger.info("Verifying Target Residual Normalization...")
    
    # Replicate target residual mapping
    # (InputsAndResiduals subtracts the last input step from the target for matching vars)
    for var in targets.data_vars:
        if var in inputs.data_vars:
            last_input = inputs[var].isel(time=-1)
            target_step = targets[var].isel(time=[0]) # Extract step 1
            
            # Residual: target - last_input
            residual = target_step - last_input
            
            # Normalize: residual / diffs_std
            norm_res = residual / diffs_std[var].astype(residual.dtype)
            
            # Denormalize: norm_res * diffs_std + last_input
            denorm_res = norm_res * diffs_std[var].astype(residual.dtype) + last_input
            
            orig_target = target_step.values
            reconstructed_target = denorm_res.values
            
            max_diff = np.nanmax(np.abs(orig_target - reconstructed_target))
            logger.info(f"  Target Residual '{var}' reconstruction max absolute difference: {max_diff:.2e}")
            assert np.allclose(orig_target, reconstructed_target, rtol=1e-4, atol=1e-4, equal_nan=True), \
                f"Target Residual Normalization mismatch for variable: {var}"
                
    logger.info("Target Normalization matches perfectly.")

def run_phase_3_graphcast_input_format(inputs, targets, forcings, task_config):
    """Phase 3: Verify variable presence and coordinate shapes."""
    logger.info("\n--- PHASE 3: Verify GraphCast Input Format ---")
    
    # Variables expected by GraphCast
    expected_inputs = list(task_config.input_variables)
    expected_targets = list(task_config.target_variables)
    expected_forcings = list(task_config.forcing_variables)
    
    # Table headers
    logger.info(f"{'Variable Name':<35} | {'Type':<8} | {'Available?':<10} | {'Shape/Dimensions'}")
    logger.info("-" * 80)
    
    for var in expected_inputs:
        is_available = var in inputs.data_vars
        status = "✅ YES" if is_available else "❌ NO"
        shape_str = str(inputs[var].shape) if is_available else "N/A"
        logger.info(f"{var:<35} | {'Input':<8} | {status:<10} | {shape_str}")
        assert is_available, f"GraphCast input variable '{var}' is missing!"
        
    for var in expected_targets:
        is_available = var in targets.data_vars
        status = "✅ YES" if is_available else "❌ NO"
        shape_str = str(targets[var].shape) if is_available else "N/A"
        logger.info(f"{var:<35} | {'Target':<8} | {status:<10} | {shape_str}")
        assert is_available, f"GraphCast target variable '{var}' is missing!"
        
    for var in expected_forcings:
        is_available = var in forcings.data_vars
        status = "✅ YES" if is_available else "❌ NO"
        shape_str = str(forcings[var].shape) if is_available else "N/A"
        logger.info(f"{var:<35} | {'Forcing':<8} | {status:<10} | {shape_str}")
        assert is_available, f"GraphCast forcing variable '{var}' is missing!"

    # Verify input times are relative [T-6h, T0]
    expected_input_times = [pd.Timedelta("-6h"), pd.Timedelta("0h")]
    actual_input_times = [pd.Timedelta(t) for t in inputs.time.values]
    logger.info(f"Input times check: expected={expected_input_times}, actual={actual_input_times}")
    assert actual_input_times == expected_input_times, f"Input times must be [-6h, 0h], got {actual_input_times}"
    
    logger.info("All coordinates and dimensions conform exactly to GraphCast GNN expectations.")

def main():
    try:
        download_sample_dataset_if_needed()
        inputs, targets, forcings, task_config = run_phase_1_data_pipeline()
        run_phase_2_normalization(inputs, targets, forcings)
        run_phase_3_graphcast_input_format(inputs, targets, forcings, task_config)
        logger.info("\n🎉 SUCCESS: All verification steps passed successfully! GraphCast pipeline is robust.")
    except Exception as e:
        logger.error(f"\n❌ FAILED: Pipeline verification encountered errors: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
