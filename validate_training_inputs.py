# validate_training_inputs.py
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graphcast import data_utils
from graphcast import graphcast
from production_pipeline import preprocessing
import dataclasses
from production_pipeline.utils import logger

DATASET_LOCAL_PATH = "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

def main():
    logger.info("=== Starting Phase 4A: Validate Training Inputs ===")
    
    if not os.path.exists(DATASET_LOCAL_PATH):
        logger.error(f"Local sample dataset not found at: {DATASET_LOCAL_PATH}")
        sys.exit(1)
        
    try:
        # 1. Load data
        logger.info(f"Loading local dataset: {DATASET_LOCAL_PATH}")
        ds = xr.open_dataset(DATASET_LOCAL_PATH, engine="scipy")
        
        # 2. Preprocess / Align Coordinates
        ds = preprocessing.align_coordinates(ds)
        ds = preprocessing.add_graphcast_forcings(ds)
        
        # 3. Extract inputs, targets, and forcings
        task_config = graphcast.TaskConfig(
            input_variables=graphcast.TASK.input_variables,
            target_variables=graphcast.TASK.target_variables,
            forcing_variables=graphcast.TASK.forcing_variables,
            pressure_levels=graphcast.PRESSURE_LEVELS[13],
            input_duration=graphcast.TASK.input_duration,
        )
        
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=slice("6h", "12h"),
            **dataclasses.asdict(task_config)
        )
        
        errors = 0
        
        # Check coordinates and dimensions
        logger.info("Checking dimensions:")
        expected_dims = {
            "batch": 1,
            "time": 2,
            "level": 13,
            "lat": 181,
            "lon": 360
        }
        
        for coord, size in expected_dims.items():
            if coord in inputs.dims:
                actual_size = inputs.dims[coord]
                if actual_size != size:
                    logger.error(f"  Dimension '{coord}' mismatch in inputs! Expected {size}, got {actual_size}")
                    errors += 1
                else:
                    logger.info(f"  Inputs '{coord}' dimension matches expected size {size}")
            else:
                logger.error(f"  Dimension '{coord}' missing from inputs!")
                errors += 1
                
        # Check target dimensions (level dimension is present for atmospheric targets, but not for surface targets)
        # So we just verify dimensions mapping sizes
        
        # Check NaN values
        logger.info("Checking for NaN values:")
        for name, var in inputs.data_vars.items():
            nan_count = int(var.isnull().sum())
            if nan_count > 0:
                logger.warning(f"  Input variable '{name}' has {nan_count} NaN values!")
            else:
                logger.info(f"  Input variable '{name}': 0 NaNs")
                
        for name, var in targets.data_vars.items():
            nan_count = int(var.isnull().sum())
            if nan_count > 0:
                logger.warning(f"  Target variable '{name}' has {nan_count} NaN values!")
            else:
                logger.info(f"  Target variable '{name}': 0 NaNs")
                
        for name, var in forcings.data_vars.items():
            nan_count = int(var.isnull().sum())
            if nan_count > 0:
                logger.warning(f"  Forcing variable '{name}' has {nan_count} NaN values!")
            else:
                logger.info(f"  Forcing variable '{name}': 0 NaNs")
                
        # Check variable names
        logger.info("Checking variable presence:")
        for var in task_config.input_variables:
            if var not in inputs.data_vars:
                logger.error(f"  Required input variable '{var}' is missing!")
                errors += 1
                
        for var in task_config.target_variables:
            if var not in targets.data_vars:
                logger.error(f"  Required target variable '{var}' is missing!")
                errors += 1
                
        for var in task_config.forcing_variables:
            if var not in forcings.data_vars:
                logger.error(f"  Required forcing variable '{var}' is missing!")
                errors += 1
                
        # Check pressure levels
        logger.info("Checking pressure levels:")
        expected_levels = list(graphcast.PRESSURE_LEVELS[13])
        actual_levels = list(inputs.level.values)
        if actual_levels != expected_levels:
            logger.error(f"  Pressure levels mismatch! Expected {expected_levels}, got {actual_levels}")
            errors += 1
        else:
            logger.info("  Pressure levels match expected list.")
            
        if errors > 0:
            logger.error(f"Validation finished with {errors} errors.")
            sys.exit(1)
            
        logger.info("🎉 Validation Successful: All checks passed! Ready for training.")
        
    except Exception as e:
        logger.error(f"Validation crashed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
