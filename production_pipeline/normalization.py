# normalization.py
import xarray as xr
import numpy as np
from typing import Tuple, Optional
from graphcast import normalization as official_norm
from production_pipeline.utils import logger

def compute_mean_std(ds: xr.Dataset, dims: Tuple[str, ...] = ("batch", "time", "lat", "lon")) -> Tuple[xr.Dataset, xr.Dataset]:
    """Computes statistical mean and standard deviation along specific dims.
    
    Generally, we average over time and spatial coordinates per pressure level.
    """
    logger.info(f"Computing dataset mean and standard deviation along dimensions: {dims}")
    
    # Filter dimensions that are actually present
    calc_dims = [d for d in dims if d in ds.dims]
    
    mean = ds.mean(dim=calc_dims).compute()
    std = ds.std(dim=calc_dims).compute()
    
    # Avoid zero standard deviations to prevent division by zero
    for var in std.data_vars:
        std[var] = xr.where(std[var] == 0, 1.0, std[var])
        
    logger.info("Mean and standard deviation computed successfully.")
    return mean, std

def compute_residual_std(ds: xr.Dataset, dims: Tuple[str, ...] = ("batch", "time", "lat", "lon")) -> xr.Dataset:
    """Computes standard deviation of time-difference residuals: Delta x_t = x_t - x_{t-1}."""
    logger.info(f"Computing residual standard deviations along dimensions: {dims}")
    
    # Assume 'time' is a dimension, diff along time axis
    if "time" not in ds.dims:
        raise ValueError("Dataset must contain 'time' dimension to compute time-differences (residuals).")
        
    diff_ds = ds.diff(dim="time")
    calc_dims = [d for d in dims if d in diff_ds.dims]
    
    diffs_std = diff_ds.std(dim=calc_dims).compute()
    
    # Avoid division by zero
    for var in diffs_std.data_vars:
        diffs_std[var] = xr.where(diffs_std[var] == 0, 1.0, diffs_std[var])
        
    logger.info("Residual differences standard deviation computed successfully.")
    return diffs_std

def load_google_stats(
    diffs_std_path: str, 
    mean_path: str, 
    stddev_path: str
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Loads pre-computed statistics NetCDF datasets downloaded from Google's Cloud Bucket."""
    logger.info("Loading precomputed Google GraphCast statistics...")
    try:
        diffs_std = xr.load_dataset(diffs_std_path).compute()
        mean = xr.load_dataset(mean_path).compute()
        stddev = xr.load_dataset(stddev_path).compute()
        logger.info("Precomputed statistics loaded successfully.")
        return diffs_std, mean, stddev
    except Exception as e:
        logger.error(f"Failed to load statistics from paths: {e}")
        raise e

def normalize_dataset(ds: xr.Dataset, mean: xr.Dataset, stddev: xr.Dataset) -> xr.Dataset:
    """Helper to apply standard z-score normalization: z = (x - mean) / stddev."""
    return official_norm.normalize(ds, stddev, mean)

def denormalize_dataset(ds: xr.Dataset, mean: xr.Dataset, stddev: xr.Dataset) -> xr.Dataset:
    """Helper to reverse standard z-score normalization: x = z * stddev + mean."""
    return official_norm.unnormalize(ds, stddev, mean)

def wrap_predictor_with_norm(
    predictor, 
    mean_by_level: xr.Dataset, 
    stddev_by_level: xr.Dataset, 
    diffs_stddev_by_level: xr.Dataset
):
    """Wraps a GraphCast Predictor with DeepMind's official InputsAndResiduals class.
    
    Ensures input normalization and output residual addition/denormalization are performed
    exactly like the pre-trained GraphCast checkpoints.
    """
    logger.info("Wrapping model predictor with official GraphCast InputsAndResiduals normalization layers.")
    return official_norm.InputsAndResiduals(
        predictor=predictor,
        stddev_by_level=stddev_by_level,
        mean_by_level=mean_by_level,
        diffs_stddev_by_level=diffs_stddev_by_level
    )
