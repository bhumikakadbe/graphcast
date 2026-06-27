"""
windowing.py
============
Computes rolling window statistics (mean/sum) for ERA5 weather datasets.
Supports 7-day, 14-day, and 30-day windows at 6-hourly temporal resolution.
"""

import logging
import xarray as xr
from typing import Dict, List, Optional

log = logging.getLogger("Compression.Windowing")

# 6-hourly timesteps (4 steps per day)
DEFAULT_WINDOWS = {
    "7day": 28,
    "14day": 56,
    "30day": 120,
}


def compute_rolling_windows(
    ds: xr.Dataset,
    windows: Optional[Dict[str, int]] = None,
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Computes rolling statistics for variables in the dataset.
    Uses rolling mean for standard variables and rolling sum for precipitation.

    Args:
        ds: Input xarray Dataset.
        windows: Dict mapping window label to number of timesteps.
                 Defaults to 7, 14, and 30 days.
        variables: List of variables to compute rolling windows for.
                   If None, computes for all data variables.

    Returns:
        xr.Dataset containing only the computed rolling window variables.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if variables is None:
        variables = list(ds.data_vars)

    log.info("=" * 50)
    log.info(f"Computing rolling windows for {len(variables)} variables...")
    log.info(f"Windows: {windows}")

    if "time" not in ds.dims:
        raise ValueError("Dataset must contain 'time' dimension for rolling computations.")

    rolling_ds = xr.Dataset(coords=ds.coords)

    for var in variables:
        if var not in ds.data_vars:
            log.warning(f"Variable '{var}' not in dataset. Skipping.")
            continue

        # Check if this is a precipitation/accumulation variable
        # Precipitation variables are summed, while others are averaged
        is_accum = any(term in var.lower() for term in ["precipitation", "precip", "rain", "snow"])
        stat_name = "sum" if is_accum else "mean"

        for label, steps in windows.items():
            new_var_name = f"{var}_{label}_{stat_name}"
            log.info(f"  Computing {label} rolling {stat_name} for '{var}' -> '{new_var_name}'...")
            
            # Compute rolling operation along time dimension
            rolling_obj = ds[var].rolling(time=steps, min_periods=1)
            
            if is_accum:
                rolling_ds[new_var_name] = rolling_obj.sum()
            else:
                rolling_ds[new_var_name] = rolling_obj.mean()

    log.info("Rolling window computation complete.")
    log.info("=" * 50)
    return rolling_ds
