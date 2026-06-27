"""
missing_values.py
=================
Handles missing / invalid values in ERA5 datasets using three strategies:
  1. Linear temporal interpolation (for short gaps up to `limit` steps)
  2. Climatological mean fill (same day-of-year mean from the dataset)
  3. Nearest-neighbor spatial fill (for isolated NaN cells)
"""

import logging
import numpy as np
import xarray as xr
from pathlib import Path

log = logging.getLogger("Compression.MissingValues")


def fill_linear_interpolation(ds: xr.Dataset, dim: str = "time", limit: int = 3) -> xr.Dataset:
    """
    Fills NaN gaps using linear interpolation along `dim`.
    Only fills gaps of `limit` or fewer consecutive NaN steps.
    Larger gaps are left for the climatological fill stage.
    """
    log.info(f"Applying linear interpolation (dim='{dim}', limit={limit})...")
    filled = ds.interpolate_na(dim=dim, method="linear", limit=limit)
    nan_before = int(sum(int(np.isnan(ds[v].values).sum()) for v in ds.data_vars))
    nan_after  = int(sum(int(np.isnan(filled[v].values).sum()) for v in filled.data_vars))
    log.info(f"  NaN count: {nan_before:,} -> {nan_after:,} (removed {nan_before - nan_after:,})")
    return filled


def fill_climatological_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Fills remaining NaNs with the climatological mean for the same day-of-year.
    Works even for multi-year datasets.
    """
    log.info("Applying climatological day-of-year mean fill...")
    if "time" not in ds.coords:
        log.warning("No 'time' coordinate found. Skipping climatological fill.")
        return ds

    doy_mean = ds.groupby("time.dayofyear").mean("time")

    def fill_with_clim(group):
        doy = int(group["time.dayofyear"].values[0])
        clim_slice = doy_mean.sel(dayofyear=doy)
        return group.fillna(clim_slice)

    try:
        filled = ds.groupby("time.dayofyear").map(fill_with_clim)
        nan_count = int(sum(int(np.isnan(filled[v].values).sum()) for v in filled.data_vars))
        log.info(f"  Remaining NaNs after climatological fill: {nan_count:,}")
        return filled
    except Exception as exc:
        log.warning(f"Climatological fill failed: {exc}. Returning partially filled dataset.")
        return ds


def fill_nearest_neighbor_spatial(ds: xr.Dataset) -> xr.Dataset:
    """
    Fills isolated NaN cells using nearest-neighbor interpolation across lat/lon.
    Used as a last resort for cells that have NaNs in every time step.
    """
    log.info("Applying nearest-neighbor spatial fill for residual NaNs...")
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    filled_vars = {}
    for var in ds.data_vars:
        arr = ds[var].values
        if not np.isnan(arr).any():
            filled_vars[var] = ds[var]
            continue
        # Interpolate along spatial dimensions using xarray
        try:
            filled = ds[[var]].interpolate_na(dim=lat_dim, method="nearest")
            filled = filled.interpolate_na(dim=lon_dim, method="nearest")
            filled_vars[var] = filled[var]
        except Exception:
            filled_vars[var] = ds[var]

    result = ds.copy()
    for var, arr in filled_vars.items():
        result[var] = arr

    final_nan = int(sum(int(np.isnan(result[v].values).sum()) for v in result.data_vars))
    log.info(f"  Residual NaNs after spatial fill: {final_nan:,}")
    return result


def fill_missing_values(
    ds: xr.Dataset,
    interp_limit: int = 3,
    use_climatological: bool = True,
    use_spatial: bool = True,
) -> xr.Dataset:
    """
    Master missing value handler. Runs all three strategies in sequence:
      1. Linear interpolation
      2. Climatological mean fill
      3. Nearest-neighbor spatial fill

    Args:
        ds: Input xarray Dataset (ERA5).
        interp_limit: Max consecutive NaN steps to fill by interpolation.
        use_climatological: Whether to apply climatological fill.
        use_spatial: Whether to apply spatial NN fill.

    Returns:
        xr.Dataset with missing values filled.
    """
    log.info("=" * 50)
    log.info("Starting missing value handling pipeline...")

    ds = fill_linear_interpolation(ds, limit=interp_limit)
    if use_climatological:
        ds = fill_climatological_mean(ds)
    if use_spatial:
        ds = fill_nearest_neighbor_spatial(ds)

    total_nan = int(sum(int(np.isnan(ds[v].values).sum()) for v in ds.data_vars))
    log.info(f"Missing value pipeline complete. Total residual NaNs: {total_nan:,}")
    log.info("=" * 50)
    return ds
