"""
event_extractor.py
==================
Identifies meteorologically significant events and separates them from
background conditions.

Events are preserved at full 6-hourly temporal resolution.
Background periods are compressed to daily means.

This results in a much smaller dataset that still faithfully represents
the high-impact weather events that GraphCast needs to learn.
"""

import logging
from typing import Optional
import numpy as np
import xarray as xr
import pandas as pd

log = logging.getLogger("Compression.EventExtractor")

# ─── Event Thresholds ─────────────────────────────────────────────────────────
# These define what constitutes a "significant" meteorological event.
# Values are in native ERA5 units.
DEFAULT_THRESHOLDS = {
    # Surface variables (CDS short names)
    "tp":   ("absolute", 0.01),       # >10mm/6h total precipitation (heavy rain)
    "t2m":  ("percentile", 95),       # Above 95th percentile temperature (heat wave)
    "msl":  ("absolute_low", 100000), # Below 1000 hPa — low-pressure system / cyclone
    "u10":  ("absolute_abs", 12.0),   # Wind speed > 12 m/s (strong wind)
    "v10":  ("absolute_abs", 12.0),   # Wind speed > 12 m/s

    # Long-name equivalents
    "total_precipitation":       ("absolute", 0.01),
    "2m_temperature":            ("percentile", 95),
    "mean_sea_level_pressure":   ("absolute_low", 100000),
    "10m_u_component_of_wind":   ("absolute_abs", 12.0),
    "10m_v_component_of_wind":   ("absolute_abs", 12.0),
}


def _compute_event_mask(
    ds: xr.Dataset,
    thresholds: Optional[dict] = None,
) -> xr.DataArray:
    """
    Builds a boolean mask over the time dimension marking event timesteps.

    A timestep is an "event" if ANY variable in the dataset exceeds its threshold
    at ANY grid point in the domain.

    Args:
        ds: ERA5 xarray Dataset.
        thresholds: Dict mapping variable_name -> (threshold_type, value).
                    threshold_type can be:
                      'absolute'      — value > threshold
                      'absolute_low'  — value < threshold
                      'absolute_abs'  — abs(value) > threshold
                      'percentile'    — value > Nth percentile of that variable

    Returns:
        xr.DataArray of shape (time,) with dtype bool.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension for event extraction.")

    n_time = ds.sizes["time"]
    # Start with all-False mask
    event_mask = xr.DataArray(
        np.zeros(n_time, dtype=bool),
        dims=["time"],
        coords={"time": ds["time"]},
    )

    for var, (thresh_type, thresh_val) in thresholds.items():
        if var not in ds.data_vars:
            continue

        da = ds[var]
        # Reduce spatial dims to get a (time,) array of max/mean
        spatial_dims = [d for d in da.dims if d != "time"]

        try:
            if thresh_type == "absolute":
                # Event if max over space exceeds threshold
                spatial_max = da.max(dim=spatial_dims)
                var_mask = spatial_max > thresh_val

            elif thresh_type == "absolute_low":
                # Event if min over space is below threshold
                spatial_min = da.min(dim=spatial_dims)
                var_mask = spatial_min < thresh_val

            elif thresh_type == "absolute_abs":
                # Event if abs max over space exceeds threshold (catches both +/-)
                spatial_absmax = abs(da).max(dim=spatial_dims)
                var_mask = spatial_absmax > thresh_val

            elif thresh_type == "percentile":
                # Event if spatial max exceeds the Nth percentile of the entire series
                pct_val = float(np.nanpercentile(da.values, thresh_val))
                spatial_max = da.max(dim=spatial_dims)
                var_mask = spatial_max > pct_val

            else:
                log.warning(f"Unknown threshold type '{thresh_type}' for '{var}'. Skipping.")
                continue

            n_events = int(var_mask.sum().values)
            log.info(f"  '{var}' ({thresh_type}={thresh_val}): {n_events} event timesteps")
            event_mask = event_mask | var_mask

        except Exception as exc:
            log.warning(f"  Event mask failed for '{var}': {exc}")

    total_events = int(event_mask.sum().values)
    log.info(f"  Total event timesteps: {total_events}/{n_time} "
             f"({total_events/n_time*100:.1f}% of dataset)")
    return event_mask


def extract_significant_events(
    ds: xr.Dataset,
    thresholds: Optional[dict] = None,
    background_resample: str = "1D",
) -> tuple[xr.Dataset, xr.Dataset, xr.DataArray]:
    """
    Splits the dataset into event and background components.

    Args:
        ds: ERA5 xarray Dataset (6-hourly).
        thresholds: Event threshold definitions (see DEFAULT_THRESHOLDS).
        background_resample: Temporal resampling for background periods (default: '1D').

    Returns:
        (event_ds, background_ds, event_mask)
        - event_ds:     Timesteps where events occur, at full 6h resolution.
        - background_ds: Non-event timesteps, resampled to daily means.
        - event_mask:   Boolean DataArray marking event timesteps.
    """
    log.info("=" * 50)
    log.info("Building event mask...")
    event_mask = _compute_event_mask(ds, thresholds)

    # Split into events and background
    event_ds      = ds.where(event_mask, drop=True)
    background_ds_6h = ds.where(~event_mask, drop=True)

    log.info(f"Event dataset:      {event_ds.sizes.get('time', 0)} timesteps @ 6h")

    # Resample background to daily means (4× compression)
    if len(background_ds_6h.sizes.get("time", [0])) > 0:
        try:
            background_ds = background_ds_6h.resample(time=background_resample).mean()
            log.info(f"Background dataset: {background_ds.sizes.get('time', 0)} "
                     f"daily means (from {background_ds_6h.sizes.get('time', 0)} timesteps)")
        except Exception as exc:
            log.warning(f"Background resampling failed: {exc}. Using raw background.")
            background_ds = background_ds_6h
    else:
        background_ds = background_ds_6h

    # Report compression
    total_6h = ds.sizes.get("time", 1)
    event_6h = event_ds.sizes.get("time", 0)
    bg_daily = background_ds.sizes.get("time", 0)
    compressed_equiv = event_6h + bg_daily
    log.info(f"\nEvent-based compression summary:")
    log.info(f"  Original:    {total_6h} × 6h timesteps")
    log.info(f"  Event data:  {event_6h} × 6h (full resolution, {event_6h/total_6h*100:.1f}%)")
    log.info(f"  Background:  {bg_daily} daily means (compressed {(total_6h-event_6h)//4:.0f} → {bg_daily})")
    log.info(f"  Equivalent total timesteps: {compressed_equiv} (was {total_6h}, "
             f"ratio {total_6h/max(compressed_equiv,1):.1f}x)")
    log.info("=" * 50)

    return event_ds, background_ds, event_mask


def merge_event_and_background(
    event_ds: xr.Dataset,
    background_ds: xr.Dataset,
    resample_background_to_6h: bool = True,
) -> xr.Dataset:
    """
    Merges event and background datasets back into a single timeline.
    Background daily means are optionally upsampled back to 6h via forward fill.

    Useful for feeding the merged result into statistical analysis or visualization.
    """
    log.info("Merging event and background datasets...")

    if resample_background_to_6h and "time" in background_ds.dims:
        try:
            bg_upsampled = background_ds.resample(time="6h").ffill()
        except Exception:
            bg_upsampled = background_ds
    else:
        bg_upsampled = background_ds

    # Combine: events take priority over background
    merged = xr.merge([event_ds, bg_upsampled], join="outer", compat="override")
    merged = merged.sortby("time")
    log.info(f"Merged dataset: {merged.sizes.get('time', 0)} timesteps")
    return merged


def event_summary_report(
    event_ds: xr.Dataset,
    background_ds: xr.Dataset,
    original_ds: xr.Dataset,
) -> pd.DataFrame:
    """Generates a compression summary report."""
    rows = []
    for var in original_ds.data_vars:
        if var not in event_ds.data_vars:
            continue
        orig_vals = original_ds[var].values.flatten()
        event_vals = event_ds[var].values.flatten() if var in event_ds.data_vars else np.array([])
        bg_vals = background_ds[var].values.flatten() if var in background_ds.data_vars else np.array([])

        rows.append({
            "variable": var,
            "original_timesteps": original_ds.sizes.get("time", 0),
            "event_timesteps": event_ds.sizes.get("time", 0),
            "background_days": background_ds.sizes.get("time", 0),
            "original_mean": float(np.nanmean(orig_vals)),
            "event_mean": float(np.nanmean(event_vals)) if len(event_vals) > 0 else np.nan,
        })
    return pd.DataFrame(rows)
