"""
outlier_detection.py
====================
Detects and removes physically impossible or sensor-error values in ERA5 data.

Two complementary methods:
  1. Physical bounds clipping — hard limits based on Earth's observed extremes
  2. Rolling z-score detection — flags values that deviate unusually from local mean
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger("Compression.OutlierDetection")

# ─── Physical Bounds ──────────────────────────────────────────────────────────
# Hard limits derived from absolute recorded extremes on Earth.
# Values outside these ranges are physically impossible and must be errors.
PHYSICAL_BOUNDS = {
    # Surface variables (CDS short names)
    "t2m":  (180.0, 340.0),      # 2m temp: -93°C (Antarctica) to +67°C (Death Valley)
    "msl":  (87000.0, 108400.0), # MSLP in Pa: typhoon eye to Siberian winter high
    "sp":   (50000.0, 110000.0), # Surface pressure in Pa
    "u10":  (-100.0, 100.0),     # 10m U-wind in m/s
    "v10":  (-100.0, 100.0),     # 10m V-wind in m/s
    "tp":   (0.0, 0.5),          # Total precip in m/6h (0.5m = 500mm in 6h = impossible)
    "tcc":  (0.0, 1.0),          # Cloud cover: fraction 0-1

    # Pressure-level variables (generic names used at all levels)
    "temperature":          (150.0, 340.0),   # K
    "specific_humidity":    (0.0, 0.04),      # kg/kg — 40 g/kg max
    "geopotential":         (-5000.0, 600000.0),  # m²/s² — surface to top of atm
    "u_component_of_wind":  (-200.0, 200.0),  # m/s at jet stream level
    "v_component_of_wind":  (-200.0, 200.0),
    "vertical_velocity":    (-20.0, 20.0),    # Pa/s — extreme convection

    # Alternative CDS names (NetCDF files may use these)
    "2m_temperature":           (180.0, 340.0),
    "mean_sea_level_pressure":  (87000.0, 108400.0),
    "total_precipitation":      (0.0, 0.5),
    "total_cloud_cover":        (0.0, 1.0),
    "10m_u_component_of_wind":  (-100.0, 100.0),
    "10m_v_component_of_wind":  (-100.0, 100.0),
    "surface_pressure":         (50000.0, 110000.0),
}

# ─── Z-Score Parameters ───────────────────────────────────────────────────────
DEFAULT_Z_THRESHOLD  = 4.0   # Flag values > 4 std deviations from rolling mean
DEFAULT_ROLLING_DAYS = 30    # Rolling window: 30 days = 120 timesteps at 6h


def apply_physical_bounds(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """
    Clips all variables to their physically possible range.
    Values outside bounds are set to NaN (to be filled by the missing-value stage).

    Returns:
        (clipped_dataset, report_dict)
    """
    log.info("Applying physical bounds clipping...")
    report = {}

    for var in ds.data_vars:
        bounds = PHYSICAL_BOUNDS.get(var)
        if bounds is None:
            log.debug(f"  No bounds defined for '{var}' — skipping.")
            continue

        lo, hi = bounds
        original = ds[var].values
        n_below = int((original < lo).sum())
        n_above = int((original > hi).sum())
        n_total = n_below + n_above

        if n_total > 0:
            log.info(f"  '{var}': {n_total:,} outliers clipped "
                     f"(below {lo}: {n_below:,}, above {hi}: {n_above:,})")
            # Set outliers to NaN — downstream fill_missing_values will handle them
            clipped = ds[var].where((ds[var] >= lo) & (ds[var] <= hi))
            ds = ds.assign({var: clipped})
        else:
            log.debug(f"  '{var}': No physical bound violations found.")

        report[var] = {"n_clipped": n_total, "lo": lo, "hi": hi}

    total_clipped = sum(r["n_clipped"] for r in report.values())
    log.info(f"Physical bounds complete. Total values clipped: {total_clipped:,}")
    return ds, report


def apply_zscore_detection(
    ds: xr.Dataset,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> tuple[xr.Dataset, dict]:
    """
    Detects statistically extreme values using a rolling z-score.
    Values that deviate more than `z_threshold` standard deviations from
    a `rolling_days`-day rolling mean are set to NaN.

    This catches sensor drift, data corruption, and sudden unrealistic spikes
    that are within physical bounds but statistically impossible.

    Args:
        ds: Input ERA5 xarray Dataset.
        z_threshold: Number of std deviations to flag as outlier (default: 4.0).
        rolling_days: Rolling window size in days (default: 30 days = 120 timesteps).

    Returns:
        (cleaned_dataset, report_dict)
    """
    rolling_steps = rolling_days * 4  # 4 × 6h timesteps per day
    log.info(f"Applying rolling z-score detection "
             f"(threshold={z_threshold}, window={rolling_days} days / {rolling_steps} steps)...")
    report = {}

    for var in ds.data_vars:
        if "time" not in ds[var].dims:
            log.debug(f"  '{var}' has no time dimension — skipping z-score.")
            continue
        try:
            arr = ds[var]
            rolling_mean = arr.rolling(time=rolling_steps, center=True, min_periods=1).mean()
            rolling_std  = arr.rolling(time=rolling_steps, center=True, min_periods=1).std()

            # Avoid division by zero for constant fields
            rolling_std = rolling_std.where(rolling_std > 1e-10, other=1e-10)

            z_score = abs(arr - rolling_mean) / rolling_std
            n_flagged = int((z_score > z_threshold).sum().values)

            if n_flagged > 0:
                log.info(f"  '{var}': {n_flagged:,} statistical outliers flagged (z > {z_threshold})")
                cleaned = arr.where(z_score <= z_threshold)
                ds = ds.assign({var: cleaned})
            else:
                log.debug(f"  '{var}': No statistical outliers detected.")

            report[var] = {"n_flagged": n_flagged, "z_threshold": z_threshold}

        except Exception as exc:
            log.warning(f"  z-score detection failed for '{var}': {exc}")
            report[var] = {"n_flagged": 0, "error": str(exc)}

    total_flagged = sum(r.get("n_flagged", 0) for r in report.values())
    log.info(f"Z-score detection complete. Total values flagged: {total_flagged:,}")
    return ds, report


def detect_and_remove_outliers(
    ds: xr.Dataset,
    use_physical_bounds: bool = True,
    use_zscore: bool = True,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> tuple[xr.Dataset, dict]:
    """
    Master outlier handler. Runs both detection methods in sequence.

    Args:
        ds: Input ERA5 dataset.
        use_physical_bounds: Apply hard physical limit clipping.
        use_zscore: Apply rolling z-score detection.
        z_threshold: Z-score cutoff (default 4.0 = flags ~0.006% of normal data).
        rolling_days: Rolling window for z-score (default 30 days).

    Returns:
        (cleaned_dataset, combined_report_dict)
    """
    log.info("=" * 50)
    log.info("Starting outlier detection pipeline...")
    combined_report = {}

    if use_physical_bounds:
        ds, bounds_report = apply_physical_bounds(ds)
        combined_report["physical_bounds"] = bounds_report

    if use_zscore:
        ds, zscore_report = apply_zscore_detection(ds, z_threshold, rolling_days)
        combined_report["zscore"] = zscore_report

    log.info("Outlier detection pipeline complete.")
    log.info("=" * 50)
    return ds, combined_report


def generate_outlier_report(report: dict) -> pd.DataFrame:
    """Converts the outlier report dict into a readable DataFrame."""
    rows = []
    for method, var_reports in report.items():
        for var, stats in var_reports.items():
            rows.append({"method": method, "variable": var, **stats})
    return pd.DataFrame(rows)
