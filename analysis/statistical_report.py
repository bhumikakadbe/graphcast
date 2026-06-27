"""
statistical_report.py
=====================
Computes and exports per-variable descriptive statistics, seasonal breakdowns,
and a correlation matrix for ERA5 datasets.

Outputs to:  logs/stats_report_{year}.csv
             logs/stats_report_{year}.md
             logs/seasonal_stats_{year}.csv
             logs/correlation_matrix_{year}.csv
             logs/correlation_matrix_{year}.png
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("StatisticalReport")

# ─── Season Definitions ───────────────────────────────────────────────────────
SEASONS = {
    "Summer (MAM)":       [3, 4, 5],
    "Monsoon (JJAS)":     [6, 7, 8, 9],
    "Post-Monsoon (ON)":  [10, 11],
    "Winter (DJF)":       [12, 1, 2],
}

# Human-readable units per variable
VARIABLE_UNITS = {
    "2m_temperature":           "K (°C = K - 273.15)",
    "total_precipitation":      "m/6h",
    "mean_sea_level_pressure":  "Pa",
    "specific_humidity":        "kg/kg",
    "u_component_of_wind":      "m/s",
    "v_component_of_wind":      "m/s",
    "10m_u_component_of_wind":  "m/s",
    "10m_v_component_of_wind":  "m/s",
    "temperature":              "K",
    "geopotential":             "m²/s²",
    "total_cloud_cover":        "fraction (0–1)",
    "vertical_velocity":        "Pa/s",
}

# Variables to include in correlation matrix
CORRELATION_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "mean_sea_level_pressure",
    "specific_humidity",
    "total_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def load_dataset(path: str) -> xr.Dataset:
    """Loads an ERA5 NetCDF file. Handles both scipy and netcdf4 engines."""
    log.info(f"Loading dataset: {path}")
    try:
        ds = xr.open_dataset(path, engine="scipy")
    except Exception:
        ds = xr.open_dataset(path, engine="netcdf4")
    log.info(f"Loaded. Variables: {list(ds.data_vars)} | Time steps: {ds.dims.get('time', 'N/A')}")
    return ds


def compute_variable_stats(ds: xr.Dataset, year: int) -> pd.DataFrame:
    """
    Computes descriptive statistics for every variable in the dataset.

    For each variable:
        min, max, mean, median, std, variance, skewness, kurtosis, 5th percentile,
        95th percentile, count of NaN values.

    Returns:
        pd.DataFrame with one row per variable.
    """
    log.info(f"Computing descriptive statistics for year {year}...")
    rows = []
    for var in ds.data_vars:
        vals = ds[var].values.flatten().astype(np.float64)
        nan_count = int(np.isnan(vals).sum())
        vals_clean = vals[~np.isnan(vals)]

        if len(vals_clean) == 0:
            log.warning(f"Variable '{var}' is entirely NaN — skipping.")
            continue

        row = {
            "variable":   var,
            "year":       year,
            "unit":       VARIABLE_UNITS.get(var, "unknown"),
            "count":      len(vals_clean),
            "nan_count":  nan_count,
            "nan_pct":    round(nan_count / len(vals) * 100, 2),
            "min":        float(np.nanmin(vals)),
            "max":        float(np.nanmax(vals)),
            "mean":       float(np.nanmean(vals)),
            "median":     float(np.nanmedian(vals)),
            "std":        float(np.nanstd(vals)),
            "variance":   float(np.nanvar(vals)),
            "p05":        float(np.nanpercentile(vals, 5)),
            "p25":        float(np.nanpercentile(vals, 25)),
            "p75":        float(np.nanpercentile(vals, 75)),
            "p95":        float(np.nanpercentile(vals, 95)),
            "skewness":   float(sp_stats.skew(vals_clean)),
            "kurtosis":   float(sp_stats.kurtosis(vals_clean)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Statistics computed for {len(df)} variables.")
    return df


def compute_seasonal_stats(ds: xr.Dataset, year: int) -> pd.DataFrame:
    """
    Computes mean and std per variable per meteorological season.

    Returns:
        pd.DataFrame with columns: variable, season, mean, std, min, max
    """
    log.info(f"Computing seasonal statistics for year {year}...")
    rows = []

    # Ensure 'time' coordinate can provide month information
    if "time" not in ds.coords:
        log.warning("Dataset has no 'time' coordinate. Skipping seasonal analysis.")
        return pd.DataFrame()

    for season_name, months in SEASONS.items():
        # Select timesteps belonging to this season
        try:
            season_ds = ds.sel(time=ds["time.month"].isin(months))
        except AttributeError:
            # time coordinate might not support .month — try pandas
            try:
                time_vals = pd.to_datetime(ds["time"].values)
                mask = np.isin(time_vals.month, months)
                season_ds = ds.isel(time=mask)
            except Exception as exc:
                log.warning(f"Could not filter season {season_name}: {exc}")
                continue

        for var in ds.data_vars:
            if var not in season_ds.data_vars:
                continue
            vals = season_ds[var].values.flatten()
            vals_clean = vals[~np.isnan(vals)]
            if len(vals_clean) == 0:
                continue
            rows.append({
                "variable": var,
                "year":     year,
                "season":   season_name,
                "mean":     float(np.nanmean(vals)),
                "std":      float(np.nanstd(vals)),
                "min":      float(np.nanmin(vals)),
                "max":      float(np.nanmax(vals)),
                "median":   float(np.nanmedian(vals)),
            })

    df = pd.DataFrame(rows)
    log.info(f"Seasonal statistics computed: {len(df)} rows.")
    return df


def compute_correlation_matrix(ds: xr.Dataset, year: int) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix for a subset of key ERA5 surface variables.

    Returns:
        pd.DataFrame — square correlation matrix (variables × variables)
    """
    log.info(f"Computing correlation matrix for year {year}...")
    available_vars = [v for v in CORRELATION_VARIABLES if v in ds.data_vars]

    if len(available_vars) < 2:
        log.warning("Not enough surface variables for correlation analysis.")
        return pd.DataFrame()

    # Flatten spatial and time dimensions into 1D series per variable
    series = {}
    min_len = None
    for var in available_vars:
        vals = ds[var].values.flatten()
        series[var] = vals
        min_len = len(vals) if min_len is None else min(min_len, len(vals))

    # Build 2D array: (n_samples, n_variables)
    data_matrix = np.column_stack([series[v][:min_len] for v in available_vars])

    # Remove rows with any NaN
    valid_mask = ~np.isnan(data_matrix).any(axis=1)
    data_clean = data_matrix[valid_mask]
    log.info(f"Correlation computed on {data_clean.shape[0]:,} valid grid samples.")

    corr_df = pd.DataFrame(np.corrcoef(data_clean.T), index=available_vars, columns=available_vars)
    return corr_df


def plot_correlation_heatmap(corr_df: pd.DataFrame, year: int, output_path: str):
    """Renders and saves a styled seaborn correlation heatmap."""
    if corr_df.empty:
        log.warning("Empty correlation matrix — skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation"},
    )
    ax.set_title(
        f"ERA5 Variable Correlation Matrix — {year}\n(Nagpur Region, 6-hourly data)",
        fontsize=13, weight="bold", pad=15,
    )
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Correlation heatmap saved: {output_path}")


def export_markdown_report(stats_df: pd.DataFrame, seasonal_df: pd.DataFrame,
                           year: int, output_path: str):
    """Writes a human-readable Markdown statistics report."""
    lines = [
        f"# ERA5 Statistical Report — {year}",
        f"**Region:** Nagpur (17°N–25°N, 74°E–85°E)  ",
        f"**Resolution:** 0.25° (native ERA5)  ",
        f"**Temporal resolution:** 6-hourly  ",
        "",
        "---",
        "",
        "## Descriptive Statistics",
        "",
        stats_df.round(4).to_markdown(index=False),
        "",
        "---",
        "",
        "## Seasonal Statistics",
        "",
    ]

    for season in SEASONS:
        season_data = seasonal_df[seasonal_df["season"] == season] if not seasonal_df.empty else pd.DataFrame()
        if not season_data.empty:
            lines.append(f"### {season}")
            lines.append("")
            lines.append(season_data.drop(columns=["year", "season"]).round(4).to_markdown(index=False))
            lines.append("")

    lines += [
        "---",
        "",
        "## Key Observations",
        "",
        "*(Fill in after reviewing the data)*",
        "",
        "- Temperature range: ...",
        "- Peak monsoon precipitation: ...",
        "- Strongest correlations: ...",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Markdown report saved: {output_path}")


def run_analysis(dataset_path: str, year: int, output_dir: str = "logs"):
    """
    Master function: runs all statistical analyses on a single ERA5 yearly dataset.

    Args:
        dataset_path: Path to the ERA5 NetCDF file (pressure or surface).
        year: The year being analyzed (for labels/filenames).
        output_dir: Directory to save all output files.
    """
    Path(output_dir).mkdir(exist_ok=True)
    ds = load_dataset(dataset_path)

    # 1. Descriptive stats
    stats_df = compute_variable_stats(ds, year)
    stats_csv = os.path.join(output_dir, f"stats_report_{year}.csv")
    stats_df.to_csv(stats_csv, index=False)
    log.info(f"Stats CSV saved: {stats_csv}")

    # 2. Seasonal stats
    seasonal_df = compute_seasonal_stats(ds, year)
    if not seasonal_df.empty:
        seasonal_csv = os.path.join(output_dir, f"seasonal_stats_{year}.csv")
        seasonal_df.to_csv(seasonal_csv, index=False)
        log.info(f"Seasonal CSV saved: {seasonal_csv}")

    # 3. Correlation matrix
    corr_df = compute_correlation_matrix(ds, year)
    if not corr_df.empty:
        corr_csv = os.path.join(output_dir, f"correlation_matrix_{year}.csv")
        corr_df.to_csv(corr_csv)
        log.info(f"Correlation CSV saved: {corr_csv}")

        corr_png = os.path.join(output_dir, f"correlation_matrix_{year}.png")
        plot_correlation_heatmap(corr_df, year, corr_png)

    # 4. Markdown report
    md_path = os.path.join(output_dir, f"stats_report_{year}.md")
    export_markdown_report(stats_df, seasonal_df, year, md_path)

    log.info(f"✅ Statistical analysis complete for {year}. Outputs in '{output_dir}/'")
    return stats_df, seasonal_df, corr_df


def main():
    parser = argparse.ArgumentParser(description="ERA5 Statistical Analysis Report Generator")
    parser.add_argument("dataset", type=str, help="Path to ERA5 NetCDF file")
    parser.add_argument("--year", type=int, required=True, help="Year of the dataset")
    parser.add_argument("--output-dir", type=str, default="logs", help="Output directory (default: logs/)")
    args = parser.parse_args()

    run_analysis(args.dataset, args.year, args.output_dir)


if __name__ == "__main__":
    main()
