"""
sensitivity_analysis.py
========================
Uses Random Forest feature importance and permutation importance to rank
which ERA5 variables are the strongest predictors of precipitation in the
Nagpur region.

Usage:
    python analysis/sensitivity_analysis.py data/ERA5/raw/2015/ --year 2015

Outputs:
    logs/sensitivity_ranking_{year}.csv
    logs/sensitivity_ranking_{year}.png
"""

import os
import sys
import logging
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SensitivityAnalysis")

# ─── Configuration ────────────────────────────────────────────────────────────

# Target variable: what we are trying to predict
TARGET_VARIABLE = "tp"   # total_precipitation (CDS uses 'tp' shortname in NetCDF)
TARGET_ALIASES  = ["tp", "total_precipitation", "precip", "precipitation"]

# Feature variables: what we use as predictors
FEATURE_VARIABLES = [
    "t2m",    # 2m temperature
    "msl",    # mean sea level pressure
    "u10",    # 10m U-wind
    "v10",    # 10m V-wind
    "tcc",    # total cloud cover
    "sp",     # surface pressure
]
FEATURE_ALIASES = {
    "t2m":  ["t2m", "2m_temperature", "temperature_2m"],
    "msl":  ["msl", "mean_sea_level_pressure", "mslp"],
    "u10":  ["u10", "10m_u_component_of_wind", "u_10m"],
    "v10":  ["v10", "10m_v_component_of_wind", "v_10m"],
    "tcc":  ["tcc", "total_cloud_cover", "cloud_cover"],
    "sp":   ["sp",  "surface_pressure"],
}

IMPORTANCE_LABELS = {
    "t2m": "2m Temperature",
    "msl": "Mean Sea Level Pressure",
    "u10": "10m Wind (U-component)",
    "v10": "10m Wind (V-component)",
    "tcc": "Total Cloud Cover",
    "sp":  "Surface Pressure",
}

RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       10,
    "min_samples_leaf": 20,
    "n_jobs":          -1,
    "random_state":    42,
}


def _find_var(ds: xr.Dataset, aliases: list[str]) -> str | None:
    """Returns the first alias found in ds.data_vars, or None."""
    for alias in aliases:
        if alias in ds.data_vars:
            return alias
    return None


def load_surface_dataset(data_path: str) -> xr.Dataset:
    """
    Loads surface-level ERA5 NetCDF files from a directory or a single file.
    Supports both monthly files (era5_surface_2015_01.nc) and combined files.
    """
    path = Path(data_path)
    if path.is_file():
        nc_files = [str(path)]
    else:
        nc_files = sorted(glob.glob(str(path / "era5_surface_*.nc")))
        if not nc_files:
            # Fallback: try any .nc file in the directory
            nc_files = sorted(glob.glob(str(path / "*.nc")))

    if not nc_files:
        raise FileNotFoundError(f"No ERA5 NetCDF files found at: {data_path}")

    log.info(f"Loading {len(nc_files)} file(s) from {data_path}...")
    ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="scipy")
    log.info(f"Dataset loaded. Variables: {list(ds.data_vars)} | Shape: {dict(ds.sizes)}")
    return ds


def build_feature_matrix(ds: xr.Dataset) -> tuple[pd.DataFrame, pd.Series]:
    """
    Flattens the xarray dataset into a 2D feature matrix for scikit-learn.

    Each row = one (time, lat, lon) sample.
    Each column = one surface variable.

    Returns:
        X: pd.DataFrame of features (shape: n_samples × n_features)
        y: pd.Series of target (total_precipitation)
    """
    log.info("Building feature matrix...")

    # Find target variable
    target_key = None
    for alias in TARGET_ALIASES:
        if alias in ds.data_vars:
            target_key = alias
            break
    if target_key is None:
        raise ValueError(
            f"Target variable not found in dataset. "
            f"Available: {list(ds.data_vars)}. Expected one of: {TARGET_ALIASES}"
        )

    # Build feature dict
    feature_data = {}
    available_features = []
    for feat_key in FEATURE_VARIABLES:
        actual_key = _find_var(ds, FEATURE_ALIASES.get(feat_key, [feat_key]))
        if actual_key is not None and actual_key != target_key:
            vals = ds[actual_key].values.flatten()
            feature_data[feat_key] = vals
            available_features.append(feat_key)
            log.info(f"  Feature '{feat_key}' -> variable '{actual_key}' ({len(vals):,} samples)")
        else:
            log.warning(f"  Feature '{feat_key}' not found in dataset. Skipping.")

    # Build target
    target_vals = ds[target_key].values.flatten()

    # Align lengths (safety check)
    min_len = min(len(v) for v in feature_data.values())
    min_len = min(min_len, len(target_vals))

    X = pd.DataFrame({k: v[:min_len] for k, v in feature_data.items()})
    y = pd.Series(target_vals[:min_len], name="total_precipitation")

    # Drop NaN rows
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    log.info(f"Feature matrix: {X.shape[0]:,} valid samples, {X.shape[1]} features")

    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Trains a Random Forest regressor and computes feature importances.

    Returns:
        (rf_model, importance_df, X_test, y_test)
    """
    log.info("Splitting data 80/20 train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features for permutation importance (not needed for RF itself, but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    log.info(f"Training Random Forest ({RF_PARAMS['n_estimators']} trees, max_depth={RF_PARAMS['max_depth']})...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train_scaled, y_train)

    train_score = rf.score(X_train_scaled, y_train)
    test_score  = rf.score(X_test_scaled,  y_test)
    log.info(f"RF R² — Train: {train_score:.4f} | Test: {test_score:.4f}")

    # 1. Gini impurity-based importance (fast, built-in)
    gini_importance = pd.DataFrame({
        "feature":    X.columns,
        "gini_importance": rf.feature_importances_,
    }).sort_values("gini_importance", ascending=False).reset_index(drop=True)

    # 2. Permutation importance (slower but more reliable)
    log.info("Computing permutation importance (this may take ~30s)...")
    perm_result = permutation_importance(
        rf, X_test_scaled, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importance = pd.DataFrame({
        "feature":            X.columns,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std":  perm_result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    # Merge both into one table
    importance_df = gini_importance.merge(perm_importance, on="feature")
    importance_df["rank"] = importance_df["perm_importance_mean"].rank(ascending=False).astype(int)
    importance_df["label"] = importance_df["feature"].map(IMPORTANCE_LABELS)
    importance_df = importance_df.sort_values("rank")

    log.info("\nFeature Importance Ranking (by permutation importance):")
    for _, row in importance_df.iterrows():
        log.info(f"  #{row['rank']:>2}  {row['label']:<35} Gini: {row['gini_importance']:.4f} | "
                 f"Perm: {row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f}")

    return rf, importance_df, X_test_scaled, y_test


def plot_importance(importance_df: pd.DataFrame, year: int, output_path: str):
    """Saves a horizontal bar chart of feature importances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"ERA5 Feature Importance for Precipitation Prediction — {year}\n"
        f"(Nagpur Region, Random Forest n=200 trees)",
        fontsize=13, weight="bold"
    )

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df)))[::-1]

    # Left: Gini importance
    ax1 = axes[0]
    bars = ax1.barh(importance_df["label"][::-1], importance_df["gini_importance"][::-1], color=colors)
    ax1.set_xlabel("Gini Importance (MDI)", fontsize=10)
    ax1.set_title("Mean Decrease in Impurity", fontsize=11)
    ax1.set_xlim(0, importance_df["gini_importance"].max() * 1.15)
    for bar, val in zip(bars, importance_df["gini_importance"][::-1]):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)

    # Right: Permutation importance
    ax2 = axes[1]
    bars2 = ax2.barh(
        importance_df["label"][::-1],
        importance_df["perm_importance_mean"][::-1],
        xerr=importance_df["perm_importance_std"][::-1],
        color=colors, capsize=4, error_kw={"elinewidth": 1.2}
    )
    ax2.set_xlabel("Permutation Importance (mean ± std)", fontsize=10)
    ax2.set_title("Permutation Importance (test set)", fontsize=11)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Importance chart saved: {output_path}")


def run_sensitivity_analysis(data_path: str, year: int, output_dir: str = "logs"):
    """
    Master function: loads ERA5 surface data, trains RF, exports importance ranking.

    Args:
        data_path: Path to ERA5 surface NetCDF file(s) directory.
        year: Year being analyzed (for labels/filenames).
        output_dir: Directory to save outputs.

    Returns:
        importance_df: pd.DataFrame with ranked feature importance.
    """
    Path(output_dir).mkdir(exist_ok=True)

    ds = load_surface_dataset(data_path)
    X, y = build_feature_matrix(ds)

    if y.sum() == 0 or y.std() < 1e-10:
        log.warning("Target variable (precipitation) has near-zero variance. "
                    "Sensitivity analysis may not be meaningful.")

    rf, importance_df, X_test, y_test = train_random_forest(X, y)

    # Export CSV
    csv_path = os.path.join(output_dir, f"sensitivity_ranking_{year}.csv")
    importance_df.to_csv(csv_path, index=False)
    log.info(f"Ranking CSV saved: {csv_path}")

    # Export chart
    png_path = os.path.join(output_dir, f"sensitivity_ranking_{year}.png")
    plot_importance(importance_df, year, png_path)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Variable Importance Ranking for Precipitation — {year}")
    print(f"{'='*60}")
    for _, row in importance_df.iterrows():
        bar = "#" * int(row["perm_importance_mean"] * 100 / importance_df["perm_importance_mean"].max() * 20)
        print(f"  #{row['rank']:>2}  {row['label']:<35} [{bar:<20}] {row['perm_importance_mean']:.4f}")
    print(f"{'='*60}\n")

    return importance_df


def main():
    parser = argparse.ArgumentParser(
        description="ERA5 Sensitivity Analysis — Feature Importance for Precipitation"
    )
    parser.add_argument("data_path", type=str,
                        help="Path to ERA5 surface NetCDF file or directory of monthly files")
    parser.add_argument("--year", type=int, required=True, help="Year of the dataset")
    parser.add_argument("--output-dir", type=str, default="logs",
                        help="Directory to save outputs (default: logs/)")
    args = parser.parse_args()

    run_sensitivity_analysis(args.data_path, args.year, args.output_dir)


if __name__ == "__main__":
    main()
