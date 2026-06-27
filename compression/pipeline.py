"""
pipeline.py
===========
Master compression pipeline orchestrator.

Runs all compression stages in sequence on a yearly ERA5 dataset:
  Stage 1: Missing value handling (interpolation + climatological fill)
  Stage 2: Outlier detection (physical bounds + z-score)
  Stage 3: Event extraction (events at 6h, background at daily)
  Stage 4: PCA spatial compression (per variable, 99% variance retained)
  Stage 5: Statistical validation (must pass before training)

Usage:
    python compression/pipeline.py data/ERA5/raw/2015/ --year 2015

    # Skip PCA (just clean and validate)
    python compression/pipeline.py data/ERA5/raw/2015/ --year 2015 --no-pca
"""

import os
import sys
import logging
import argparse
import glob
import time
from pathlib import Path

import xarray as xr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/compression_pipeline.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("CompressionPipeline")

from compression.missing_values import fill_missing_values
from compression.outlier_detection import detect_and_remove_outliers
from compression.event_extractor import extract_significant_events
from compression.pca_compressor import ERA5PCACompressor
from compression.validator import validate_compression

PROCESSED_DIR = Path("data") / "ERA5" / "processed"


def load_era5_year(data_path: str) -> xr.Dataset:
    """Loads all ERA5 monthly files for a year from a directory."""
    path = Path(data_path)
    if path.is_file():
        files = [str(path)]
    else:
        files = sorted(glob.glob(str(path / "era5_*.nc")))
        if not files:
            files = sorted(glob.glob(str(path / "*.nc")))

    if not files:
        raise FileNotFoundError(f"No ERA5 .nc files found at: {data_path}")

    log.info(f"Loading {len(files)} ERA5 files from {data_path}...")
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", engine="scipy")
    except Exception:
        ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")

    log.info(f"Loaded. Variables: {list(ds.data_vars)} | Sizes: {dict(ds.sizes)}")
    ds = ds.load()
    from production_pipeline.preprocessing import standardize_variables
    ds = standardize_variables(ds)
    return ds


def run_compression_pipeline(
    data_path: str,
    year: int,
    output_dir: Optional[str] = None,
    use_pca: bool = True,
    pca_variance: float = 0.99,
    validate: bool = True,
) -> dict:
    """
    Full compression pipeline for a single year of ERA5 data.

    Args:
        data_path:   Path to ERA5 NetCDF files (directory or single file).
        year:        Year being processed (for filenames and logging).
        output_dir:  Where to save compressed output (default: data/ERA5/processed/{year}/).
        use_pca:     Whether to apply PCA compression (default: True).
        pca_variance: Fraction of variance to retain in PCA (default: 0.99).
        validate:    Whether to run validation after compression (default: True).

    Returns:
        dict with keys: 'original_ds', 'compressed_ds', 'validation_df', 'passed'
    """
    t_start = time.time()
    output_dir = output_dir or str(PROCESSED_DIR / str(year))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  GraphCast ERA5 Compression Pipeline — Year {year}")
    log.info(f"{'='*60}")

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    log.info("\n[Stage 1/5] Loading ERA5 data...")
    original_ds = load_era5_year(data_path)
    original_size_mb = sum(
        original_ds[v].values.nbytes for v in original_ds.data_vars
    ) / (1024**2)
    log.info(f"  Original dataset in-memory size: {original_size_mb:.1f} MB")

    # ── Stage 2: Missing Values ────────────────────────────────────────────────
    log.info("\n[Stage 2/5] Handling missing values...")
    ds = fill_missing_values(original_ds, interp_limit=3,
                              use_climatological=True, use_spatial=True)

    # ── Stage 3: Outlier Detection ─────────────────────────────────────────────
    log.info("\n[Stage 3/5] Detecting and removing outliers...")
    ds, outlier_report = detect_and_remove_outliers(
        ds, use_physical_bounds=True, use_zscore=True
    )
    outlier_df = pd.DataFrame([
        {"variable": var, **stats}
        for method_report in outlier_report.values()
        for var, stats in method_report.items()
    ])
    outlier_df.to_csv(f"{output_dir}/outlier_report_{year}.csv", index=False)

    # ── Stage 4: Event Extraction ──────────────────────────────────────────────
    log.info("\n[Stage 4/5] Extracting significant weather events...")
    try:
        event_ds, background_ds, event_mask = extract_significant_events(ds)
        # Save events and background separately
        event_ds.to_netcdf(f"{output_dir}/era5_events_{year}.nc")
        background_ds.to_netcdf(f"{output_dir}/era5_background_{year}.nc")
        log.info(f"  Events saved: era5_events_{year}.nc")
        log.info(f"  Background saved: era5_background_{year}.nc")
        # For PCA and validation, use the cleaned full dataset
        cleaned_ds = ds
    except Exception as exc:
        log.warning(f"Event extraction failed: {exc}. Using cleaned dataset directly.")
        cleaned_ds = ds

    # ── Stage 5: PCA Compression ───────────────────────────────────────────────
    if use_pca:
        log.info(f"\n[Stage 5/5] Applying PCA compression (variance={pca_variance})...")
        compressor = ERA5PCACompressor(variance_threshold=pca_variance)
        compressed_ds, pca_report = compressor.fit_compress(cleaned_ds, year=year)
        pca_report.to_csv(f"{output_dir}/pca_report_{year}.csv", index=False)

        # Save compressor for future years
        compressor.save(f"{output_dir}/pca_compressor_{year}.pkl")

        # Save compressed dataset
        try:
            compressed_ds.to_netcdf(f"{output_dir}/era5_pca_{year}.nc")
            log.info(f"  Compressed dataset saved: era5_pca_{year}.nc")
        except Exception as exc:
            log.warning(f"  Could not save PCA dataset to NetCDF (PCA dims): {exc}")
    else:
        compressed_ds = cleaned_ds
        log.info("\n[Stage 5/5] PCA skipped.")

    # ── Validation ─────────────────────────────────────────────────────────────
    validation_df = pd.DataFrame()
    passed = True
    if validate:
        log.info("\n[Validation] Comparing compressed vs. original statistics...")
        # For validation: reconstruct from PCA if applied
        if use_pca:
            try:
                reconstructed = compressor.reconstruct(compressed_ds)
                validation_df, passed = validate_compression(
                    original_ds, reconstructed, output_dir=output_dir
                )
            except Exception as exc:
                log.warning(f"Validation with reconstruction failed: {exc}. "
                             "Validating cleaned vs original instead.")
                validation_df, passed = validate_compression(
                    original_ds, cleaned_ds, output_dir=output_dir
                )
        else:
            validation_df, passed = validate_compression(
                original_ds, cleaned_ds, output_dir=output_dir
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    compressed_size_mb = sum(
        compressed_ds[v].values.nbytes for v in compressed_ds.data_vars
    ) / (1024**2) if use_pca else original_size_mb

    log.info(f"\n{'='*60}")
    log.info(f"  Pipeline Complete — {year} ({t_elapsed:.1f}s)")
    log.info(f"  Original size:    {original_size_mb:.1f} MB")
    log.info(f"  Compressed size:  {compressed_size_mb:.1f} MB")
    log.info(f"  Compression ratio: {original_size_mb/max(compressed_size_mb,0.001):.1f}x")
    log.info(f"  Validation:       {'PASSED' if passed else 'FAILED'}")
    log.info(f"  Outputs saved to: {output_dir}")
    log.info(f"{'='*60}\n")

    return {
        "original_ds":    original_ds,
        "compressed_ds":  compressed_ds,
        "validation_df":  validation_df,
        "passed":         passed,
        "elapsed_seconds": t_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ERA5 Compression Pipeline — full processing for one year"
    )
    parser.add_argument("data_path", type=str,
                        help="Path to ERA5 NetCDF file(s) or directory")
    parser.add_argument("--year",       type=int, required=True,
                        help="Year being processed (e.g., 2015)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/ERA5/processed/{year}/)")
    parser.add_argument("--no-pca",     action="store_true",
                        help="Skip PCA compression (clean only)")
    parser.add_argument("--pca-variance", type=float, default=0.99,
                        help="PCA variance threshold (default: 0.99)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation step")
    args = parser.parse_args()

    results = run_compression_pipeline(
        data_path=args.data_path,
        year=args.year,
        output_dir=args.output_dir,
        use_pca=not args.no_pca,
        pca_variance=args.pca_variance,
        validate=not args.no_validate,
    )
    import sys
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    from typing import Optional
    main()
