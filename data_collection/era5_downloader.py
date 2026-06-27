"""
era5_downloader.py
==================
Automated ERA5 reanalysis data downloader via the Copernicus Climate Data Store (CDS) API.

Setup (New CDS API — post-2024 migration):
    1. Register at https://cds.climate.copernicus.eu/
    2. Accept the ERA5 licence at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    3. Credentials file is at: C:\\Users\\Asus\\.cdsapirc

        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-API-KEY-UUID>

    4. pip install cdsapi

Usage:
    # Test connection (downloads 1 day, ~2 MB)
    python data_collection/era5_downloader.py --test --region nagpur

    # Estimate storage before downloading
    python data_collection/era5_downloader.py --estimate --start-year 2015 --end-year 2018

    # Download all of 2015 for Nagpur region
    python data_collection/era5_downloader.py --year 2015

    # Download a specific month
    python data_collection/era5_downloader.py --year 2015 --month 1

    # Download 2015-2018
    python data_collection/era5_downloader.py --start-year 2015 --end-year 2018
"""

import os
import sys
import time
import logging
import argparse
import calendar
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent dir so we can import production_pipeline.utils if available
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/era5_download.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("ERA5Downloader")

# ─── Geographic Configuration ─────────────────────────────────────────────────
# Nagpur region bounding box [North, West, South, East] in degrees
NAGPUR_REGION = [25.0, 74.0, 17.0, 85.0]

# For full India coverage
INDIA_REGION = [38.0, 66.0, 6.0, 100.0]

# For global coverage (no area restriction)
GLOBAL_REGION = None

# ─── ERA5 Variable Configuration ──────────────────────────────────────────────
# Pressure-level variables (require 'reanalysis-era5-pressure-levels')
PRESSURE_VARIABLES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "vertical_velocity",
]

# Surface/single-level variables (require 'reanalysis-era5-single-levels')
SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
    "total_cloud_cover",
    "surface_pressure",
    "land_sea_mask",
]

# The 13 pressure levels used by GraphCast (hPa)
GRAPHCAST_PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250",
    "300", "400", "500", "600", "700",
    "850", "925", "1000",
]

# 6-hourly timesteps (GraphCast operates at 6h resolution)
TIMESTEPS_6H = ["00:00", "06:00", "12:00", "18:00"]

# ─── CDS Dataset Names (new API uses different names) ─────────────────────────
# New CDS API (post-2024): dataset names changed
# Old API v2:  "reanalysis-era5-pressure-levels" / "reanalysis-era5-single-levels"
# New API:     "reanalysis-era5-pressure-levels" (same) but request format differs
CDS_PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
CDS_SURFACE_DATASET  = "reanalysis-era5-single-levels"

# ─── Output Directory ─────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ERA5_RAW_DIR = DATA_DIR / "ERA5" / "raw"
ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─── Retry Decorator ──────────────────────────────────────────────────────────
def retry_download(max_retries: int = 5, backoff: float = 30.0):
    """Decorator that retries CDS API calls with exponential backoff.
    CDS API calls can fail transiently due to server queuing; retries are essential.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries:
                        log.error(f"Download failed after {max_retries} attempts: {exc}")
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    log.warning(f"Attempt {attempt} failed: {exc}. Retrying in {wait:.0f}s...")
                    time.sleep(wait)
        return wrapper
    return decorator


# ─── Core Downloader Class ────────────────────────────────────────────────────
class ERA5Downloader:
    """
    Downloads ERA5 reanalysis data from the Copernicus CDS API.
    
    Handles:
    - Monthly chunked downloads (avoids CDS 100,000-request queue limits)
    - Separate pressure-level and surface-level requests
    - Automatic resume (skips already-downloaded files)
    - File validation (size check + xarray open check)
    """

    def __init__(self, region: Optional[List[float]] = None):
        """
        Args:
            region: [N, W, S, E] bounding box in degrees.
                    Defaults to Nagpur region (17°N–25°N, 74°E–85°E).
        """
        self.region = region or NAGPUR_REGION
        self._client = None  # Lazy-initialize to avoid import error if cdsapi not installed

    @property
    def client(self):
        if self._client is None:
            try:
                import cdsapi
                # The new CDS API (cdsapi >= 0.7) reads ~/.cdsapirc automatically.
                # url: https://cds.climate.copernicus.eu/api
                # key: <UUID>  (no UID: prefix in new API)
                self._client = cdsapi.Client()
                try:
                    from importlib.metadata import version as pkg_version
                    ver = pkg_version("cdsapi")
                except Exception:
                    ver = getattr(cdsapi, "__version__", "unknown")
                log.info(f"CDS API client initialized. Library version: {ver}")
            except ImportError:
                log.error("cdsapi not installed. Run: pip install cdsapi")
                raise
            except Exception as exc:
                log.error(f"CDS API init failed. Check C:\\Users\\Asus\\.cdsapirc: {exc}")
                raise
        return self._client

    def _output_path(self, year: int, month: int, level_type: str) -> Path:
        """Returns the output filepath for a given year/month/level_type download."""
        year_dir = ERA5_RAW_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        return year_dir / f"era5_{level_type}_{year}_{month:02d}.nc"

    def _validate_file(self, filepath: Path, min_size_mb: float = 1.0) -> bool:
        """Checks that the downloaded file exists, is large enough, and opens in xarray.
        
        The new CDS API (post-2024) returns NetCDF4 format files, not NetCDF3.
        We try netcdf4 engine first, then scipy (NetCDF3) as fallback.
        On Windows, we must close the file handle BEFORE attempting to delete.
        """
        if not filepath.exists():
            return False

        # Check if the file is a zip archive (sometimes returned by new CDS API)
        import zipfile
        import shutil
        if zipfile.is_zipfile(filepath):
            log.info(f"File {filepath.name} is a zip archive. Extracting...")
            temp_dir = filepath.parent / f"temp_{filepath.stem}_{int(time.time())}"
            temp_dir.mkdir(exist_ok=True, parents=True)
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                nc_files = list(temp_dir.glob("*.nc"))
                if nc_files:
                    import xarray as xr
                    datasets = []
                    log.info(f"  Found {len(nc_files)} .nc files in zip archive: {[f.name for f in nc_files]}")
                    for f in nc_files:
                        ds_loaded = None
                        for engine in ("netcdf4", "scipy"):
                            try:
                                ds_loaded = xr.open_dataset(f, engine=engine).load()
                                datasets.append(ds_loaded)
                                break
                            except Exception:
                                if ds_loaded is not None:
                                    ds_loaded.close()
                                continue
                    
                    if datasets:
                        merged_ds = xr.merge(datasets)
                        try:
                            filepath.unlink()
                        except PermissionError:
                            log.warning(f"Could not delete zip file {filepath.name} (locked). Overwriting instead.")
                        merged_ds.to_netcdf(filepath)
                        merged_ds.close()
                        for ds_obj in datasets:
                            ds_obj.close()
                        log.info(f"Successfully merged {len(nc_files)} files into {filepath.name}")
                    else:
                        log.warning(f"Failed to read any of the extracted .nc files from zip archive {filepath.name}")
                else:
                    log.warning(f"No .nc files found inside zip archive {filepath.name}")
            except Exception as e:
                log.error(f"Error handling zip file {filepath.name}: {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        size_mb = filepath.stat().st_size / (1024 ** 2)
        if size_mb < min_size_mb:
            log.warning(f"File {filepath.name} is suspiciously small ({size_mb:.2f} MB). Deleting.")
            try:
                filepath.unlink()
            except PermissionError:
                log.warning(f"Could not delete {filepath.name} (still locked). Will retry next run.")
            return False
        ds = None
        try:
            import xarray as xr
            # Try NetCDF4 first (new CDS API), then scipy (NetCDF3 legacy)
            for engine in ("netcdf4", "scipy", "h5netcdf"):
                try:
                    ds = xr.open_dataset(filepath, engine=engine)
                    ds.close()
                    ds = None
                    log.info(f"Validated: {filepath.name} ({size_mb:.1f} MB, engine={engine})")
                    return True
                except Exception:
                    if ds is not None:
                        ds.close()
                        ds = None
                    continue
            # All engines failed
            raise ValueError(f"No xarray engine could open {filepath.name}")
        except Exception as exc:
            if ds is not None:
                ds.close()
            log.warning(f"File {filepath.name} failed validation: {exc}. Deleting.")
            try:
                filepath.unlink()
            except PermissionError:
                log.warning(f"Could not delete {filepath.name} (still locked by OS). Will retry on next run.")
            return False

    def _build_days_list(self, year: int, month: int) -> List[str]:
        """Returns zero-padded day strings for the given month."""
        n_days = calendar.monthrange(year, month)[1]
        return [f"{d:02d}" for d in range(1, n_days + 1)]

    @retry_download(max_retries=5, backoff=30.0)
    def _download_pressure_levels(self, year: int, month: int) -> Path:
        """Downloads 3D pressure-level ERA5 data for a given year/month."""
        output_path = self._output_path(year, month, "pressure")
        if self._validate_file(output_path):
            log.info(f"Skipping {output_path.name} — already downloaded and valid.")
            return output_path

        log.info(f"Requesting ERA5 pressure-level data: {year}-{month:02d} ...")
        request = {
            "product_type": "reanalysis",
            "variable": PRESSURE_VARIABLES,
            "pressure_level": GRAPHCAST_PRESSURE_LEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": self._build_days_list(year, month),
            "time": TIMESTEPS_6H,
            "data_format": "netcdf",    # New CDS API uses 'data_format' not 'format'
            "download_format": "unarchived",
        }
        if self.region:
            request["area"] = self.region  # [N, W, S, E]

        self.client.retrieve(CDS_PRESSURE_DATASET, request, str(output_path))
        self._validate_file(output_path)
        log.info(f"Pressure-level download complete: {output_path.name}")
        return output_path

    @retry_download(max_retries=5, backoff=30.0)
    def _download_surface_levels(self, year: int, month: int) -> Path:
        """Downloads 2D surface-level ERA5 data for a given year/month."""
        output_path = self._output_path(year, month, "surface")
        if self._validate_file(output_path):
            log.info(f"Skipping {output_path.name} — already downloaded and valid.")
            return output_path

        log.info(f"Requesting ERA5 surface-level data: {year}-{month:02d} ...")
        request = {
            "product_type": "reanalysis",
            "variable": SURFACE_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": self._build_days_list(year, month),
            "time": TIMESTEPS_6H,
            "data_format": "netcdf",    # New CDS API parameter
            "download_format": "unarchived",
        }
        if self.region:
            request["area"] = self.region

        self.client.retrieve(CDS_SURFACE_DATASET, request, str(output_path))
        self._validate_file(output_path)
        log.info(f"Surface download complete: {output_path.name}")
        return output_path

    def download_month(self, year: int, month: int) -> dict:
        """Downloads both pressure and surface ERA5 data for a single month.

        Returns:
            dict with keys 'pressure_path' and 'surface_path'
        """
        log.info(f"─── Downloading ERA5: {year}-{month:02d} ───")
        pressure_path = self._download_pressure_levels(year, month)
        surface_path = self._download_surface_levels(year, month)
        return {
            "pressure_path": pressure_path,
            "surface_path": surface_path,
        }

    def download_year(self, year: int, months: Optional[List[int]] = None) -> List[dict]:
        """Downloads all 12 months of ERA5 data for a given year.

        Args:
            year: The year to download (e.g., 2015).
            months: Optional list of months (1–12). Defaults to all 12 months.

        Returns:
            List of dicts with 'pressure_path' and 'surface_path' per month.
        """
        months = months or list(range(1, 13))
        log.info(f"═══ Starting ERA5 download for year {year} ({len(months)} months) ═══")
        log.info(f"Region: {self.region} [N={self.region[0]}, W={self.region[1]}, S={self.region[2]}, E={self.region[3]}]")
        log.info(f"Pressure levels: {GRAPHCAST_PRESSURE_LEVELS}")
        log.info(f"Timesteps: {TIMESTEPS_6H}")

        results = []
        for month in months:
            try:
                result = self.download_month(year, month)
                results.append(result)
            except Exception as exc:
                log.error(f"Failed to download {year}-{month:02d}: {exc}")
                results.append({"pressure_path": None, "surface_path": None, "error": str(exc)})

        # Summary
        success = sum(1 for r in results if r.get("pressure_path") and r.get("surface_path"))
        log.info(f"═══ Year {year} download complete: {success}/{len(months)} months succeeded ═══")
        return results

    def download_range(self, start_year: int, end_year: int) -> dict:
        """Downloads ERA5 data for a range of years.

        Args:
            start_year: First year to download (e.g., 2015).
            end_year: Last year to download, inclusive (e.g., 2018).

        Returns:
            dict mapping year → list of monthly results.
        """
        log.info(f"Starting multi-year ERA5 download: {start_year} → {end_year}")
        all_results = {}
        for year in range(start_year, end_year + 1):
            all_results[year] = self.download_year(year)
        log.info(f"Multi-year download complete: {start_year}–{end_year}")
        return all_results

    def test_connection(self) -> bool:
        """Downloads a tiny ERA5 slice to verify CDS API connectivity and credentials."""
        log.info("Testing CDS API connection (downloading 2 timesteps of ERA5 2015-01-01)...")
        test_path = ERA5_RAW_DIR / "test_era5_connection.nc"
        try:
            self.client.retrieve(
                CDS_SURFACE_DATASET,
                {
                    "product_type": "reanalysis",
                    "variable": ["2m_temperature", "mean_sea_level_pressure"],
                    "year": "2015",
                    "month": "01",
                    "day": "01",
                    "time": ["00:00", "12:00"],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": self.region,  # Small Nagpur region = fast download
                },
                str(test_path),
            )
            import xarray as xr
            ds = xr.open_dataset(test_path)
            lat_dim = ds.sizes.get("latitude", ds.sizes.get("lat", "?"))
            lon_dim = ds.sizes.get("longitude", ds.sizes.get("lon", "?"))
            size_mb = test_path.stat().st_size / (1024**2)
            log.info(f"\n{'='*55}")
            log.info(f"  ✅ CDS API test PASSED")
            log.info(f"  Variables: {list(ds.data_vars)}")
            log.info(f"  Grid: lat={lat_dim}, lon={lon_dim}")
            log.info(f"  File size: {size_mb:.2f} MB")
            log.info(f"{'='*55}\n")
            ds.close()
            return True
        except Exception as exc:
            log.error(f"\n{'='*55}")
            log.error(f"  ❌ CDS API test FAILED")
            log.error(f"  Error: {exc}")
            log.error(f"  Check: C:\\Users\\Asus\\.cdsapirc exists and key is correct")
            log.error(f"{'='*55}\n")
            return False


# ─── Storage Estimator ────────────────────────────────────────────────────────
def estimate_storage_gb(
    n_years: int,
    region: List[float],
    resolution_deg: float = 0.25,
) -> dict:
    """
    Estimates the storage required for ERA5 downloads.
    
    Args:
        n_years: Number of years to download.
        region: [N, W, S, E] bounding box.
        resolution_deg: ERA5 native resolution (0.25° default).
    
    Returns:
        dict with raw_gb and notes.
    """
    lat_cells = int((region[0] - region[2]) / resolution_deg)
    lon_cells = int((region[1] - region[3]) / resolution_deg * -1 + 
                    (region[3] - region[1]) / resolution_deg)
    # Simpler: use bounding box area
    lat_span = abs(region[0] - region[2])
    lon_span = abs(region[3] - region[1])
    grid_cells = int((lat_span / resolution_deg) * (lon_span / resolution_deg))

    # 6h timesteps × days per year
    timesteps_per_year = 4 * 365.25

    # Pressure: n_vars × n_levels × grid × time × float32 (4 bytes)
    n_pressure_vars = len(PRESSURE_VARIABLES)
    n_levels = len(GRAPHCAST_PRESSURE_LEVELS)
    pressure_bytes = n_pressure_vars * n_levels * grid_cells * timesteps_per_year * 4

    # Surface: n_vars × grid × time × float32
    n_surface_vars = len(SURFACE_VARIABLES)
    surface_bytes = n_surface_vars * grid_cells * timesteps_per_year * 4

    total_bytes_per_year = pressure_bytes + surface_bytes
    # NetCDF compression typically achieves 3–5× reduction
    compressed_bytes_per_year = total_bytes_per_year / 4.0

    total_raw_gb = (compressed_bytes_per_year * n_years) / (1024 ** 3)

    return {
        "grid_cells": grid_cells,
        "timesteps_per_year": int(timesteps_per_year),
        "raw_gb_per_year": round(total_raw_gb / n_years, 1),
        "raw_gb_total": round(total_raw_gb, 1),
        "notes": f"{lat_span}°×{lon_span}° region at {resolution_deg}° resolution"
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ERA5 CDS API Downloader for GraphCast Weather Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year",       type=int, help="Download a single year (e.g., 2015)")
    parser.add_argument("--month",      type=int, help="Download a single month (1-12). Used with --year.")
    parser.add_argument("--day",        type=int, help="Day number for test download only.")
    parser.add_argument("--start-year", type=int, help="Start year for multi-year download")
    parser.add_argument("--end-year",   type=int, help="End year for multi-year download (inclusive)")
    parser.add_argument("--region",     type=str, default="nagpur",
                        choices=["nagpur", "india", "global"],
                        help="Geographic region to download (default: nagpur)")
    parser.add_argument("--test",       action="store_true",
                        help="Run connection test (downloads 1 day only)")
    parser.add_argument("--estimate",   action="store_true",
                        help="Estimate storage requirements only, do not download")
    args = parser.parse_args()

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    # Select region
    region_map = {
        "nagpur": NAGPUR_REGION,
        "india":  INDIA_REGION,
        "global": None,
    }
    region = region_map[args.region]

    downloader = ERA5Downloader(region=region)

    if args.estimate:
        n_years = 1
        if args.start_year and args.end_year:
            n_years = args.end_year - args.start_year + 1
        estimate = estimate_storage_gb(n_years, region or [-90, 0, 90, 360])
        print("\n[Storage Estimate]")
        print(f"   Region:              {estimate['notes']}")
        print(f"   Grid cells:          {estimate['grid_cells']:,}")
        print(f"   Timesteps/year:      {estimate['timesteps_per_year']:,}")
        print(f"   Est. size per year:  {estimate['raw_gb_per_year']} GB")
        print(f"   Est. total ({n_years} yr{'s' if n_years > 1 else ''}):  {estimate['raw_gb_total']} GB")
        print(f"   (NetCDF internal compression ~3-5x; actual downloads smaller than uncompressed)\n")
        return

    if args.test:
        success = downloader.test_connection()
        sys.exit(0 if success else 1)

    if args.year and args.month:
        downloader.download_month(args.year, args.month)
    elif args.year:
        downloader.download_year(args.year)
    elif args.start_year and args.end_year:
        downloader.download_range(args.start_year, args.end_year)
    else:
        parser.print_help()
        print("\n⚠️  Please specify --year, --year + --month, or --start-year + --end-year")
        sys.exit(1)


if __name__ == "__main__":
    main()
