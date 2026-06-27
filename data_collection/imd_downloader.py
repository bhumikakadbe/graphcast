"""
imd_downloader.py
=================
Downloads IMD (India Meteorological Department) gridded rainfall data
using the imdlib Python package.

Data Source:
    IMD provides gridded daily rainfall data at 0.25° resolution.
    imdlib fetches it from IMD's public FTP server.

Variables available:
    - 'rain'  : Daily rainfall (mm/day), 0.25° grid, 1901–present
    - 'tmax'  : Maximum temperature (°C), 1° grid, 1951–present
    - 'tmin'  : Minimum temperature (°C), 1° grid, 1951–present

Coverage:
    - rain : 6.5°N–38.5°N, 66.5°E–100°E
    - tmax/tmin : 7.5°N–37.5°N, 67.5°E–99.5°E

Usage:
    # Download rainfall for 2015 only
    python data_collection/imd_downloader.py --var rain --start-year 2015 --end-year 2015

    # Download rainfall + temperature for 2015–2017
    python data_collection/imd_downloader.py --all --start-year 2015 --end-year 2017

    # Show what will be downloaded (dry run)
    python data_collection/imd_downloader.py --var rain --start-year 2015 --end-year 2015 --dry-run
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/imd_download.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("IMDDownloader")

# ─── Configuration ────────────────────────────────────────────────────────────
# Nagpur region bounding box
NAGPUR_BOUNDS = {
    "lat_min": 17.0,
    "lat_max": 25.0,
    "lon_min": 74.0,
    "lon_max": 85.0,
}

DATA_DIR = Path("data/IMD/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = {
    "rain": {
        "description": "Daily rainfall (mm/day)",
        "resolution": "0.25°",
        "start_year_available": 1901,
    },
    "tmax": {
        "description": "Daily max temperature (°C)",
        "resolution": "1°",
        "start_year_available": 1951,
    },
    "tmin": {
        "description": "Daily min temperature (°C)",
        "resolution": "1°",
        "start_year_available": 1951,
    },
}


class IMDDownloader:
    """
    Downloads IMD gridded climate data using imdlib and crops
    to the Nagpur region.
    """

    def __init__(self, output_dir: str = "data/IMD/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_imdlib(self) -> bool:
        """Verify imdlib is installed."""
        try:
            import imdlib  # noqa: F401
            log.info("imdlib is available ✅")
            return True
        except ImportError:
            log.error("imdlib not installed. Run: pip install imdlib")
            return False

    def download(self, variable: str, start_year: int, end_year: int, dry_run: bool = False):
        """
        Download IMD data for a variable over a year range.

        Args:
            variable: 'rain', 'tmax', or 'tmin'
            start_year: First year to download (e.g. 2015)
            end_year: Last year to download (inclusive)
            dry_run: If True, only show what would be downloaded
        """
        if variable not in VARIABLES:
            raise ValueError(f"Unknown variable '{variable}'. Choose from: {list(VARIABLES.keys())}")

        meta = VARIABLES[variable]
        log.info(f"Variable: {variable} — {meta['description']} ({meta['resolution']} resolution)")
        log.info(f"Years: {start_year}–{end_year}")
        log.info(f"Region: Nagpur ({NAGPUR_BOUNDS['lat_min']}°N–{NAGPUR_BOUNDS['lat_max']}°N, "
                 f"{NAGPUR_BOUNDS['lon_min']}°E–{NAGPUR_BOUNDS['lon_max']}°E)")

        if dry_run:
            log.info("[DRY RUN] Would download the above. Run without --dry-run to proceed.")
            return

        if not self.check_imdlib():
            sys.exit(1)

        import imdlib

        year_dir = self.output_dir / str(start_year)
        year_dir.mkdir(parents=True, exist_ok=True)

        for year in range(start_year, end_year + 1):
            out_dir = self.output_dir / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"Downloading {variable} for {year}...")
            try:
                data = imdlib.get_data(
                    variable,
                    year,
                    year,
                    fn_format="yearwise",
                    file_dir=str(out_dir),
                )
                log.info(f"  ✅ {variable} {year} downloaded to {out_dir}")

                # Crop to Nagpur region and save as NetCDF
                self._crop_and_save(data, variable, year, out_dir)

            except Exception as e:
                log.error(f"  ❌ Failed to download {variable} {year}: {e}")
                continue

        log.info("IMD download complete!")

    def _crop_and_save(self, data, variable: str, year: int, out_dir: Path):
        """Crop IMD data to Nagpur region and save as NetCDF."""
        try:
            import xarray as xr

            ds = data.get_xarray()

            # Crop to Nagpur bounding box
            ds_cropped = ds.sel(
                lat=slice(NAGPUR_BOUNDS["lat_min"], NAGPUR_BOUNDS["lat_max"]),
                lon=slice(NAGPUR_BOUNDS["lon_min"], NAGPUR_BOUNDS["lon_max"]),
            )

            out_path = out_dir / f"imd_{variable}_{year}_nagpur.nc"
            ds_cropped.to_netcdf(out_path)
            log.info(f"  ✅ Cropped NetCDF saved: {out_path} "
                     f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

        except Exception as e:
            log.warning(f"  ⚠️ Could not crop/save NetCDF for {variable} {year}: {e}")

    def show_summary(self):
        """Show what IMD data is already downloaded."""
        log.info("\n── IMD Data Summary ──────────────────────────────")
        total = 0
        for year_dir in sorted(self.output_dir.iterdir()):
            if year_dir.is_dir():
                files = list(year_dir.glob("*.nc")) + list(year_dir.glob("*.grd")) + list(year_dir.glob("*.bin"))
                size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
                total += size_mb
                log.info(f"  {year_dir.name}: {len(files)} files, {size_mb:.1f} MB")
                for f in files:
                    log.info(f"    - {f.name}")
        log.info(f"  Total: {total:.1f} MB")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download IMD gridded climate data for Nagpur region"
    )
    parser.add_argument("--var", choices=["rain", "tmax", "tmin"],
                        help="Variable to download")
    parser.add_argument("--all", action="store_true",
                        help="Download rain + tmax + tmin")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument("--output-dir", default="data/IMD/raw",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--summary", action="store_true",
                        help="Show already downloaded data summary")
    args = parser.parse_args()

    downloader = IMDDownloader(output_dir=args.output_dir)

    if args.summary:
        downloader.show_summary()
        return

    if args.all:
        variables = ["rain", "tmax", "tmin"]
    elif args.var:
        variables = [args.var]
    else:
        parser.print_help()
        log.error("\n❌ Please specify --var <rain|tmax|tmin> or --all")
        sys.exit(1)

    for var in variables:
        downloader.download(
            variable=var,
            start_year=args.start_year,
            end_year=args.end_year,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
