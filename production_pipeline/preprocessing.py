# preprocessing.py
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple
from graphcast import data_utils
from production_pipeline.utils import logger

def align_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Aligns latitude, longitude, and levels to match GraphCast assumptions.
    
    GraphCast expects:
    - Latitudes ordered North-to-South (90 down to -90).
    - Longitudes ordered West-to-East [0, 360).
    - Pressure levels ordered from highest pressure (surface, e.g. 1000 hPa) to lowest (top of atmosphere, e.g. 50 hPa) or vice-versa. We sort pressure ascending.
    """
    logger.info("Aligning coordinates for GraphCast...")
    
    # 1. Coordinate naming standardization
    rename_dict = {}
    if "latitude" in ds.coords and "lat" not in ds.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.dims:
        rename_dict["longitude"] = "lon"
    if "pressure_level" in ds.coords and "level" not in ds.dims:
        rename_dict["pressure_level"] = "level"
        
    if rename_dict:
        logger.info(f"Renaming coordinates: {rename_dict}")
        ds = ds.rename(rename_dict)
        
    # 2. Sort Latitude descending (North to South)
    ds = ds.sortby("lat", ascending=False)
    
    # 3. Sort Longitude ascending and shift range to [0, 360)
    lon_coords = ds["lon"].values
    if np.any(lon_coords < 0):
        logger.info("Shifting longitude from [-180, 180) to [0, 360)")
        # Shift longitudes
        ds = ds.assign_coords(lon=(ds["lon"] % 360))
        
    ds = ds.sortby("lon", ascending=True)
    
    # 4. Sort Pressure levels ascending
    if "level" in ds.coords:
        ds = ds.sortby("level", ascending=True)
        
    logger.info(f"Coordinates aligned. Shape: Lat={ds.dims.get('lat')}, Lon={ds.dims.get('lon')}")
    return ds

VAR_RENAME_MAP = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation_6hr",
    "tcc": "total_cloud_cover",
    "sp": "surface_pressure",
    "lsm": "land_sea_mask",
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "w": "vertical_velocity",
}

def standardize_variables(
    ds: xr.Dataset,
    static_template_path: str = "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
) -> xr.Dataset:
    """Standardizes coordinate names, renames variables to GNN long names, and merges static variables.
    """
    logger.info("Standardizing dataset variables and loading static features...")
    
    # 1. Align coordinates first
    ds = align_coordinates(ds)
    
    # 2. Rename variables to long names
    rename_dict = {}
    for short_name, long_name in VAR_RENAME_MAP.items():
        if short_name in ds.data_vars and long_name not in ds.data_vars:
            rename_dict[short_name] = long_name
            
    if rename_dict:
        logger.info(f"Renaming variables: {rename_dict}")
        ds = ds.rename(rename_dict)

    # 3. Add static variables only if genuinely missing (avoids grid-mismatch MergeError)
    import os
    static_vars_needed = [v for v in ["geopotential_at_surface", "land_sea_mask"]
                          if v not in ds.data_vars]
    if static_vars_needed and os.path.exists(static_template_path):
        logger.info(f"Loading static variables from template: {static_template_path}")
        logger.info(f"Static vars to add: {static_vars_needed}")
        template_ds = xr.open_dataset(static_template_path, engine="scipy")
        template_ds = align_coordinates(template_ds)
        available = [v for v in static_vars_needed if v in template_ds.data_vars]
        if available:
            static_subset = template_ds[available]
            static_subset = static_subset.interp(lat=ds.lat, lon=ds.lon, method="linear")
            for var in static_subset.data_vars:
                if "time" in static_subset[var].dims:
                    static_subset[var] = static_subset[var].isel(time=0, drop=True)
                if "batch" in static_subset[var].dims:
                    static_subset[var] = static_subset[var].isel(batch=0, drop=True)
            ds = xr.merge([ds, static_subset])
            logger.info(f"Static variables merged: {available}")
        template_ds.close()
    elif not static_vars_needed:
        logger.info("All static variables already present in dataset -- skipping template merge.")
    else:
        logger.warning(f"Static template not found at '{static_template_path}'. Static variables remain missing.")
            
    return ds

def regrid_dataset(ds: xr.Dataset, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.Dataset:
    """Regrids the dataset onto the target lat-lon grid using bilinear interpolation."""
    logger.info(f"Regridding dataset to target shape: Lat={len(target_lat)}, Lon={len(target_lon)}")
    
    # Standardize coordinate names first
    ds = align_coordinates(ds)
    
    # Slicing out spatial boundaries matching target
    regridded_ds = ds.interp(lat=target_lat, lon=target_lon, method="linear")
    return regridded_ds

def add_graphcast_forcings(ds: xr.Dataset) -> xr.Dataset:
    """Adds the required day-progress, year-progress and TISR (Incident Solar Radiation) forcings.
    
    Leverages the official `data_utils.add_derived_vars` and `data_utils.add_tisr_var`
    to guarantee identical mathematics and key naming.
    """
    logger.info("Calculating atmospheric calendar progress and incident solar radiation (TISR)...")
    
    # Ensure coordinates standard
    ds = align_coordinates(ds)
    
    # Ensure time coordinates are in timedelta64 format and 'datetime' coordinate exists
    if "datetime" not in ds.coords:
        # Generate datetime coordinate from time coordinate if time is Timestamp
        if ds["time"].dtype == "O" or isinstance(ds["time"].values[0], (pd.Timestamp, np.datetime64)):
            ds = ds.assign_coords(datetime=ds["time"])
        else:
            # Assuming time coordinate contains timedeltas or relative offsets from an epoch
            # Generate a mock epoch starting point if not set
            epoch = pd.Timestamp("2020-01-01 00:00:00")
            datetimes = [epoch + pd.Timedelta(t) for t in ds["time"].values]
            ds = ds.assign_coords(datetime=("time", datetimes))
            
    # Cast variables in place using data_utils
    # Copy dataset to avoid mutating source dataset in unexpected ways
    ds_with_forcings = ds.copy(deep=True)
    
    # Add year_progress, year_progress_sin, year_progress_cos, day_progress, day_progress_sin, day_progress_cos
    data_utils.add_derived_vars(ds_with_forcings)
    
    # Add toa_incident_solar_radiation
    data_utils.add_tisr_var(ds_with_forcings)
    
    logger.info("Successfully added all GraphCast forcing and derived variables.")
    return ds_with_forcings

def extract_variables_and_cast(
    ds: xr.Dataset, 
    input_vars: List[str], 
    target_vars: List[str],
    forcing_vars: List[str],
    pressure_levels: List[int]
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Casts float64 variables to float32 and extracts inputs, targets and forcings.
    
    Ensures that the output contains the exact variable groupings expected by the GraphCast model.
    """
    logger.info("Extracting variable sets and casting variables to float32...")
    
    # Ensure all data variables are float32 (JAX standard)
    for var in ds.data_vars:
        if ds[var].dtype == np.float64:
            ds[var] = ds[var].astype(np.float32)
            
    # Filter level dimension
    if "level" in ds.coords:
        ds = ds.sel(level=list(pressure_levels))
        
    # Standardize time dimension slice logic (e.g. 12 hours of inputs, lead times as targets)
    # The official data_utils has extract_inputs_targets_forcings
    # Let's leverage it:
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        input_variables=tuple(input_vars),
        target_variables=tuple(target_vars),
        forcing_variables=tuple(forcing_vars),
        pressure_levels=tuple(pressure_levels),
        input_duration="12h",
        target_lead_times="6h" # Default one-step target lead time
    )
    
    return inputs, targets, forcings
