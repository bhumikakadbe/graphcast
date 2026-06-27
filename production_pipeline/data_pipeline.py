# data_pipeline.py
import xarray as xr
import dask
import pandas as pd
from typing import List, Tuple, Union, Optional
from production_pipeline.utils import logger, log_system_resources

def open_zarr_dataset(url: str, chunks: Optional[dict] = None) -> xr.Dataset:
    """Lazily opens a Zarr dataset from a cloud URL or local path using Dask.
    
    Args:
        url: Path or URL to the Zarr group.
        chunks: Optional dictionary defining chunking (e.g. {'time': 1, 'lat': 720, 'lon': 1440})
        
    Returns:
        xr.Dataset: The lazily loaded dataset.
    """
    logger.info(f"Lazily opening Zarr dataset from: {url}")
    log_system_resources("Before Open Zarr")
    
    # Configure default chunking optimized for atmospheric streaming if none is provided
    if chunks is None:
        chunks = {
            'time': 1,
            'level': 13,
            'lat': 180,
            'lon': 360
        }
        
    try:
        # Use consolidation for faster metadata reads if available, falling back to false if it errors
        ds = xr.open_zarr(url, chunks=chunks, consolidated=True)
    except Exception as e:
        logger.warning(f"Consolidated read failed, falling back to standard zarr open: {e}")
        ds = xr.open_zarr(url, chunks=chunks, consolidated=False)
        
    logger.info(f"Dataset successfully opened. Variables: {list(ds.data_vars.keys())}")
    return ds

def retrieve_time_slice(ds: xr.Dataset, start_time: Union[str, pd.Timestamp], end_time: Union[str, pd.Timestamp]) -> xr.Dataset:
    """Extracts a slice from the time dimension of the dataset."""
    logger.info(f"Retrieving time slice: {start_time} to {end_time}")
    return ds.sel(time=slice(start_time, end_time))

def retrieve_variable_subset(ds: xr.Dataset, variables: List[str]) -> xr.Dataset:
    """Extracts only a subset of data variables from the dataset."""
    logger.info(f"Extracting variable subset: {variables}")
    existing_vars = [v for v in variables if v in ds.data_vars]
    missing_vars = list(set(variables) - set(existing_vars))
    if missing_vars:
        logger.warning(f"Requested variables not found in dataset: {missing_vars}")
    return ds[existing_vars]

def retrieve_regional_subset(
    ds: xr.Dataset, 
    lat_range: Tuple[float, float] = (5.0, 38.0), 
    lon_range: Tuple[float, float] = (65.0, 98.0)
) -> xr.Dataset:
    """Extracts a regional subset of the dataset (defaults to India bounding box).
    
    Nagpur sits close to the geographic center of India at Latitude ~21.15N, Longitude ~79.09E.
    
    Args:
        ds: Xarray Dataset.
        lat_range: Tuple of (min_lat, max_lat). Supports both ascending and descending latitudes.
        lon_range: Tuple of (min_lon, max_lon).
        
    Returns:
        xr.Dataset: The sliced dataset.
    """
    logger.info(f"Slicing regional bounds: Latitudes {lat_range}, Longitudes {lon_range}")
    
    # Handle coordinate names (lat vs latitude, lon vs longitude)
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    
    # Check coordinate order for latitude
    lat_coords = ds[lat_name].values
    if len(lat_coords) > 1 and lat_coords[0] > lat_coords[1]:
        # Latitude is descending (North to South). Slice accordingly: slice(max, min)
        logger.debug("Detected descending latitude coordinate.")
        lat_slice = slice(max(lat_range), min(lat_range))
    else:
        # Latitude is ascending (South to North)
        lat_slice = slice(min(lat_range), max(lat_range))
        
    lon_slice = slice(min(lon_range), max(lon_range))
    
    sliced_ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    logger.info(f"Slicing complete. New grid shape: {sliced_ds[lat_name].shape} x {sliced_ds[lon_name].shape}")
    return sliced_ds

def chunk_optimizer(ds: xr.Dataset, max_chunk_memory_mb: float = 128.0) -> xr.Dataset:
    """Inspects the dataset and re-chunks variables if the individual dask chunk sizes are too large.
    
    Prevents large array chunks from triggering out-of-memory errors on nodes.
    """
    logger.info(f"Optimizing dataset chunks for max chunk size of {max_chunk_memory_mb} MB...")
    optimized_dims = {}
    
    for var_name, var in ds.data_vars.items():
        if hasattr(var.data, "chunksize"):
            # It's a dask array
            element_size = var.dtype.itemsize
            chunk_elements = 1
            for dim_size in var.data.chunksize:
                chunk_elements *= dim_size
            chunk_size_mb = (chunk_elements * element_size) / (1024 ** 2)
            
            logger.debug(f"Variable '{var_name}' chunk size: {chunk_size_mb:.2f} MB")
            
            if chunk_size_mb > max_chunk_memory_mb:
                logger.info(f"Variable '{var_name}' chunk size exceeds threshold. Flagging for re-chunking.")
                # Force rechunk along space dimensions to fit the memory budget
                optimized_dims['lat'] = min(ds.dims.get('lat', 720), 90)
                optimized_dims['lon'] = min(ds.dims.get('lon', 1440), 180)
                
    if optimized_dims:
        logger.info(f"Re-chunking dataset with optimized dimensions: {optimized_dims}")
        return ds.chunk(optimized_dims)
        
    logger.info("Dataset chunks are already optimal. No re-chunking needed.")
    return ds
