# visualization.py
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for web servers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Optional
from production_pipeline.utils import logger

# Try loading Cartopy, with standard matplotlib fallbacks if spatial libs are missing
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
    logger.info("Cartopy loaded successfully. Geographic projection support enabled.")
except ImportError:
    CARTOPY_AVAILABLE = False
    logger.warning("Cartopy not installed. Weather maps will fall back to standard grid plots.")

def plot_surface_map(
    ds: xr.Dataset,
    variable: str,
    time_idx: int = 0,
    output_path: str = "output_map.png",
    lat_range: Tuple[float, float] = (-90.0, 90.0),
    lon_range: Tuple[float, float] = (0.0, 360.0),
    marker_lat: Optional[float] = None,
    marker_lon: Optional[float] = None,
    projection_type: str = "Orthographic"
):
    """Generates a beautiful color-contoured meteorological map for a surface variable."""
    logger.info(f"Generating surface map for '{variable}' (Time Step Index: {time_idx}) -> {output_path}")
    
    # Standardize coordinate names
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    
    # Extract the target slice
    frame = ds.isel(time=time_idx)
    if "batch" in frame.dims:
        frame = frame.squeeze("batch")
    if "level" in frame.coords and frame["level"].ndim > 0:
        # If level coordinate is present, take the first level for surface representation
        frame = frame.isel(level=0)
        
    data = frame[variable].values
    lats = frame[lat_name].values
    lons = frame[lon_name].values
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    
    if CARTOPY_AVAILABLE:
        if projection_type == "PlateCarree":
            ax = plt.axes(projection=ccrs.PlateCarree())
            if lat_range == (-90.0, 90.0) and lon_range == (0.0, 360.0):
                ax.set_global()
            else:
                ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        else:
            # 3D Orthographic globe projection centered on marker coordinates
            c_lon = marker_lon if marker_lon is not None else 79.0882
            c_lat = marker_lat if marker_lat is not None else 21.1458
            ax = plt.axes(projection=ccrs.Orthographic(central_longitude=c_lon, central_latitude=c_lat))
            ax.set_global()
        
        # Add high-resolution map features
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f4")
        ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe")
        ax.add_feature(cfeature.COASTLINE, edgecolor="#1e293b", linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, edgecolor="#475569", linestyle=":", linewidth=0.8)
        
        # Gridlines (labels are not supported on Orthographic, so just add gridlines)
        ax.gridlines(linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        
        # Color contour
        cmap_dict = {
            "2m_temperature": "viridis",
            "total_precipitation_6hr": "Blues",
            "mean_sea_level_pressure": "viridis"
        }
        cmap = cmap_dict.get(variable, "Spectral_r")
        
        contour = ax.contourf(
            lons, lats, data, 
            transform=ccrs.PlateCarree(), 
            cmap=cmap, 
            levels=20, 
            alpha=0.85
        )
        
        # Add selected coordinate marker dynamically if passed
        if marker_lat is not None and marker_lon is not None:
            norm_mlon = marker_lon % 360
            ax.plot(norm_mlon, marker_lat, 'ro', markersize=6, label=f"Coord: {marker_lat:.2f}N, {marker_lon:.2f}E", transform=ccrs.PlateCarree())
            ax.legend(loc="upper right")
    else:
        # Standard matplotlib fallback
        ax = plt.gca()
        cmap = "coolwarm" if "temp" in variable else "Blues" if "precip" in variable else "viridis"
        contour = ax.contourf(lons, lats, data, cmap=cmap, levels=20)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        
        # Add selected coordinate marker dynamically if passed
        if marker_lat is not None and marker_lon is not None:
            norm_mlon = marker_lon % 360
            ax.plot(norm_mlon, marker_lat, 'ro', label=f"Coord: {marker_lat:.2f}N, {marker_lon:.2f}E")
            ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
    cbar = plt.colorbar(contour, orientation="horizontal", pad=0.08, aspect=40)
    cbar.set_label(f"{variable} unit metrics")
    
    plt.title(f"GraphCast Medium-Range Forecast: {variable}\nLead Step Index: {time_idx} (T + {(time_idx + 1) * 6} hrs)", weight="bold", pad=15)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Surface map successfully saved.")

def plot_wind_vectors(
    ds: xr.Dataset,
    time_idx: int = 0,
    output_path: str = "output_wind.png",
    lat_range: Tuple[float, float] = (-90.0, 90.0),
    lon_range: Tuple[float, float] = (0.0, 360.0),
    marker_lat: Optional[float] = None,
    marker_lon: Optional[float] = None,
    projection_type: str = "Orthographic"
):
    """Plots wind vector streamplots or quivers (U/V components) at 10m height over standard surfaces."""
    logger.info(f"Generating wind streamplot (Time Step Index: {time_idx}) -> {output_path}")
    
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    
    frame = ds.isel(time=time_idx)
    if "batch" in frame.dims:
        frame = frame.squeeze("batch")
    lats = frame[lat_name].values
    lons = frame[lon_name].values
    
    # Retrieve Wind Components
    u_var = "10m_u_component_of_wind" if "10m_u_component_of_wind" in ds.data_vars else "u_component_of_wind"
    v_var = "10m_v_component_of_wind" if "10m_v_component_of_wind" in ds.data_vars else "v_component_of_wind"
    
    if u_var not in ds.data_vars or v_var not in ds.data_vars:
        logger.warning("Wind component variables not found in dataset. Skipping wind vector plot.")
        return
        
    u_data = frame[u_var].values
    v_data = frame[v_var].values
    
    if "level" in frame.coords and frame["level"].ndim > 0:
        # Take standard lower-atmospheric level if multi-level wind
        u_data = u_data[0]
        v_data = v_data[0]
        
    speed = np.sqrt(u_data**2 + v_data**2)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    
    if CARTOPY_AVAILABLE:
        if projection_type == "PlateCarree":
            ax = plt.axes(projection=ccrs.PlateCarree())
            if lat_range == (-90.0, 90.0) and lon_range == (0.0, 360.0):
                ax.set_global()
            else:
                ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        else:
            # 3D Orthographic globe projection centered on marker coordinates
            c_lon = marker_lon if marker_lon is not None else 79.0882
            c_lat = marker_lat if marker_lat is not None else 21.1458
            ax = plt.axes(projection=ccrs.Orthographic(central_longitude=c_lon, central_latitude=c_lat))
            ax.set_global()
            
        ax.add_feature(cfeature.LAND, facecolor="#f8fafc")
        ax.add_feature(cfeature.OCEAN, facecolor="#f0f9ff")
        ax.add_feature(cfeature.COASTLINE, edgecolor="#0f172a", linewidth=1.0)
        
        # Plot streamline flow
        contour = ax.streamplot(
            lons, lats, u_data, v_data, 
            transform=ccrs.PlateCarree(),
            color=speed, 
            cmap="viridis", 
            linewidth=1.2,
            density=2.0
        )
        
        if marker_lat is not None and marker_lon is not None:
            norm_mlon = marker_lon % 360
            ax.plot(norm_mlon, marker_lat, 'ro', markersize=6, label=f"Coord: {marker_lat:.2f}N, {marker_lon:.2f}E", transform=ccrs.PlateCarree())
            ax.legend(loc="upper right")
            
        plt.colorbar(contour.lines, orientation="horizontal", pad=0.08, label="Wind Speed (m/s)")
    else:
        ax = plt.gca()
        contour = ax.streamplot(lons, lats, u_data, v_data, color=speed, cmap="viridis", linewidth=1.2)
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        
        if marker_lat is not None and marker_lon is not None:
            norm_mlon = marker_lon % 360
            ax.plot(norm_mlon, marker_lat, 'ro', label=f"Coord: {marker_lat:.2f}N, {marker_lon:.2f}E")
            ax.legend()
        plt.colorbar(contour.lines, label="Wind Speed (m/s)")
        ax.grid(True)
        
    plt.title(f"GraphCast 10m Atmospheric Circulation & Wind Streamline\nStep Index: {time_idx} (T + {(time_idx + 1) * 6} hrs)", weight="bold", pad=15)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Wind vector plot saved successfully.")

def plot_vertical_profile(
    ds: xr.Dataset,
    time_idx: int = 0,
    lat: float = 21.1458,
    lon: float = 79.0882,
    output_path: str = "output_profile.png"
):
    """Plots vertical pressure profiles for temperature/humidity above a coordinate (e.g. Nagpur)."""
    logger.info(f"Generating atmospheric vertical profile for coordinate: Lat={lat}, Lon={lon} -> {output_path}")
    
    if "level" not in ds.coords:
        logger.warning("No level dimension present in dataset. Cannot generate vertical profile.")
        return
        
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    
    # Nearest neighbor extraction
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest").isel(time=time_idx)
    
    levels = point["level"].values
    
    fig, ax1 = plt.subplots(figsize=(6, 8), dpi=150)
    
    # Plot Temperature
    t_var = "temperature" if "temperature" in ds.data_vars else None
    if t_var:
        temps_c = point[t_var].values - 273.15  # Convert Kelvin to Celsius
        ax1.plot(temps_c, levels, "r-o", linewidth=2, label="Temperature (°C)")
        ax1.set_xlabel("Temperature (°C)", color="r")
        ax1.tick_params(axis="x", labelcolor="r")
        
    # Plot Specific Humidity on twin axis
    q_var = "specific_humidity" if "specific_humidity" in ds.data_vars else None
    if q_var:
        ax2 = ax1.twiny()
        q_gkg = point[q_var].values * 1000.0  # Convert kg/kg to g/kg
        ax2.plot(q_gkg, levels, "b--s", linewidth=1.5, label="Specific Humidity (g/kg)")
        ax2.set_xlabel("Specific Humidity (g/kg)", color="b")
        ax2.tick_params(axis="x", labelcolor="b")
        
    ax1.set_ylabel("Pressure Level (hPa)")
    ax1.set_yscale("log")
    ax1.invert_yaxis()  # Standard meteorological orientation: 1000 hPa at bottom, 1 hPa at top
    
    # Formatting
    plt.title(f"Vertical Atmospheric Profile (Lat: {lat:.2f}, Lon: {lon:.2f})\nLead Step Index: {time_idx}", weight="bold", pad=15)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Vertical profile plot saved successfully.")


# ─── Colourmap helper ────────────────────────────────────────────────────────
VARIABLE_CMAPS = {
    "2m_temperature":            "RdYlBu_r",
    "mean_sea_level_pressure":   "viridis",
    "10m_u_component_of_wind":   "PuOr",
    "10m_v_component_of_wind":   "PuOr",
    "total_precipitation_6hr":   "Blues",
    "temperature":               "RdYlBu_r",
    "geopotential":              "plasma",
    "u_component_of_wind":       "PuOr",
    "v_component_of_wind":       "PuOr",
    "vertical_velocity":         "coolwarm",
    "specific_humidity":         "YlGnBu",
}

VARIABLE_UNITS = {
    "2m_temperature":            "K  (raw) / σ (norm)",
    "mean_sea_level_pressure":   "Pa (raw) / σ (norm)",
    "10m_u_component_of_wind":   "m/s",
    "10m_v_component_of_wind":   "m/s",
    "total_precipitation_6hr":   "m (raw) / σ (norm)",
    "temperature":               "K  (raw) / σ (norm)",
    "geopotential":              "m²/s² (raw) / σ (norm)",
    "u_component_of_wind":       "m/s",
    "v_component_of_wind":       "m/s",
    "vertical_velocity":         "Pa/s",
    "specific_humidity":         "kg/kg",
}


def plot_normalization_comparison(
    ds: xr.Dataset,
    mean_ds: xr.Dataset,
    stddev_ds: xr.Dataset,
    variable: str,
    time_idx: int = 0,
    level_hpa: Optional[int] = None,
    output_path: str = "output_norm.png",
    projection_type: str = "PlateCarree",
):
    """Renders a 3-panel figure: Raw data | Normalized data | Difference.

    This replicates the graphcast_demo.ipynb *plot_example_data* cell,
    showing what the downloaded ERA5 batch looks like before and after
    the InputsAndResiduals normalization applied by GraphCast.
    """
    logger.info(f"Generating normalization comparison for '{variable}' (time={time_idx}, level={level_hpa}) → {output_path}")

    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"

    # ── Extract raw data slice ─────────────────────────────────────────────
    frame = ds.isel(time=time_idx)
    if "batch" in frame.dims:
        frame = frame.squeeze("batch")

    is_atmospheric = ("level" in ds[variable].dims) if variable in ds.data_vars else False
    if is_atmospheric and level_hpa is not None and "level" in frame.coords:
        frame = frame.sel(level=level_hpa, method="nearest")

    if variable not in frame.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset. Available: {list(frame.data_vars.keys())}")

    raw_data = frame[variable].values.astype(np.float64)
    lats = frame[lat_name].values
    lons = frame[lon_name].values

    # ── Compute normalised data ────────────────────────────────────────────
    # Mirror exactly what graphcast normalization.InputsAndResiduals does:
    #   z = (x - mean) / stddev
    mean_val = 0.0
    std_val  = 1.0

    if variable in mean_ds.data_vars:
        m = mean_ds[variable].values
        if m.ndim == 1:          # level-dependent
            if level_hpa is not None and "level" in mean_ds.coords:
                m = float(mean_ds[variable].sel(level=level_hpa, method="nearest").values)
            else:
                m = float(m[0])
        else:
            m = float(np.squeeze(m))
        mean_val = m

    if variable in stddev_ds.data_vars:
        s = stddev_ds[variable].values
        if s.ndim == 1:
            if level_hpa is not None and "level" in stddev_ds.coords:
                s = float(stddev_ds[variable].sel(level=level_hpa, method="nearest").values)
            else:
                s = float(s[0])
        else:
            s = float(np.squeeze(s))
        if s > 0:
            std_val = s

    norm_data = (raw_data - mean_val) / std_val

    cmap = VARIABLE_CMAPS.get(variable, "viridis")
    unit = VARIABLE_UNITS.get(variable, "")

    # ── Build figure ────────────────────────────────────────────────────────
    if CARTOPY_AVAILABLE and projection_type != "Flat":
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=120,
                                 subplot_kw={"projection": proj})
        transform = ccrs.PlateCarree()

        datasets = [
            (raw_data,  f"Raw ERA5 — {variable}", cmap,      None),
            (norm_data, f"Normalised (z-score)",  cmap,      None),
            (norm_data - ((raw_data - mean_val) / std_val), "Δ Residual (norm − raw_z)", "RdBu_r", 0),
        ]
        # Third panel: difference is always zero when computed the same way — show norm vs raw_scaled for visual interest
        raw_scaled = (raw_data - np.nanmean(raw_data)) / (np.nanstd(raw_data) + 1e-9)
        datasets[2] = (norm_data - raw_scaled, "Δ (GraphCast norm − local z-score)", "RdBu_r", 0)

        for ax, (data, title, cm, center) in zip(axes, datasets):
            ax.add_feature(cfeature.LAND,      facecolor="#1a1f3a")
            ax.add_feature(cfeature.OCEAN,     facecolor="#0d1b2a")
            ax.add_feature(cfeature.COASTLINE, edgecolor="#60a5fa", linewidth=0.8)
            ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

            vmax = np.nanpercentile(np.abs(data), 98)
            vmin = -vmax if center == 0 else np.nanpercentile(data, 2)

            cf = ax.contourf(lons, lats, data, 20,
                             transform=transform, cmap=cm,
                             vmin=vmin, vmax=vmax, alpha=0.9)
            plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=30,
                         label=unit)
            ax.set_title(title, weight="bold", fontsize=11, pad=10)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=120)
        raw_scaled = (raw_data - np.nanmean(raw_data)) / (np.nanstd(raw_data) + 1e-9)
        datasets = [
            (raw_data,               f"Raw ERA5 — {variable}", cmap,      False),
            (norm_data,              "Normalised (z-score)",    cmap,      False),
            (norm_data - raw_scaled, "Δ (GraphCast norm − local z-score)", "RdBu_r", True),
        ]
        for ax, (data, title, cm, symmetric) in zip(axes, datasets):
            vmax = np.nanpercentile(np.abs(data), 98)
            vmin = -vmax if symmetric else np.nanpercentile(data, 2)
            cf = ax.contourf(lons, lats, data, 20, cmap=cm, vmin=vmin, vmax=vmax, alpha=0.9)
            plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06, aspect=30, label=unit)
            ax.set_title(title, weight="bold", fontsize=11)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.grid(True, linestyle="--", alpha=0.4)

    level_str = f" @ {level_hpa} hPa" if (is_atmospheric and level_hpa) else ""
    plt.suptitle(
        f"GraphCast ERA5 Batch Inspector — {variable}{level_str}\n"
        f"Time step {time_idx}  ·  mean={mean_val:.4g}  ·  σ={std_val:.4g}",
        weight="bold", fontsize=13, y=1.01
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Normalization comparison plot saved.")

