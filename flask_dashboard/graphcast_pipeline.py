"""
pipeline.py — GraphCast Inference Pipeline
==========================================
CDS download  →  xarray transformation  →  GraphCast inference
Based exactly on the data flow found in 13.py (Google DeepMind GraphCast).

All coordinate expectations, variable names, and dim orders are derived
directly from the analysed notebook. Do NOT modify the rename/reindex
logic without re-verifying against extract_inputs_targets_forcings().
"""

import dataclasses
import datetime
import functools
import os
import tempfile
import logging
from typing import Optional

import cdsapi
import haiku as hk
import jax
import numpy as np
import xarray as xr

from google.cloud import storage as gcs_storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Constants & GCS helpers
# ─────────────────────────────────────────────────────────────────────────────

GCS_BUCKET  = "dm_graphcast"
GCS_PREFIX  = "graphcast/"
STATS_FILES = [
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
    "stats/diffs_stddev_by_level.nc",
]

_DEFAULT_CHECKPOINTS = [
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13.npz",
    "GraphCast - ERA5 1979-2015 - resolution 0.25 - pressure levels 13.npz",
    "GraphCast - ERA5 1979-2015 - resolution 0.25 - pressure levels 37.npz",
    "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13.npz",
]


def list_checkpoints() -> list[str]:
    """Return available checkpoint names from the public GCS bucket."""
    try:
        client = gcs_storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET)
        blobs  = bucket.list_blobs(prefix=GCS_PREFIX + "params/")
        names  = [
            b.name.removeprefix(GCS_PREFIX + "params/")
            for b in blobs
            if b.name != GCS_PREFIX + "params/"
        ]
        return names if names else _DEFAULT_CHECKPOINTS
    except Exception as exc:
        log.warning("Could not list GCS checkpoints (%s). Using defaults.", exc)
        return _DEFAULT_CHECKPOINTS


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CDS credentials & client
# ─────────────────────────────────────────────────────────────────────────────

def validate_cds_credentials() -> dict:
    """
    Validate ~/.cdsapirc format WITHOUT making a network call.
    Returns {"ok": True, "key_preview": "..."} or {"ok": False, "error": "..."}.
    """
    rc_path = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(rc_path):
        return {"ok": False, "error":
                f"~/.cdsapirc not found at {rc_path}. Create it with:\n"
                "  url: https://cds.climate.copernicus.eu/api\n"
                "  key: <your-CDS-API-key>"}

    content = open(rc_path).read()

    if "url:" not in content:
        return {"ok": False, "error": "~/.cdsapirc is missing the 'url:' field."}
    if "key:" not in content:
        return {"ok": False, "error": "~/.cdsapirc is missing the 'key:' field."}

    key_val = ""
    for line in content.splitlines():
        if line.strip().startswith("key:"):
            key_val = line.split("key:", 1)[1].strip()
            break

    placeholders = ("", "YOUR_KEY", "YOUR_CDS_API_KEY_HERE", "<your-CDS-API-key>")
    if key_val in placeholders or len(key_val) < 8:
        return {"ok": False, "error":
                "CDS API key is empty or a placeholder. "
                "Get your real key from https://cds.climate.copernicus.eu/profile"}

    preview = key_val[:8] + "..." + key_val[-4:]
    return {"ok": True, "key_preview": preview, "rc_path": rc_path}


def _cds_client() -> cdsapi.Client:
    """
    Return a CDS API client with full logging (quiet=False).
    Credentials must be in ~/.cdsapirc:
        url: https://cds.climate.copernicus.eu/api
        key: <your-UUID-key>
    """
    rc_path = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(rc_path):
        raise FileNotFoundError(
            f"CDS credentials file not found at {rc_path}.\n"
            "Create it with:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <paste your CDS API key here>"
        )
    return cdsapi.Client(quiet=False, verify=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CDS download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_era5(
    date: datetime.date,
    n_steps: int,
    resolution: float,
    pressure_levels_count: int,
    tmpdir: str,
) -> dict[str, str]:
    """
    Download ERA5 data for a forecast starting at `date` 00:00 UTC.

    GraphCast needs 2 input timesteps + n_steps target timesteps = n_steps+2.
    Day 1 covers up to 4 slots (00,06,12,18); any remaining spill to Day 2.

    Returns dict of {tag: local_nc_path}.
    """
    c       = _cds_client()
    grid    = [str(resolution), str(resolution)]
    if pressure_levels_count not in graphcast.PRESSURE_LEVELS:
        raise ValueError("pressure_levels must be 13 or 37")
        
    pressure_levels = graphcast.PRESSURE_LEVELS[pressure_levels_count]
    pl_list = [str(p) for p in pressure_levels]
    # GraphCast requires 2 input timesteps + n_steps target timesteps = n_steps+2.
    # We must download n_steps + 3 because the first timestep is consumed/dropped 
    # to compute the 6h total precipitation differences.
    total_steps = n_steps + 3
    all_slots   = ["00:00", "06:00", "12:00", "18:00"]
    day1_times  = all_slots[:min(total_steps, 4)]
    remaining   = total_steps - len(day1_times)
    day2_times  = all_slots[:min(remaining, 4)] if remaining > 0 else []
    remaining   = remaining - len(day2_times)
    day3_times  = all_slots[:remaining] if remaining > 0 else []


    d1  = date
    d2  = date + datetime.timedelta(days=1)
    d3  = date + datetime.timedelta(days=2)
    res = {}

    def _fmt(d: datetime.date) -> dict:
        return {"year": d.strftime("%Y"), "month": d.strftime("%m"), "day": d.strftime("%d")}

    def _nc(tag: str) -> str:
        path = os.path.join(tmpdir, f"era5_{tag}.nc")
        res[tag] = path
        return path

    # ── Pressure-level variables ──────────────────────────────────────────────
    pressure_vars = [
        "temperature", "geopotential",
        "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "vertical_velocity",
    ]
    log.info("CDS: Downloading pressure-level data (Day 1) for %s…", date)
    c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type": "reanalysis", "data_format": "netcdf",
        **_fmt(d1), "time": day1_times, "grid": grid,
        "pressure_level": pl_list, "variable": pressure_vars,
    }, _nc("pressure_d1"))
    log.info("CDS: pressure_d1 done → %s", res["pressure_d1"])

    if day2_times:
        log.info("CDS: Downloading pressure-level data (Day 2) for %s…", d2)
        c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d2), "time": day2_times, "grid": grid,
            "pressure_level": pl_list, "variable": pressure_vars,
        }, _nc("pressure_d2"))
        log.info("CDS: pressure_d2 done → %s", res["pressure_d2"])

    if day3_times:
        log.info("CDS: Downloading pressure-level data (Day 3) for %s…", d3)
        c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d3), "time": day3_times, "grid": grid,
            "pressure_level": pl_list, "variable": pressure_vars,
        }, _nc("pressure_d3"))
        log.info("CDS: pressure_d3 done → %s", res["pressure_d3"])

    # ── Surface variables ─────────────────────────────────────────────────────
    # "geopotential" here is the surface geopotential (CDS short name "z")
    # which we'll extract later as the static geopotential_at_surface field.
    surface_vars = [
        "2m_temperature", "mean_sea_level_pressure",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "land_sea_mask", "geopotential",
    ]
    log.info("CDS: Downloading surface data (Day 1)…")
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "data_format": "netcdf",
        **_fmt(d1), "time": day1_times, "grid": grid, "variable": surface_vars,
    }, _nc("surface_d1"))
    log.info("CDS: surface_d1 done → %s", res["surface_d1"])

    if day2_times:
        log.info("CDS: Downloading surface data (Day 2)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d2), "time": day2_times, "grid": grid, "variable": surface_vars,
        }, _nc("surface_d2"))
        log.info("CDS: surface_d2 done → %s", res["surface_d2"])

    if day3_times:
        log.info("CDS: Downloading surface data (Day 3)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d3), "time": day3_times, "grid": grid, "variable": surface_vars,
        }, _nc("surface_d3"))
        log.info("CDS: surface_d3 done → %s", res["surface_d3"])

    # ── Total precipitation ───────────────────────────────────────────────────
    log.info("CDS: Downloading total precipitation (Day 1)…")
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "data_format": "netcdf",
        **_fmt(d1), "time": day1_times, "grid": grid,
        "variable": ["total_precipitation"],
    }, _nc("tp_d1"))
    log.info("CDS: tp_d1 done → %s", res["tp_d1"])

    if day2_times:
        log.info("CDS: Downloading total precipitation (Day 2)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d2), "time": day2_times, "grid": grid,
            "variable": ["total_precipitation"],
        }, _nc("tp_d2"))
        log.info("CDS: tp_d2 done → %s", res["tp_d2"])

    if day3_times:
        log.info("CDS: Downloading total precipitation (Day 3)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d3), "time": day3_times, "grid": grid,
            "variable": ["total_precipitation"],
        }, _nc("tp_d3"))
        log.info("CDS: tp_d3 done → %s", res["tp_d3"])

    # ── TOA incident solar radiation ──────────────────────────────────────────
    log.info("CDS: Downloading TISR (Day 1)…")
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "data_format": "netcdf",
        **_fmt(d1), "time": day1_times, "grid": grid,
        "variable": ["toa_incident_solar_radiation"],
    }, _nc("tisr_d1"))
    log.info("CDS: tisr_d1 done → %s", res["tisr_d1"])

    if day2_times:
        log.info("CDS: Downloading TISR (Day 2)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d2), "time": day2_times, "grid": grid,
            "variable": ["toa_incident_solar_radiation"],
        }, _nc("tisr_d2"))
        log.info("CDS: tisr_d2 done → %s", res["tisr_d2"])
        
    if day3_times:
        log.info("CDS: Downloading TISR (Day 3)…")
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "data_format": "netcdf",
            **_fmt(d3), "time": day3_times, "grid": grid,
            "variable": ["toa_incident_solar_radiation"],
        }, _nc("tisr_d3"))
        log.info("CDS: tisr_d3 done → %s", res["tisr_d3"])

    log.info("CDS: All downloads complete.")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3.  xarray transformation — CDS → GraphCast Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _open_concat(paths: list[str]) -> xr.Dataset:
    datasets = [xr.open_dataset(p) for p in paths if p and os.path.exists(p)]
    if not datasets:
        raise FileNotFoundError(f"No valid .nc files found in: {paths}")
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="valid_time")


def build_graphcast_dataset(
    files: dict[str, str],
    pressure_levels_count: int,
) -> xr.Dataset:
    """
    Transform raw CDS .nc files into a GraphCast-ready xarray.Dataset.

    Coordinate conventions enforced:
      - dims:  batch=1, time (timedelta64), level (ascending hPa), lat (-90→90), lon
      - vars:  all GraphCast input/forcing variables (renamed from CDS short names)
      - coords: time (timedelta64 from t0), datetime (batch × time, datetime64)
    """
    def _paths(*tags):
        return [files[t] for t in tags if t in files and files[t] and os.path.exists(files[t])]

    ds_pressure = _open_concat(_paths("pressure_d1", "pressure_d2", "pressure_d3"))
    ds_surface  = _open_concat(_paths("surface_d1",  "surface_d2", "surface_d3"))
    ds_tp       = _open_concat(_paths("tp_d1",       "tp_d2", "tp_d3"))
    ds_tisr     = _open_concat(_paths("tisr_d1",     "tisr_d2", "tisr_d3"))

    ds_merged = xr.merge(
        [ds_pressure, ds_surface, ds_tp, ds_tisr],
        join="inner", compat="override",
    )

    # ── 1. Rename coordinates ─────────────────────────────────────────────────
    coord_renames = {}
    for old, new in [("valid_time","time"),("latitude","lat"),
                     ("longitude","lon"),("pressure_level","level")]:
        if old in ds_merged.dims:
            coord_renames[old] = new
    ds_cds = ds_merged.rename(coord_renames)

    # ── 2. Rename variables (CDS short names → GraphCast names) ──────────────
    var_map = {
        "t":    "temperature",
        "z":    "geopotential",
        "u":    "u_component_of_wind",
        "v":    "v_component_of_wind",
        "q":    "specific_humidity",
        "w":    "vertical_velocity",
        "t2m":  "2m_temperature",
        "msl":  "mean_sea_level_pressure",
        "u10":  "10m_u_component_of_wind",
        "v10":  "10m_v_component_of_wind",
        "lsm":  "land_sea_mask",
        "tp":   "total_precipitation_6hr",
        "tisr": "toa_incident_solar_radiation",
    }
    ds_cds = ds_cds.rename({k: v for k, v in var_map.items() if k in ds_cds})

    # ── 3. Fix lat direction: CDS 90→-90, GraphCast -90→90 ───────────────────
    if float(ds_cds.lat[0]) > float(ds_cds.lat[-1]):
        ds_cds = ds_cds.reindex(lat=ds_cds.lat[::-1])

    # ── 4. Fix level direction: CDS 1000→50, GraphCast 50→1000 ───────────────
    if "level" in ds_cds.dims and ds_cds.level[0] > ds_cds.level[-1]:
        ds_cds = ds_cds.reindex(level=ds_cds.level[::-1])

    # ── 5. Add batch dimension ─────────────────────────────────────────────────
    ds_cds = ds_cds.expand_dims("batch", axis=0)

    # ── 6. Drop stray CDS-only coordinates ────────────────────────────────────
    ds_cds = ds_cds.drop_vars(["number", "expver"], errors="ignore")

    # ── 7. Fix dtypes ─────────────────────────────────────────────────────────
    ds_cds = ds_cds.assign_coords(
        lat=ds_cds.lat.astype("float32"),
        lon=ds_cds.lon.astype("float32"),
    )
    if "level" in ds_cds.coords:
        ds_cds["level"] = ds_cds.level.astype("int32")

    # ── 8. Fix total precipitation: instantaneous→6h accumulation ────────────
    if "total_precipitation_6hr" in ds_cds:
        tp = ds_cds["total_precipitation_6hr"]
        ds_cds["total_precipitation_6hr"] = tp.diff(dim="time").clip(min=0).fillna(0)

    # ── 8b. Make static fields truly static (no time dim) ─────────────────────
    # GraphCast errors with: "Time-dependent input variable land_sea_mask must
    # either be a forcing variable, or a target variable" if these have time dim.
    # Ref: 13.py lines 337–343
    for static_var in ["land_sea_mask", "geopotential_at_surface"]:
        if static_var in ds_cds and "time" in ds_cds[static_var].dims:
            ds_cds[static_var] = ds_cds[static_var].isel(time=0).drop_vars("time")

    # ── 9. Add geopotential_at_surface (static, no time dim) ─────────────────
    sfc_path = files.get("surface_d1")
    if sfc_path and os.path.exists(sfc_path):
        ds_sfc_raw = xr.open_dataset(sfc_path)
        if "z" in ds_sfc_raw:
            z_sfc = ds_sfc_raw["z"].isel(valid_time=0).drop_vars("valid_time", errors="ignore")
            if "latitude"  in z_sfc.dims: z_sfc = z_sfc.rename({"latitude": "lat"})
            if "longitude" in z_sfc.dims: z_sfc = z_sfc.rename({"longitude": "lon"})
            if float(z_sfc.lat[0]) > float(z_sfc.lat[-1]):
                z_sfc = z_sfc.reindex(lat=z_sfc.lat[::-1])
            z_sfc = z_sfc.astype("float32")
            z_sfc.name = "geopotential_at_surface"
            ds_cds["geopotential_at_surface"] = z_sfc

    # ── 10. Compute time coordinates ─────────────────────────────────────
    # We must set t0 to times[1] so that the dataset times are [-6h, 0h, 6h...]
    # GraphCast explicitly expects 2 input timesteps at -6h and 0h.
    times      = ds_cds.time.values
    t0         = times[1] if len(times) > 1 else times[0]
    timedeltas = times - t0
    datetime_vals = np.array(times).reshape(1, -1)   # (batch=1, time=N)

    ds_cds = ds_cds.assign_coords(
        time=timedeltas,
        datetime=(["batch", "time"], datetime_vals),
    )

    log.info("GraphCast dataset built: dims=%s  vars=%s",
             dict(ds_cds.dims), list(ds_cds.data_vars))
    return ds_cds


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_normalization_stats(tmpdir: str) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Download and return (diffs_stddev, mean, stddev) normalization datasets."""
    client = gcs_storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)
    results = {}
    for stat_path in STATS_FILES:
        key  = stat_path.split("/")[-1].replace(".nc", "")
        dest = os.path.join(tmpdir, stat_path.split("/")[-1])
        if not os.path.exists(dest):
            log.info("GCS: Downloading normalization stat: %s", stat_path)
            bucket.blob(GCS_PREFIX + stat_path).download_to_filename(dest)
        results[key] = xr.load_dataset(dest).compute()

    return (
        results["diffs_stddev_by_level"],
        results["mean_by_level"],
        results["stddev_by_level"],
    )


def load_checkpoint(checkpoint_name: str, tmpdir: str):
    """Load checkpoint from local folder or GCS and return (params, state, model_config, task_config)."""
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", checkpoint_name)
    dest   = os.path.join(tmpdir, checkpoint_name)

    if os.path.exists(local_path):
        log.info("Loading local checkpoint from: %s", local_path)
        import shutil
        shutil.copy2(local_path, dest)
    else:
        # Download from GCS
        client = gcs_storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET)
        if not os.path.exists(dest):
            log.info("GCS: Downloading checkpoint: %s", checkpoint_name)
            bucket.blob(GCS_PREFIX + "params/" + checkpoint_name).download_to_filename(dest)

    with open(dest, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    log.info("Checkpoint loaded: %s", ckpt.description)
    return ckpt.params, {}, ckpt.model_config, ckpt.task_config


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GraphCast model construction & JIT
# ─────────────────────────────────────────────────────────────────────────────

def _construct_predictor(model_config, task_config,
                         diffs_stddev_by_level, mean_by_level, stddev_by_level):
    """Build GraphCast → Bfloat16Cast → InputsAndResiduals → autoregressive.Predictor."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
    return predictor


def get_jitted_model(params, state, model_config, task_config,
                     diffs_stddev_by_level, mean_by_level, stddev_by_level,
                     sample_inputs, sample_targets, sample_forcings):
    """Return the JIT-compiled inference function with params/config baked in."""

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = _construct_predictor(
            model_config, task_config,
            diffs_stddev_by_level, mean_by_level, stddev_by_level,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    if params is None:
        init_jitted = jax.jit(with_configs(run_forward.init))
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=sample_inputs,
            targets_template=sample_targets,
            forcings=sample_forcings,
        )

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
    return run_forward_jitted


# ─────────────────────────────────────────────────────────────────────────────
# 6.  End-to-end inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    date: datetime.date,
    n_steps: int,
    resolution: float,
    pressure_levels_count: int,
    checkpoint_name: str,
    tmpdir: Optional[str] = None,
    progress_cb=None,
) -> xr.Dataset:
    """
    Full pipeline: download ERA5 → build Dataset → load model → rollout.

    Returns xr.Dataset of predictions with all target variables.
    """
    def _progress(msg: str):
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    own_tmpdir = tmpdir is None
    if own_tmpdir:
        tmpdir = tempfile.mkdtemp(prefix="graphcast_")

    try:
        _progress("⬇️  Downloading ERA5 data from CDS (this may take 5–20 min)…")
        files = download_era5(date, n_steps, resolution, pressure_levels_count, tmpdir)

        _progress("🔄  Transforming CDS data to GraphCast format…")
        example_batch = build_graphcast_dataset(files, pressure_levels_count)

        _progress("📊  Loading normalization statistics from GCS…")
        diffs_stddev, mean_stats, stddev_stats = load_normalization_stats(tmpdir)

        _progress(f"🧠  Loading GraphCast checkpoint: {checkpoint_name}…")
        params, state, model_config, task_config = load_checkpoint(checkpoint_name, tmpdir)
        # 🔒 Ensure model & data match
        model_levels = len(task_config.pressure_levels)

        if model_levels != pressure_levels_count:
            raise ValueError(
                f"❌ Mismatch: Model expects {model_levels} pressure levels "
                f"but received {pressure_levels_count}. "
                f"Please select the correct model."
            )

        # 🔒 Ensure resolution matches model
        expected_resolution = model_config.resolution

        if expected_resolution != 0 and abs(expected_resolution - resolution) > 1e-6:
            raise ValueError(
                f"❌ Resolution mismatch: Model expects {expected_resolution}°, "
                f"but got {resolution}°"
            )
        if model_config.resolution not in (0, 360.0 / example_batch.sizes["lon"]):
            raise ValueError(
                f"Model resolution ({model_config.resolution}°) does not match "
                f"data resolution ({resolution}°). Choose the matching checkpoint."
            )

        _progress("✂️  Extracting inputs, targets, forcings…")
        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{n_steps * 6}h"),
            **dataclasses.asdict(task_config),
        )
        log.info("Inputs: %s | Targets: %s | Forcings: %s",
                 eval_inputs.dims.mapping, eval_targets.dims.mapping,
                 eval_forcings.dims.mapping)

        _progress("⚙️  Compiling GraphCast model (JIT — first run takes ~5 min on T4)…")
        run_forward_jitted = get_jitted_model(
            params, state, model_config, task_config,
            diffs_stddev, mean_stats, stddev_stats,
            eval_inputs, eval_targets, eval_forcings,
        )

        _progress(f"🚀  Running autoregressive rollout ({n_steps} × 6h steps)…")
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings,
        )
        _progress("✅  Inference complete.")
        return predictions

    finally:
        if own_tmpdir:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_precipitation(
    predictions: xr.Dataset,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    is_global: bool = False,
) -> xr.DataArray:
    """
    Extract total_precipitation_6hr for a bounding box and convert m → mm.
    Returns DataArray with dims (batch, time, lat, lon).
    """
    if "total_precipitation_6hr" not in predictions:
        raise KeyError(
            "'total_precipitation_6hr' not found in predictions. "
            "Ensure task_config.target_variables includes it."
        )
    tp = predictions["total_precipitation_6hr"] * 1000.0   # m → mm
    if is_global or lat_min is None:
        return tp
    return tp.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def predictions_to_json(tp_region: xr.DataArray) -> dict:
    """
    Convert regional precipitation DataArray to JSON-serialisable dict:
        {"times": [...], "lats": [...], "lons": [...], "data": [[[mm,...],...],...]}
    """
    if "batch" in tp_region.dims:
        tp_region = tp_region.isel(batch=0)

    times_str = [
        str(datetime.timedelta(microseconds=int(t) // 1000))
        for t in tp_region.time.values
    ]
    return {
        "times": times_str,
        "lats":  tp_region.lat.values.tolist(),
        "lons":  tp_region.lon.values.tolist(),
        "data":  tp_region.values.tolist(),
    }


def save_predictions_netcdf(predictions: xr.Dataset, path: str) -> None:
    """Save full predictions Dataset to a NetCDF file."""
    predictions.to_netcdf(path)
    log.info("Predictions saved to %s", path)
