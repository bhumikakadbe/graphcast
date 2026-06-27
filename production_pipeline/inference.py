# inference.py
import functools
import xarray as xr
import numpy as np
import pandas as pd
import jax
import haiku as hk
from typing import Tuple, List

from graphcast import graphcast
from graphcast import rollout
from graphcast import xarray_tree
from production_pipeline import training
from production_pipeline.utils import logger, log_system_resources

def build_jitted_forward(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    norm_stats: Tuple[xr.Dataset, xr.Dataset, xr.Dataset],
    params: dict,
    state: dict
):
    """Creates a compiled JIT function of the one-step forward predictor for maximum inference speed."""
    logger.info("Building and JIT-compiling one-step GraphCast predictor for inference...")
    
    diffs_std, mean, stddev = norm_stats
    
    @hk.transform_with_state
    def forward_pass(inputs, targets_template, forcings):
        # We construct the model with gradient_checkpointing=False for faster inference speed
        predictor = training.construct_full_model(
            model_config=model_config,
            task_config=task_config,
            diffs_stddev_by_level=diffs_std,
            mean_by_level=mean,
            stddev_by_level=stddev,
            gradient_checkpointing=False
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)
        
    # Bind parameters and states using closure
    forward_fn = functools.partial(
        forward_pass.apply, 
        params, 
        state, 
        jax.random.PRNGKey(0)
    )
    
    # Wrap to discard state outputs (our inference is not stateful)
    def clean_forward_fn(inputs, targets_template, forcings):
        predictions, _ = forward_fn(inputs=inputs, targets_template=targets_template, forcings=forcings)
        return predictions
        
    jitted_forward = jax.jit(clean_forward_fn)
    logger.info("Inference JIT compilation complete.")
    return jitted_forward

def recursive_prediction_loop(
    jitted_forward_fn,
    inputs: xr.Dataset,
    forcings: xr.Dataset,
    targets_template: xr.Dataset,
    lead_steps: int = 40  # 40 steps of 6 hours = 10 days (240 hours)
) -> xr.Dataset:
    """Executes a recursive autoregressive forecasting loop.
    
    Args:
        jitted_forward_fn: The JIT-compiled one-step forward predictor.
        inputs: Dataset containing initial conditions at T-6h and T0.
        forcings: Dataset containing the forcing variables for the entire forecast window (T+6h to target).
        targets_template: Template dataset of one-step target.
        lead_steps: Number of 6-hour prediction steps to execute.
        
    Returns:
        xr.Dataset: The combined dataset containing the rolled out predictions for all lead times.
    """
    logger.info(f"Starting autoregressive forecast rollout for {lead_steps} steps ({lead_steps * 6} hours)...")
    log_system_resources("Rollout Start")
    
    # Cast input coordinates & data variables to float32
    def to_float32(ds):
        ds_c = ds.compute()
        for var in ds_c.data_vars:
            if ds_c[var].dtype == np.float64:
                ds_c[var] = ds_c[var].astype(np.float32)
        return ds_c
        
    current_inputs = to_float32(inputs)
    forcings = to_float32(forcings)
    targets_template = to_float32(targets_template)
    
    predictions_by_step = []
    
    # Initial input times are relative, e.g. -6h and 0h.
    # Prediction steps occur at 6h, 12h, 18h, etc.
    step_duration = pd.Timedelta("6h")
    
    for step in range(lead_steps):
        target_lead_time = (step + 1) * step_duration
        logger.info(f"Executing step {step + 1}/{lead_steps} (Lead Time: {target_lead_time})")
        
        # 1. Extract forcing slice for the current target lead time
        current_forcing = forcings.sel(time=[target_lead_time])
        
        # 2. Setup the target template with correct time coordinate
        current_template = targets_template.copy(deep=True)
        current_template = current_template.assign_coords(time=[target_lead_time])
        
        # 3. Call the JIT compiled model
        raw_prediction = jitted_forward_fn(
            inputs=current_inputs,
            targets_template=current_template,
            forcings=current_forcing
        )
        
        # 4. Enforce physical constraints to prevent model drift
        # Total precipitation should never be negative
        if "total_precipitation_6hr" in raw_prediction:
            raw_prediction["total_precipitation_6hr"] = xr.where(
                raw_prediction["total_precipitation_6hr"] < 0,
                0.0,
                raw_prediction["total_precipitation_6hr"]
            )
            
        predictions_by_step.append(raw_prediction)
        
        # 5. Prepare inputs for the next step:
        # We drop the oldest frame (T-6h), slide T0 to T-6h, and append the prediction as the new T0.
        # This keeps the time coordinates of our inputs at [-6h, 0h] relative to the next target.
        if step < lead_steps - 1:
            next_inputs_dict = {}
            # Merge predictions and current forcings to form the complete next step frame
            next_frame = xr.merge([raw_prediction, current_forcing])
            # Assign time coordinate 0h to next_frame to match input bounds
            next_frame = next_frame.assign_coords(time=[pd.Timedelta(0)])
            
            for var in current_inputs.data_vars:
                if "time" in current_inputs[var].dims:
                    # Get the frame at T0 (last time index) and assign time coordinate -6h
                    t0_frame = current_inputs[var].isel(time=[-1]).assign_coords(time=[-step_duration])
                    # Concatenate along time dimension to form the new [T-6h, T0] input
                    next_inputs_dict[var] = xr.concat([t0_frame, next_frame[var]], dim="time")
                else:
                    # Static variables (no time dimension) remain unchanged
                    next_inputs_dict[var] = current_inputs[var]
                
            current_inputs = xr.Dataset(next_inputs_dict, coords=current_inputs.coords)
            
    # Combine predictions across the time dimension
    logger.info("Combining forecast rollouts into final Xarray Dataset...")
    full_forecast = xr.concat(predictions_by_step, dim="time")
    
    logger.info("Forecast rollout complete.")
    log_system_resources("Rollout End")
    return full_forecast
