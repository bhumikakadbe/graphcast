# run_tiny_training.py
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import jax

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graphcast import data_utils
from graphcast import graphcast
from production_pipeline import preprocessing
from production_pipeline import normalization
from production_pipeline import training
from production_pipeline.utils import logger

DATASET_LOCAL_PATH = "checkpoints/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
DIFFS_STD_PATH = "checkpoints/diffs_stddev_by_level.nc"
MEAN_PATH = "checkpoints/mean_by_level.nc"
STDDEV_PATH = "checkpoints/stddev_by_level.nc"
OUTPUT_CKPT_PATH = "checkpoints/fine_tuned_model.nc"

def main():
    logger.info("=== Starting Phase 4: Run Tiny Fine-Tuning Experiment ===")
    
    # 1. Load data
    logger.info(f"Loading dataset from: {DATASET_LOCAL_PATH}")
    ds = xr.open_dataset(DATASET_LOCAL_PATH, engine="scipy")
    
    # 2. Preprocess / Align Coordinates & Add Forcings
    ds = preprocessing.align_coordinates(ds)
    ds = preprocessing.add_graphcast_forcings(ds)
    
    # 3. Task & Model Configurations
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )
    
    model_config = graphcast.ModelConfig(
        resolution=0,  # Match mesh size resolution helper
        mesh_size=4,   # Lightweight mesh
        latent_size=32, # Fast latent vector
        gnn_msg_steps=4,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6
    )
    
    # 4. Extract inputs, targets, and forcings
    logger.info("Extracting training subsets...")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        target_lead_times=slice("6h", "12h"),
        **dataclasses_asdict(task_config)
    )
    
    # 5. Log training shapes
    os.makedirs("logs", exist_ok=True)
    shapes_log_path = "logs/training_shapes.log"
    logger.info(f"Logging training tensor shapes to {shapes_log_path}...")
    with open(shapes_log_path, "w", encoding="utf-8") as f:
        f.write("=== Training Tensor Shapes ===\n")
        f.write(f"Inputs:   {inputs.dims.mapping}\n")
        f.write(f"Targets:  {targets.dims.mapping}\n")
        f.write(f"Forcings: {forcings.dims.mapping}\n")
        f.write("\nVariables:\n")
        for var in inputs.data_vars:
            f.write(f"  Input: {var:<30} {inputs[var].shape}\n")
        for var in targets.data_vars:
            f.write(f"  Target: {var:<30} {targets[var].shape}\n")
        for var in forcings.data_vars:
            f.write(f"  Forcing: {var:<30} {forcings[var].shape}\n")
            
    # 6. Load stats
    norm_stats = normalization.load_google_stats(
        DIFFS_STD_PATH,
        MEAN_PATH,
        STDDEV_PATH
    )
    
    # 7. Run training loop (1 epoch)
    logger.info("Starting fine-tuning loop for 1 epoch...")
    params = training.run_fine_tuning_loop(
        train_inputs=inputs,
        train_targets=targets,
        train_forcings=forcings,
        norm_stats=norm_stats,
        model_config=model_config,
        task_config=task_config,
        epochs=1,
        checkpoint_out_path=OUTPUT_CKPT_PATH
    )
    
    # 8. Verify checkpoint output
    logger.info("Verifying generated checkpoint...")
    if not os.path.exists(OUTPUT_CKPT_PATH):
        logger.error(f"Checkpoint file {OUTPUT_CKPT_PATH} was not created!")
        sys.exit(1)
        
    ckpt = training.load_pretrained_checkpoint(OUTPUT_CKPT_PATH)
    parameter_leaves = jax.tree_util.tree_leaves(ckpt.params)
    parameter_count = sum(p.size for p in parameter_leaves)
    
    logger.info(f"Checkpoint Loaded Successfully!")
    logger.info(f"  Checkpoint Size: {os.path.getsize(OUTPUT_CKPT_PATH) / (1024**2):.2f} MB")
    logger.info(f"  Parameter Count: {parameter_count}")
    
    if parameter_count > 0:
        logger.info("🎉 SUCCESS: Checkpoint verified and has non-zero parameters!")
    else:
        logger.error("❌ FAILED: Checkpoint parameter count is zero!")
        sys.exit(1)

def dataclasses_asdict(obj):
    import dataclasses
    return dataclasses.asdict(obj)

if __name__ == "__main__":
    main()
