"""
validation_experiment.py
========================
Trains two lightweight GraphCast models on:
  1. Original ERA5 dataset
  2. Compressed/reconstructed ERA5 dataset
and compares their forecast RMSE/MAE to validate that the compression
pipeline does not degrade model performance.
"""

import os
import sys
import time
import logging
import argparse
import dataclasses
import numpy as np
import pandas as pd
import xarray as xr

# Add parent directory to path
sys.path.insert(0, str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))

from graphcast import data_utils
from graphcast import graphcast
from production_pipeline import preprocessing
from production_pipeline import normalization
from production_pipeline import training
from production_pipeline import inference
from compression.pipeline import run_compression_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Compression.ValidationExperiment")


def evaluate_model(
    params: dict,
    test_ds: xr.Dataset,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    norm_stats: tuple,
    lead_steps: int = 2,
) -> dict:
    """Evaluates the fine-tuned parameters on a test dataset."""
    log.info(f"Evaluating model on test dataset ({lead_steps} steps rollout)...")

    # Extract inputs, targets, and forcings
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        test_ds,
        target_lead_times=slice("6h", f"{lead_steps * 6}h"),
        **dataclasses.asdict(task_config)
    )

    # Use first target step as targets template
    targets_template = targets.isel(time=[0])

    # Compile the forward predictor function
    jitted_forward = inference.build_jitted_forward(
        model_config=model_config,
        task_config=task_config,
        norm_stats=norm_stats,
        params=params,
        state={}
    )

    # Generate predictions autoregressively
    predictions = inference.recursive_prediction_loop(
        jitted_forward_fn=jitted_forward,
        inputs=inputs,
        forcings=forcings,
        targets_template=targets_template,
        lead_steps=lead_steps
    )

    # Calculate metrics per variable
    metrics = {}
    for var in targets.data_vars:
        if var not in predictions.data_vars:
            continue
        
        t_vals = targets[var].values.flatten()
        p_vals = predictions[var].values.flatten()
        
        mask = ~(np.isnan(t_vals) | np.isnan(p_vals))
        t_clean = t_vals[mask]
        p_clean = p_vals[mask]
        
        if len(t_clean) == 0:
            continue

        rmse = np.sqrt(np.mean((t_clean - p_clean) ** 2))
        mae = np.mean(np.abs(t_clean - p_clean))
        
        try:
            corr = np.corrcoef(t_clean, p_clean)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        metrics[var] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "corr": float(corr),
        }
        log.info(f"  {var}: RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compression Validation Experiment")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/ERA5/raw/2015",
        help="Path to raw 2015 NetCDF files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=60,
        help="Number of days to use for training (default: 60)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=15,
        help="Number of days to use for testing (default: 15)",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log.info("=" * 60)
    log.info("  Starting Compression Validation Experiment")
    log.info("=" * 60)

    # 1. Paths to precomputed stats
    diffs_std_path = "checkpoints/diffs_stddev_by_level.nc"
    mean_path = "checkpoints/mean_by_level.nc"
    stddev_path = "checkpoints/stddev_by_level.nc"

    if not (os.path.exists(diffs_std_path) and os.path.exists(mean_path) and os.path.exists(stddev_path)):
        log.error("Google statistics files missing in checkpoints/ directory!")
        sys.exit(1)

    norm_stats = normalization.load_google_stats(diffs_std_path, mean_path, stddev_path)

    # 2. Run compression pipeline to get compressed and reconstructed dataset
    log.info("Running compression pipeline on raw data...")
    pipeline_results = run_compression_pipeline(
        data_path=args.raw_dir,
        year=2015,
        use_pca=True,
        validate=True,
    )

    original_ds = pipeline_results["original_ds"]
    
    # Load PCA compressor and reconstruct dataset
    # We do this explicitly to simulate loading the compressed file and reconstructing it
    from compression.pca_compressor import ERA5PCACompressor
    compressor = ERA5PCACompressor.load("data/ERA5/processed/2015/pca_compressor_2015.pkl")
    compressed_ds = xr.open_dataset("data/ERA5/processed/2015/era5_pca_2015.nc")
    reconstructed_ds = compressor.reconstruct(compressed_ds)
    
    # Add calendar progress and solar radiation
    log.info("Preprocessing original and reconstructed datasets...")
    original_ds = preprocessing.add_graphcast_forcings(original_ds)
    reconstructed_ds = preprocessing.add_graphcast_forcings(reconstructed_ds)

    # 3. Define Model and Task Configs
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )

    # Lightweight configuration for fast training on CPU
    model_config = graphcast.ModelConfig(
        resolution=0,
        mesh_size=4,
        latent_size=32,
        gnn_msg_steps=4,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )

    # 4. Partition datasets into train and test periods
    # We select day limits by converting days to 6h steps
    train_steps = args.train_days * 4
    test_steps = args.test_days * 4

    log.info(f"Partitioning data: train_steps={train_steps}, test_steps={test_steps}")
    
    # Run A: Original Dataset
    orig_train = original_ds.isel(time=slice(0, train_steps))
    orig_test = original_ds.isel(time=slice(train_steps, train_steps + test_steps))

    # Run B: Reconstructed Dataset
    recon_train = reconstructed_ds.isel(time=slice(0, train_steps))
    recon_test = reconstructed_ds.isel(time=slice(train_steps, train_steps + test_steps))

    # Extract inputs/targets/forcings for original training
    orig_train_inputs, orig_train_targets, orig_train_forcings = data_utils.extract_inputs_targets_forcings(
        orig_train,
        target_lead_times="6h",
        **dataclasses.asdict(task_config)
    )

    # Extract inputs/targets/forcings for reconstructed training
    recon_train_inputs, recon_train_targets, recon_train_forcings = data_utils.extract_inputs_targets_forcings(
        recon_train,
        target_lead_times="6h",
        **dataclasses.asdict(task_config)
    )

    # 5. Train Model A (Original)
    log.info("\n" + "="*50)
    log.info("  Training Model A on ORIGINAL dataset...")
    log.info("="*50)
    t_start_a = time.time()
    params_a = training.run_fine_tuning_loop(
        train_inputs=orig_train_inputs,
        train_targets=orig_train_targets,
        train_forcings=orig_train_forcings,
        norm_stats=norm_stats,
        model_config=model_config,
        task_config=task_config,
        epochs=args.epochs,
        checkpoint_out_path="checkpoints/model_original.nc"
    )
    time_a = time.time() - t_start_a
    log.info(f"Model A training finished in {time_a:.1f} seconds.")

    # 6. Train Model B (Reconstructed)
    log.info("\n" + "="*50)
    log.info("  Training Model B on RECONSTRUCTED/COMPRESSED dataset...")
    log.info("="*50)
    t_start_b = time.time()
    params_b = training.run_fine_tuning_loop(
        train_inputs=recon_train_inputs,
        train_targets=recon_train_targets,
        train_forcings=recon_train_forcings,
        norm_stats=norm_stats,
        model_config=model_config,
        task_config=task_config,
        epochs=args.epochs,
        checkpoint_out_path="checkpoints/model_compressed.nc"
    )
    time_b = time.time() - t_start_b
    log.info(f"Model B training finished in {time_b:.1f} seconds.")

    # 7. Evaluate both models on the uncompressed test dataset
    log.info("\n" + "="*50)
    log.info("  Evaluating Model A (Original Model)...")
    log.info("="*50)
    metrics_a = evaluate_model(params_a, orig_test, model_config, task_config, norm_stats)

    log.info("\n" + "="*50)
    log.info("  Evaluating Model B (Compressed Model)...")
    log.info("="*50)
    metrics_b = evaluate_model(params_b, orig_test, model_config, task_config, norm_stats)

    # 8. Compare Results
    log.info("\n" + "="*50)
    log.info("  Validation Experiment Summary")
    log.info("="*50)

    report_rows = []
    for var in metrics_a:
        if var in metrics_b:
            rmse_a = metrics_a[var]["rmse"]
            rmse_b = metrics_b[var]["rmse"]
            mae_a = metrics_a[var]["mae"]
            mae_b = metrics_b[var]["mae"]
            
            # RMSE degradation percentage (higher is worse)
            if rmse_a > 1e-10:
                degradation = (rmse_b - rmse_a) / rmse_a * 100
            else:
                degradation = 0.0

            report_rows.append({
                "variable": var,
                "rmse_original": rmse_a,
                "rmse_compressed": rmse_b,
                "degradation_pct": degradation,
                "mae_original": mae_a,
                "mae_compressed": mae_b,
                "corr_original": metrics_a[var]["corr"],
                "corr_compressed": metrics_b[var]["corr"],
            })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("logs/compression_validation_experiment.csv", index=False)
    log.info("Experiment report saved to: logs/compression_validation_experiment.csv")

    print(report_df[["variable", "rmse_original", "rmse_compressed", "degradation_pct"]].to_string(index=False))

    avg_degradation = report_df["degradation_pct"].mean()
    print(f"\nAverage RMSE degradation across variables: {avg_degradation:.2f}%")

    # Decision rule: degradation < 5.0%
    if avg_degradation < 5.0:
        log.info("🎉 SUCCESS: Compression validated! Degradation is within 5%% limit.")
        sys.exit(0)
    else:
        log.error("❌ FAILED: Compression degradation exceeds 5%% limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
