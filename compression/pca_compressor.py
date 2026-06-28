"""
pca_compressor.py
=================
Applies Incremental PCA to each ERA5 variable independently, reducing the
spatial dimensionality while retaining a user-specified fraction of variance.

Key design decisions:
- Uses IncrementalPCA to handle large arrays without loading everything into RAM
- Compresses the (time × space) matrix: space dimension is reduced
- Saves the PCA model so it can be used to reconstruct data and compress new years
- Generates a per-variable compression report

Usage:
    from compression.pca_compressor import ERA5PCACompressor
    compressor = ERA5PCACompressor(variance_threshold=0.99)
    compressed_ds, report = compressor.fit_compress(ds, year=2015)
    # Later, compress 2016 using same PCA basis:
    compressed_2016, _ = compressor.transform(ds_2016)
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import IncrementalPCA

log = logging.getLogger("Compression.PCA")

MODELS_DIR = Path("compression") / "pca_models"


class ERA5PCACompressor:
    """
    Applies Incremental PCA to ERA5 xarray Datasets, variable by variable.

    The spatial dimensions (lat × lon) are flattened into a single dimension,
    resulting in a (time, n_grid_cells) matrix per variable. PCA then finds
    the directions of maximum variance in the spatial patterns.

    The compressed representation is an (time, n_components) matrix — far
    smaller than the original (time, lat × lon) representation.

    Attributes:
        variance_threshold: Fraction of variance to retain (default 0.99 = 99%).
        max_components: Hard upper bound on number of PCA components.
        batch_size: Number of time steps per IncrementalPCA batch.
        pca_models: Dict mapping variable_name -> fitted sklearn PCA object.
        metadata: Dict with compression info per variable.
    """

    def __init__(
        self,
        variance_threshold: float = 0.99,
        max_components: int = 100,
        batch_size: int = 500,
    ):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.batch_size = batch_size
        self.pca_models: dict[str, IncrementalPCA] = {}
        self.metadata: dict = {}

    def _get_spatial_matrix(self, da: xr.DataArray) -> tuple[np.ndarray, tuple]:
        """
        Flattens a DataArray into (time, n_spatial_cells) matrix.

        Returns:
            (matrix, original_spatial_shape)
        """
        # Identify time dim and spatial dims
        non_time_dims = [d for d in da.dims if d != "time"]
        # Move time to first axis
        da_t = da.transpose("time", *non_time_dims)
        arr = da_t.values.astype(np.float32)

        n_time = arr.shape[0]
        spatial_shape = arr.shape[1:]
        matrix = arr.reshape(n_time, -1)  # (time, lat*lon*level)

        # Replace NaNs with column means (PCA requires no NaNs)
        col_means = np.nanmean(matrix, axis=0)
        nan_mask = np.isnan(matrix)
        matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        return matrix, spatial_shape

    def fit_compress(
        self, ds: xr.Dataset, year: int, save_models: bool = True
    ) -> tuple[xr.Dataset, pd.DataFrame]:
        """
        Fits PCA on the dataset and returns compressed DataArrays.

        This is the TRAINING step — fits a new PCA model from scratch.
        Use `transform()` for subsequent years.

        Args:
            ds: ERA5 xarray Dataset for the training year.
            year: The year (for labeling/saving).
            save_models: If True, saves PCA models to disk for reuse.

        Returns:
            (compressed_ds, report_df)
        """
        log.info(f"Fitting PCA on {year} ERA5 data (variance threshold={self.variance_threshold})...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        compressed_arrays = {}
        report_rows = []

        for var in ds.data_vars:
            log.info(f"  Compressing variable: '{var}'")
            da = ds[var]
            # Accept 'valid_time' (ERA5 CDS download) as well as 'time'
            time_dim = None
            for candidate in ("time", "valid_time"):
                if candidate in da.dims:
                    time_dim = candidate
                    break
            if time_dim is None:
                log.debug(f"  '{var}' has no time dim — copying as-is.")
                compressed_arrays[var] = da
                continue
            # Rename valid_time → time so the rest of the code is uniform
            if time_dim == "valid_time":
                da = da.rename({"valid_time": "time"})

            matrix, spatial_shape = self._get_spatial_matrix(da)
            n_time, n_spatial = matrix.shape
            n_comps = min(self.max_components, n_time, n_spatial)

            log.info(f"    Original shape: ({n_time}, {n_spatial}) | "
                     f"Fitting {n_comps} components...")

            # Fit IncrementalPCA in batches
            pca = IncrementalPCA(n_components=n_comps, batch_size=self.batch_size)
            for start in range(0, n_time, self.batch_size):
                batch = matrix[start:start + self.batch_size]
                pca.partial_fit(batch)

            # Determine how many components to keep for desired variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_keep = int(np.searchsorted(cumvar, self.variance_threshold)) + 1
            n_keep = min(n_keep, n_comps)

            log.info(f"    Components kept: {n_keep}/{n_comps} "
                     f"(explains {cumvar[n_keep-1]*100:.2f}% variance)")

            # Transform to compressed space
            compressed = pca.transform(matrix)[:, :n_keep]  # (time, n_keep)

            # Get original coordinate names and coordinates except time
            non_time_dims = [d for d in da.dims if d != "time"]
            orig_coords = {d: da[d].values for d in non_time_dims}

            # Store model and compressed array
            self.pca_models[var] = pca
            self.metadata[var] = {
                "n_spatial_original": n_spatial,
                "spatial_shape": spatial_shape,
                "n_components_kept": n_keep,
                "variance_explained": float(cumvar[n_keep - 1]),
                "compression_ratio": round(n_spatial / n_keep, 1),
                "non_time_dims": non_time_dims,
                "orig_coords": orig_coords,
            }

            # Wrap as DataArray with component dimension
            # Use whichever time coord is present in the original dataset
            time_coord_name = "time" if "time" in ds.coords else "valid_time"
            comp_da = xr.DataArray(
                compressed,
                dims=["time", "pca_component"],
                coords={
                    "time": ds[time_coord_name].values,
                    "pca_component": np.arange(n_keep),
                },
                name=var,
                attrs={**da.attrs, "pca_n_components": n_keep,
                       "pca_variance_explained": float(cumvar[n_keep - 1])},
            )
            compressed_arrays[var] = comp_da

            # Size info
            original_mb = (n_time * n_spatial * 4) / (1024**2)
            compressed_mb = (n_time * n_keep * 4) / (1024**2)
            report_rows.append({
                "variable": var,
                "original_shape": f"({n_time}, {n_spatial})",
                "n_components": n_keep,
                "variance_explained_pct": round(float(cumvar[n_keep-1]) * 100, 2),
                "compression_ratio": round(n_spatial / n_keep, 1),
                "original_size_mb": round(original_mb, 2),
                "compressed_size_mb": round(compressed_mb, 2),
            })

            # Save PCA model to disk
            if save_models:
                model_path = MODELS_DIR / f"pca_{var}_{year}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(pca, f)
                log.info(f"    PCA model saved: {model_path}")

        compressed_ds = xr.Dataset(compressed_arrays)
        report_df = pd.DataFrame(report_rows)

        if not report_df.empty and "original_size_mb" in report_df.columns:
            total_orig = report_df["original_size_mb"].sum()
            total_comp = report_df["compressed_size_mb"].sum()
            log.info(f"\nPCA Compression Summary for {year}:")
            log.info(f"  Total original size:    {total_orig:.1f} MB")
            log.info(f"  Total compressed size:  {total_comp:.1f} MB")
            ratio = total_orig / total_comp if total_comp > 0 else float('inf')
            log.info(f"  Overall compression:    {ratio:.1f}x")
        else:
            log.warning("PCA report is empty — all variables were copied as-is (no time dimension found). "
                        "Check that the ERA5 dataset uses 'time' or 'valid_time' as a coordinate.")

        return compressed_ds, report_df

    def reconstruct(self, compressed_ds: xr.Dataset) -> xr.Dataset:
        """
        Reconstructs the original spatial fields from compressed PCA representation.
        Used for the validation experiment to measure information loss.
        """
        log.info("Reconstructing fields from PCA components...")
        reconstructed = {}

        for var in compressed_ds.data_vars:
            if var not in self.pca_models or "pca_component" not in compressed_ds[var].dims:
                reconstructed[var] = compressed_ds[var]
                continue

            pca = self.pca_models[var]
            meta = self.metadata[var]
            n_keep = meta["n_components_kept"]
            spatial_shape = meta["spatial_shape"]

            # Slice to actual components kept (xarray aligns and pads other elements with NaN)
            compressed_matrix = compressed_ds[var].values[:, :n_keep]  # (time, n_keep)

            # Pad with zeros for components we dropped
            n_comps_total = pca.n_components_
            if n_keep < n_comps_total:
                pad = np.zeros((compressed_matrix.shape[0], n_comps_total - n_keep))
                compressed_matrix = np.hstack([compressed_matrix, pad])

            # Inverse transform
            reconstructed_matrix = pca.inverse_transform(compressed_matrix)  # (time, n_spatial)
            n_time = reconstructed_matrix.shape[0]
            arr = reconstructed_matrix.reshape(n_time, *spatial_shape)

            # Rebuild DataArray
            # Re-create coordinate dims (time + whatever spatial dims existed)
            time_coords = compressed_ds["time"].values
            non_time_dims = meta.get("non_time_dims", [f"dim_{i}" for i in range(len(spatial_shape))])
            orig_coords = meta.get("orig_coords", {})
            coords = {"time": time_coords}
            for d in non_time_dims:
                if d in orig_coords:
                    coords[d] = orig_coords[d]

            reconstructed_da = xr.DataArray(
                arr.astype(np.float32),
                dims=["time"] + non_time_dims,
                coords=coords,
                name=var,
            )
            reconstructed[var] = reconstructed_da

        return xr.Dataset(reconstructed)

    def save(self, path: str):
        """Saves the full compressor (all PCA models + metadata) to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump({"models": self.pca_models, "metadata": self.metadata,
                         "variance_threshold": self.variance_threshold}, f)
        log.info(f"PCA compressor saved: {path}")

    @classmethod
    def load(cls, path: str) -> "ERA5PCACompressor":
        """Loads a previously saved compressor from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(variance_threshold=data["variance_threshold"])
        obj.pca_models = data["models"]
        obj.metadata = data["metadata"]
        log.info(f"PCA compressor loaded: {path}")
        return obj
