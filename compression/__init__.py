# compression/__init__.py
"""ERA5 data compression pipeline: missing values, outlier detection, PCA, event extraction."""

from compression.missing_values import fill_missing_values
from compression.outlier_detection import detect_and_remove_outliers
from compression.pca_compressor import ERA5PCACompressor
from compression.event_extractor import extract_significant_events
from compression.validator import validate_compression
from compression.windowing import compute_rolling_windows

__all__ = [
    "fill_missing_values",
    "detect_and_remove_outliers",
    "ERA5PCACompressor",
    "extract_significant_events",
    "validate_compression",
    "compute_rolling_windows",
]
