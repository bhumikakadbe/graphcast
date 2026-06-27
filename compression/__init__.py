# compression/__init__.py
"""ERA5 data compression pipeline: missing values, outlier detection, PCA, event extraction."""

# NOTE: Python 3.14+ added a stdlib module also named 'compression' (used by lz4 via joblib
# worker processes). To prevent our local package from shadowing it, any code using joblib
# parallelism (n_jobs != 1) must set n_jobs=1 when run from this project root, or run from
# a directory where this package is not on sys.path.
# See: sensitivity_analysis.py permutation_importance n_jobs=1

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
