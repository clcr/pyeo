"""
uncertainty.py — PyEO Uncertainty Quantification Module
--------------------------------------------------------
Adds per-pixel uncertainty estimation on top of PyEO's existing
classify_image() output.

Two metrics are computed for each pixel:
    - Margin uncertainty : how close the top two class probabilities are
    - Normalised entropy : how spread the full probability distribution is

Both are saved as bands in a GeoTIFF alongside the existing classification
outputs, and a filtered classification is produced with uncertain pixels
masked out.

Author  : Satyam Shah
Date    : February 2026
Built on: PyEO (https://github.com/clcr/pyeo)
"""

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import logging
import os

gdal.UseExceptions()
log = logging.getLogger(__name__)


def compute_uncertainty(prob_raster_path: str, out_path: str) -> np.ndarray:
    """
    Reads a per-class probability raster from PyEO's classify_image() and
    computes per-pixel uncertainty using margin and normalised entropy.

    Parameters
    ----------
    prob_raster_path : str
        Path to multi-band probability .tif from classify_image()
    out_path : str
        Path to save the 2-band uncertainty raster

    Returns
    -------
    uncertainty : np.ndarray
        Shape (2, height, width) — band 0 is margin, band 1 is entropy
    """
    ds = gdal.Open(prob_raster_path)
    n_bands = ds.RasterCount
    width   = ds.RasterXSize
    height  = ds.RasterYSize

    probs = np.stack(
        [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(n_bands)],
        axis=0
    ).astype(np.float32)

    nodata_mask = np.all(probs == 0, axis=0)

    # Margin uncertainty
    sorted_probs = np.sort(probs, axis=0)[::-1]
    margin = 1.0 - (sorted_probs[0] - sorted_probs[1])
    margin[nodata_mask] = 0.0

    # Normalised Shannon entropy
    epsilon  = 1e-10
    entropy  = -np.sum(probs * np.log(probs + epsilon), axis=0)
    entropy  = entropy / np.log(n_bands)
    entropy[nodata_mask] = 0.0

    uncertainty = np.stack([margin, entropy], axis=0)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_path, width, height, 2, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(uncertainty[0])
    out_ds.GetRasterBand(1).SetDescription("Margin Uncertainty")
    out_ds.GetRasterBand(2).WriteArray(uncertainty[1])
    out_ds.GetRasterBand(2).SetDescription("Normalised Entropy")
    out_ds.FlushCache()
    out_ds = None
    ds     = None

    log.info(f"Uncertainty raster saved: {out_path}")
    return uncertainty


def add_uncertainty_to_pipeline(
    prob_out_path: str,
    class_out_path: str,
    uncertainty_out_path: str,
    high_uncertainty_threshold: float = 0.7
) -> dict:
    """
    Sits directly on top of PyEO's classify_image() output.
    Produces an uncertainty raster, a filtered classification map,
    and a summary report.

    Call this immediately after classify_image() in your pipeline.

    Parameters
    ----------
    prob_out_path : str
        Path to probability raster from classify_image()
    class_out_path : str
        Path to classification raster from classify_image()
    uncertainty_out_path : str
        Path to save the uncertainty raster
    high_uncertainty_threshold : float
        Pixels with margin uncertainty above this are flagged. Default 0.7

    Returns
    -------
    report : dict
        Summary statistics about classification confidence
    """
    uncertainty = compute_uncertainty(prob_out_path, uncertainty_out_path)
    margin      = uncertainty[0]
    entropy     = uncertainty[1]

    ds          = gdal.Open(class_out_path)
    class_array = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds          = None

    valid_mask            = class_array > 0
    high_uncertainty_mask = (margin > high_uncertainty_threshold) & valid_mask

    filtered_class = class_array.copy()
    filtered_class[high_uncertainty_mask] = 0

    filtered_path = class_out_path.replace(".tif", "_uncertainty_filtered.tif")
    src_ds = gdal.Open(class_out_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        filtered_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Byte
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(filtered_class.astype(np.uint8))
    out_ds.FlushCache()
    out_ds = None
    src_ds = None

    n_valid       = int(np.sum(valid_mask))
    n_uncertain   = int(np.sum(high_uncertainty_mask))
    pct_uncertain = (n_uncertain / n_valid * 100) if n_valid > 0 else 0

    report = {
        "total_valid_pixels"      : n_valid,
        "high_uncertainty_pixels" : n_uncertain,
        "pct_high_uncertainty"    : round(pct_uncertain, 2),
        "mean_margin_uncertainty" : round(float(margin[valid_mask].mean()), 4),
        "mean_entropy"            : round(float(entropy[valid_mask].mean()), 4),
        "filtered_class_path"     : filtered_path,
        "uncertainty_raster_path" : uncertainty_out_path,
        "threshold_used"          : high_uncertainty_threshold
    }

    log.info(f"Uncertainty report: {report}")
    return report


def plot_uncertainty(uncertainty: np.ndarray, out_path: str = None):
    """
    Plots margin and entropy uncertainty side by side.

    Parameters
    ----------
    uncertainty : np.ndarray
        Shape (2, height, width) as returned by compute_uncertainty()
    out_path : str, optional
        If provided, saves the figure here
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Margin Uncertainty", "Normalised Entropy"]

    for i, ax in enumerate(axes):
        masked = np.ma.masked_where(uncertainty[i] == 0, uncertainty[i])
        im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(titles[i], fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Uncertainty (0=confident, 1=uncertain)")

    plt.suptitle("PyEO Classification Uncertainty Map", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info(f"Uncertainty figure saved: {out_path}")
    plt.show()