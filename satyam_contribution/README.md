# Uncertainty Quantification for PyEO Forest Alerts

**Author:** Satyam Shah  
**Date:** February 2026  
**Built on:** [PyEO](https://github.com/clcr/pyeo) by CLCR, University of Leicester

---

## What this adds

PyEO's `classify_image()` already produces a per-class probability raster when 
`prob_out_path` is specified. This module takes those probabilities and computes 
two complementary uncertainty metrics for every pixel:

| Metric | Description |
|---|---|
| **Margin Uncertainty** | How close the top two class probabilities are. High value means the model nearly couldn't decide between two classes. |
| **Normalised Entropy** | How spread the full probability distribution is across all classes. High value means the model had no strong preference for any class. |

Both metrics are saved as bands in a single GeoTIFF alongside the existing 
classification outputs. A filtered classification is also produced where 
high-uncertainty pixels are masked out, giving a cleaner, more reliable alert map.

---

## Why this matters

PyEO is used operationally by the Kenya Forest Service and for EU Deforestation 
Regulation supply chain compliance. In both contexts, a forest alert that comes 
with a confidence score is significantly more useful than a binary yes/no — 
field teams can prioritise high-confidence alerts and treat uncertain ones with 
appropriate caution.

---

## Files
```
my_contribution/
├── uncertainty.py            # the module — import and use in your pipeline
└── uncertainty_layer.ipynb   # development notebook with worked examples
```

---

## How to use

### 1. After calling classify_image()
```python
from my_contribution.uncertainty import add_uncertainty_to_pipeline

report = add_uncertainty_to_pipeline(
    prob_out_path          = "path/to/your_prob.tif",       # from classify_image()
    class_out_path         = "path/to/your_class.tif",      # from classify_image()
    uncertainty_out_path   = "path/to/uncertainty_out.tif",
    high_uncertainty_threshold = 0.7                        # adjustable
)

print(report)
```

### 2. Just compute uncertainty from a probability raster
```python
from my_contribution.uncertainty import compute_uncertainty, plot_uncertainty

uncertainty = compute_uncertainty(
    prob_raster_path = "path/to/your_prob.tif",
    out_path         = "path/to/uncertainty_out.tif"
)

plot_uncertainty(uncertainty, out_path="uncertainty_map.png")
```

---

## Output

### Uncertainty raster
A 2-band GeoTIFF with the same projection and extent as the input:
- Band 1 — Margin Uncertainty (float32, 0 to 1)
- Band 2 — Normalised Entropy (float32, 0 to 1)

### Filtered classification
A copy of the classification raster with high-uncertainty pixels set to nodata, 
saved as `[original_name]_uncertainty_filtered.tif`

### Report dictionary
```python
{
    "total_valid_pixels"      : 40000,
    "high_uncertainty_pixels" : 8241,
    "pct_high_uncertainty"    : 20.6,
    "mean_margin_uncertainty" : 0.312,
    "mean_entropy"            : 0.289,
    "filtered_class_path"     : "..._uncertainty_filtered.tif",
    "uncertainty_raster_path" : "...uncertainty_out.tif",
    "threshold_used"          : 0.7
}
```

---

## Dependencies

No new dependencies — uses only packages already present in PyEO's environment:
- numpy
- gdal (osgeo)
- matplotlib
- logging