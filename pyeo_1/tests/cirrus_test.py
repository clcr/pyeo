import pytest
from osgeo import gdal

from pyeo.cirrus_correction import cirrus_correction

def test_cirrus_correction():
    stacked_band_path = "test_data/test_cirrus/T48MYT_20180803T025539_band_RGB_Cirrus.tif"
    out_path = "test_outputs/cirrus_correction.tif"
    cirrus_correction(stacked_band_path, out_path)
    result = gdal.Open(out_path)
    assert result
