import sys
import gdal
import numpy
import os

def test_windows_monkeypatch(monkeypatch):
    # monkeypatch.setattr(sys,'platform','win32')
    try:
        os.remove("test_outputs/windows_stack.tif")
    except FileNotFoundError:
        pass
    from pyeo import raster_manipulation as ras
    test_rasters = [
            "test_data/landsat_8_data/RASTER/band2_reproj.tif",
            "test_data/landsat_8_data/RASTER/band3_reproj.tif"
            ]
    ras.stack_images(test_rasters, "test_outputs/windows_stack.tif")
    out = gdal.Open("test_outputs/windows_stack.tif")
    out_array = out.ReadAsArray()
    assert out_array.max() > 0
