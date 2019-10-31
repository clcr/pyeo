import sys

def test_windows_monkeypatch(monkeypatch):
    monkeypatch.setattr(sys,'platform','win32')
    from pyeo import raster_manipulation as ras
    test_rasters = [
            "test_data/composite_T36MZE_20190509T073621_20190519T073621.tif",
            "test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"
            ]
    ras.stack_images(test_rasters, "test_outputs/windows_stack.tif")


