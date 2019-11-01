import sys

def test_windows_monkeypatch(monkeypatch):
    # monkeypatch.setattr(sys,'platform','win32')
    from pyeo import raster_manipulation as ras
    test_rasters = [
            "test_data/all.tif",
            "test_data/original_all.tif"
            ]
    ras.stack_images(test_rasters, "test_outputs/windows_stack.tif")


