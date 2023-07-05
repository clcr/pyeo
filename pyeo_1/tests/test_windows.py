import sys
from osgeo import gdal
import os
import pyeo.windows_compatability

def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_windows_stacking(monkeypatch):
    monkeypatch.setattr(pyeo.windows_compatability, 'WINDOWS_PREFIX', '')
    out_file = r"test_outputs/windows_stack.tif"
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass
    from pyeo import raster_manipulation as ras
    test_rasters = [
            r"test_data/indonesia_s2_image.tif",
            r"test_data/indonesia_s2_l1_image.tif"
            ]
    ras.stack_images(test_rasters, out_file)
    out = gdal.Open(out_file)
    out_array = out.ReadAsArray()
    assert out_array.max() > 0


def test_windows_class_extraction(monkeypatch):
    monkeypatch.setattr(pyeo.windows_compatability,'WINDOWS_PREFIX','')
    out_file = r"test_outputs/windows_sigs.csv"
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass
    from pyeo import classification as cls
    cls.extract_features_to_csv(
        in_ras_path="/home/localadmin1/maps/indonesia/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.tif",
        training_shape_path="/home/localadmin1/maps/indonesia/training_Indonesia.shp",
        out_path=out_file,
        attribute="Id"
    )
