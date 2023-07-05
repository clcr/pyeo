import glob
import os

from osgeo import gdal
import numpy as np
import pytest

import pyeo.classification


@pytest.mark.slow
def test_classification():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/class_composite_T36MZE_20190509T073621_20190519T073621_clipped.tif")
    except FileNotFoundError:
        pass
    pyeo.classification.classify_image("test_data/composite_T36MZE_20190509T073621_20190519T073621_clipped.tif",
                                       "test_data/manantlan_v1.pkl",
                                       "test_outputs/class_composite_T36MZE_20190509T073621_20190519T073621_clipped.tif",
                                       num_chunks=4)
    image = gdal.Open("test_outputs/class_composite_T36MZE_20190509T073621_20190519T073621_clipped.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert not np.all(image_array == 0)


def test_raster_reclass_binary():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_image_name = 'test_data/class_composite_T36MZE_20190509T073621_20190519T073621_clipped.tif'
    test_value = 1
    out_filename = 'test_outputs/class_composite_T36NYF_20180112T075259_20180117T075241_rcl.tif'
    a = pyeo.classification.raster_reclass_binary(test_image_name, test_value, outFn=out_filename)
    assert np.all(np.unique(a) == [0, 1])
