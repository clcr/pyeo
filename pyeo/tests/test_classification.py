import glob
import os

import gdal
import numpy as np
import pytest

import pyeo.classification


@pytest.mark.slow
def test_classification():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/class_20180103T172709_20180319T172021.tif")
    except FileNotFoundError:
        pass
    pyeo.classification.classify_image("test_data/T13QFB_20180103T172709_20180329T171921.tif", "test_data/manantlan_v1.pkl",
                        "test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif", num_chunks=4)
    image = gdal.Open("test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert not np.all(image_array == 0)


def test_raster_reclass_binary():
    test_image_name = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/cherangany/class_composite_T36NYF_20180112T075259_20180117T075241.tif'
    test_value = 1
    out_fn = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/cherangany/class_composite_T36NYF_20180112T075259_20180117T075241_rcl.tif'
    a = pyeo.classification.raster_reclass_binary(test_image_name, test_value, outFn=out_fn)
    print(np.unique(a) == [0, 1])


def test_raster_reclass_directory():
    test_dir = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/mt_elgon'
    rst_list = glob.glob(os.path.join(test_dir, '*.tif'))
    test_value = 1
    suffix = '_rcl.'
    for i, in_rst in enumerate(rst_list):
        path, fn = os.path.split(in_rst)
        n, fmt = fn.split('.', 2)
        fn = n+suffix+fmt
        out_fn = os.path.join(path, fn)
        pyeo.classification.raster_reclass_binary(in_rst, test_value, outFn=out_fn)