import os
import shutil

import gdal
import pytest

import pyeo.filesystem_utilities
import pyeo.raster_manipulation
from pyeo.tests.utilities import load_test_conf


@pytest.mark.slow
def test_preprocessing():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/L2")
    except FileNotFoundError:
        pass
    conf = load_test_conf()
    pyeo.raster_manipulation.atmospheric_correction("test_data/L1", "test_outputs/L2",
                                                    conf['sen2cor']['path'])
    assert os.path.isfile(
        "test_outputs/L2/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.SAFE/GRANULE/L2A_T13QFB_A004328_20180103T172711/IMG_DATA/R10m/T13QFB_20180103T172709_B08_10m.jp2"
    )
    assert os.path.isfile(
        "test_outputs/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE/GRANULE/L2A_T13QFB_A014452_20180329T173239/IMG_DATA/R10m/T13QFB_20180329T173239_B08_10m.jp2"
    )


def test_merging():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif")
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.msk")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.msk")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.preprocess_sen2_images("test_data/L2/", "test_outputs/", "test_data/L1/", buffer_size=5)
    assert os.path.exists("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif")
    assert os.path.exists("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif")
    assert os.path.exists("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.msk")
    assert os.path.exists("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.msk")


def test_stacking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.msk")
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.stack_old_and_new_images(r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif",
                                  r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                                  r"test_outputs")
    image = gdal.Open("test_outputs/T13QFB_20180319T172021_20180103T172709.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    for layer in range(image_array.shape(0)):
        assert image_array[layer, :, :].max > 10


def test_mask_closure():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_mask_path = "test_outputs/test_mask.msk"
    if os.path.exists(out_mask_path):
        os.remove(out_mask_path)
    shutil.copy("test_data/20180103T172709.msk", out_mask_path)
    pyeo.raster_manipulation.buffer_mask_in_place(out_mask_path, 3)


def test_get_l1():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    output = pyeo.filesystem_utilities.get_l1_safe_file(
        "test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
        "test_data/L1"
    )
    assert output == "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_get_l2():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    output = pyeo.filesystem_utilities.get_l2_safe_file(
        "test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
        "test_data/L2"
    )
    assert output == "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_get_tile():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    assert pyeo.filesystem_utilities.get_sen_2_image_tile("test_data/T13QFB_20180103T172709_20180329T171921.tif") == "T13QFB"
    assert pyeo.filesystem_utilities.get_sen_2_image_tile("test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif") == "T13QFB"