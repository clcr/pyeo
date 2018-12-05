"""A set of tests using the data in real_data. This is designed to run slow and keep the test outputs after running.
Notes:
    - Anything in test_data should not be touched and will remain as constant input for inputs
"""
import os, sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..','..')))
import pyeo.core as pyeo
import gdal
import numpy as np
import pytest


def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pyeo.init_log("test_log.log")


def test_mask_buffering():
    """This is a bad test, but never mind"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_masks = [r"test_data/buffered_masks/20180103T172709.msk",
                  r"test_data/buffered_masks/20180319T172021.msk",
                  r"test_data/buffered_masks/20180329T171921.msk"]
    try:
        [os.remove(mask) for mask in test_masks]
    except FileNotFoundError:
        pass
    [shutil.copy(mask, "test_data/buffered_masks/") for mask in
     ["test_data/20180103T172709.msk", "test_data/20180319T172021.msk", r"test_data/20180329T171921.msk"]]
    [pyeo.buffer_mask_in_place(mask, 20) for mask in test_masks]
    assert [os.path.exists(mask) for mask in test_masks]


def test_composite_images_with_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/20180103T172709.tif",
                 r"test_data/20180319T172021.tif",
                 r"test_data/20180329T171921.tif"]
    out_file = r"test_outputs/composite_test.tif"
    pyeo.composite_images_with_mask(test_data, out_file)


def test_buffered_composite():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/buffered_composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/buffered_masks/20180103T172709.tif",
                 r"test_data/buffered_masks/20180319T172021.tif",
                 r"test_data/buffered_masks/20180329T171921.tif"]
    out_file = r"test_outputs/buffered_composite_test.tif"
    pyeo.composite_images_with_mask(test_data, out_file)


@pytest.mark.slow
def test_ml_masking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/ml_mask_test.tif")
        os.remove("test_outputs/ml_mask_test.msk")
    except FileNotFoundError:
        pass
    shutil.copy(r"test_data/20180103T172709.tif", r"test_outputs/ml_mask_test.tif")
    pyeo.create_mask_from_model(
        r"test_outputs/ml_mask_test.tif",
        r"test_data/cloud_model_v0.1.pkl",
        buffer_size=10
    )

@pytest.mark.slow
def test_preprocessing():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    l1_dirs = ["S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
               "S2B_MSIL1C_20180103T172709_N0206_R012_T13QFB_20180103T192359.SAFE"]
    try:
        [shutil.rmtree(os.path.join("test_outputs", l1_dir)) for l1_dir in l1_dirs]
    except FileNotFoundError:
        pass
    [pyeo.atmospheric_correction("test_data/"+l1_dir) for l1_dir in l1_dirs]
    assert os.path.isfile(
        "test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.SAFE/GRANULE/L2A_T13QFB_A004328_20180103T172711/IMG_DATA/R10m/T13QFB_20180103T172709_B08_10m.jp2"
    )
    assert os.path.isfile(
        "test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE/GRANULE/L2A_T13QFB_A014452_20180329T173239/IMG_DATA/R10m/T13QFB_20180329T173239_B08_10m.jp2"
    )


def test_merging():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    l2_dirs = ["S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
               "S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.SAFE"]
    try:
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif")
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.msk")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.msk")
    except FileNotFoundError:
        pass
    pyeo.aggregate_and_mask_10m_bands("test_data/L2/",
                                      "test_outputs/", buffer_size=3)


def test_stacking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.msk")
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.tif")
    except FileNotFoundError:
        pass
    pyeo.stack_old_and_new_images(r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif",
                                  r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                                  r"test_outputs")
    #assert os.path.exists(os.path.abspath(r"test_outputs/T13QFB_20180103T172709_20180319T172021.tif"))
    image = gdal.Open(os.path.abspath("test_outputs/T13QFB_20180319T172021_20180103T172709.tif"))
    image_array = image.GetVirtualMemArray()
    for layer in range(image_array.shape(0)):
        assert image_array[layer, :, :].max > 1000


@pytest.mark.slow
def test_classification():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/class_20180103T172709_20180319T172021.tif")
    except FileNotFoundError:
        pass
    pyeo.classify_image("test_data/T13QFB_20180103T172709_20180329T171921.tif", "test_data/manantlan_v1.pkl",
                        "test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif", num_chunks=4)
    image = gdal.Open("test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert not np.all(image_array == 0)


def test_mask_combination():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    masks = [r"test_data/"+mask for mask in os.listdir("test_data") if mask.endswith(".msk")]
    try:
        os.remove("test_outputs/union_or_combination.tif")
        os.remove("test_outputs/intersection_and_combination.tif")
    except FileNotFoundError:
        pass
    pyeo.combine_masks(masks, "test_outputs/union_or_combination.tif",
                       geometry_func="union", combination_func="or")
    pyeo.combine_masks(masks, "test_outputs/intersection_and_combination.tif",
                       geometry_func="intersect", combination_func="and")
    mask_1 = gdal.Open("test_outputs/union_or_combination.tif")
    assert not mask_1.GetVirtualMemArray().all == False


def test_mask_closure():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_mask_path = "test_outputs/test_mask.msk"
    if os.path.exists(out_mask_path):
        os.remove(out_mask_path)
    shutil.copy("test_data/20180103T172709.msk", out_mask_path)
    pyeo.buffer_mask_in_place(out_mask_path, 30)


if __name__ == "__main__":
    print(sys.path)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log = pyeo.init_log("test_log.log")
    test_mask_combination()
