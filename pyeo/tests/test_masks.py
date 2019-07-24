import os
import shutil

import gdal
import pytest

import pyeo.masks
import pyeo.sen2_funcs


def test_mask_buffering():
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
    [pyeo.masks.buffer_mask_in_place(mask, 2) for mask in test_masks]
    assert [os.path.exists(mask) for mask in test_masks]


@pytest.mark.slow
def test_ml_masking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/ml_mask_test.tif")
        os.remove("test_outputs/ml_mask_test.msk")
    except FileNotFoundError:
        pass
    shutil.copy(r"test_data/20180103T172709.tif", r"test_outputs/ml_mask_test.tif")
    pyeo.masks.create_mask_from_model(
        r"test_outputs/ml_mask_test.tif",
        r"test_data/cloud_model_v0.1.pkl",
        buffer_size=10
    )


def test_mask_combination():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    masks = [r"test_data/"+mask for mask in os.listdir("test_data") if mask.endswith(".msk")]
    try:
        os.remove("test_outputs/union_or_combination.tif")
        os.remove("test_outputs/intersection_and_combination.tif")
    except FileNotFoundError:
        pass
    pyeo.masks.combine_masks(masks, "test_outputs/union_or_combination.tif",
                             geometry_func="union", combination_func="or")
    pyeo.masks.combine_masks(masks, "test_outputs/intersection_and_combination.tif",
                             geometry_func="intersect", combination_func="and")
    mask_1 = gdal.Open("test_outputs/union_or_combination.tif")
    assert not mask_1.GetVirtualMemArray().all == False
    mask_2 = gdal.Open("test_outputs/intersection_and_combination.tif")
    assert not mask_2.GetVirtualMemArray().all == False


def test_mask_from_confidence_layer():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/confidence_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.masks.create_mask_from_confidence_layer(
        "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/confidence_mask.tif",
        cloud_conf_threshold=0,
        buffer_size=3)


def test_fmask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.sen2_funcs.apply_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/fmask.tif"
    )
    assert gdal.Open("test_outputs/masks/fmask.tif")


def test_fmask_cloud_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.masks.create_mask_from_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/fmask_cloud_and_shadow.tif"
    )


def test_combination_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/combined_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.masks.create_mask_from_sen2cor_and_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/combined_mask.tif")


def test_mask_joining():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/joining_test.tif")
    except FileNotFoundError:
        pass
    pyeo.masks.combine_masks(["test_data/masks/fmask_cloud_and_shadow.tif", "test_data/masks/confidence_mask.tif"],
                       "test_outputs/masks/joining_test.tif", combination_func="and", geometry_func="union")
    out_image = gdal.Open("test_outputs/masks/joining_test.tif")
    assert out_image
    out_array = out_image.GetVirtualMemArray()
    assert 1 in out_array
    assert 0 in out_array