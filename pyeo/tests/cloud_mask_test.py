import os, sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))
import pyeo.core as pyeo


def test_mask_from_confidence_layer():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/confidence_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.create_mask_from_confidence_layer(
        "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_ouputs/masks/confidence_mask.tif",
        cloud_conf_threshold=0,
        buffer_size=3)


def test_fmask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.apply_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/fmask.tif"
    )

def test_fmask_cloud_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.create_mask_from_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_outputs/masks/fmask_cloud_and_shadow.tif"
    )

def test_combination_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/combined_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.create_combined_sen2_fmask(
        "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE",
        "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"
        "test_outputs/masks/combined_mask.tif"
    )
