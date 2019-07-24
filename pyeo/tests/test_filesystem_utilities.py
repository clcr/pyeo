import os

import pyeo.filesystem_utilities


def test_get_preceding_image():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_dir = "test_data/L2"
    test_image_name = "S2B_MSIL2A_20180713T172709_N0206_R012_T13QFB_2018073T192359.SAFE"
    assert pyeo.filesystem_utilities.get_preceding_image_path(test_image_name, test_dir) == "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"