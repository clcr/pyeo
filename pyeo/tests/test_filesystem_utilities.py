import os

import pyeo.filesystem_utilities


def test_get_preceding_image():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_dir = "test_data/L2"
    test_image_name = "S2B_MSIL2A_20180713T172709_N0206_R012_T13QFB_2018073T192359.SAFE"
    assert pyeo.filesystem_utilities.get_preceding_image_path(test_image_name, test_dir) == "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_l2_validation():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_safe = "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE"
    assert pyeo.filesystem_utilities.check_for_invalid_l2_data(test_safe) == 1
    test_wrong = "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031550.SAFE"
    assert pyeo.filesystem_utilities.check_for_invalid_l2_data(test_wrong) == 0

