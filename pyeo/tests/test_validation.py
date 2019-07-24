from pyeo import validation
from pyeo.filesystem_utilities import init_log
import numpy as np
import gdal
import os
import pytest


def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))



@pytest.mark.hi_mem
def test_stratified_random_sample():
    image_path = r"test_data/class_composite_T36MZE_20190509T073621_20190519T073621_clipped.tif"
    points = validation.stratified_random_sample(
        map_path=image_path,
        class_sample_count={
            1: 300,
            2: 10,
            3: 10,
            4: 10,
            5: 10,
            6: 10,
            7: 10
        },
        no_data="0"
    )
    assert len(points) == 100
    assert len(points[50]) == 2


@pytest.mark.hi_mem
def test_produce_stratifed_validation_points():
    image_path = r"test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"
    out_path = r"test_outputs/strat_sample_test/strat_sample_test.shp"
    validation.produce_stratified_validation_points(image_path, 500, out_path, no_data=0)


@pytest.mark.hi_mem
def test_get_class_point_lists():
    image_array = np.array([0, 1, 2, 4]*600)
    image_array = np.reshape(image_array, (600, 4))
    class_point_dict = validation.build_class_dict(image_array)
    assert class_point_dict
    print(class_point_dict)


def test_convert_point_list_to_shapefile():
    image_path = r"test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"
    out_path = r"test_outputs/conversion_test/conversion_test.shp"
    point_list = {
        1:[(0, 50)],
        2:[(2000, 0)],
        3:[(1000, 500)],
        4:[(750, 2000)]
    }
    image = gdal.Open(image_path)
    gt = image.GetGeoTransform()
    proj = image.GetProjection()
    validation.save_point_list_to_shapefile(point_list, out_path, gt, proj)
    assert os.path.exists(out_path)


def test_point_allocation():
    #TODO: Get some working numbers for this from Qing.
    se_expected_overall = 0.01  # the standard error of the estimated overall accuracy that we would like to achieve
    U = {'defore': 0.7, 'gain': 0.6, 'stable_forest': 0.9,
         'stable_nonforest': 0.95}  # user's uncertainty for each class (estimated)
    pixel_numbers = {'defore': 200000, 'gain': 150000, 'stable_forest': 3200000, 'stable_nonforest': 6450000}
    total_sample_size = 641
    required_sd = 0.05  # expected user's accuracy for each class - this is larger than the sd of overall accuracy
    required_val = required_sd ** 2  # variance is the root of standard error
    allocate_sample = validation.allocate_category_sample_sizes(total_sample_size=total_sample_size, user_accuracy=U,
                                                                class_total_sizes=pixel_numbers, variance_tolerance=required_val,
                                                                allocate_type='olofsson')
    assert allocate_sample   #
    print(allocate_sample)


def test_cal_total_sample_size():
    """Test samples are from P. Olofsson et al: Good Practices for estimating area and assessing accuracy
    of land change, last paragraph of section 5.1.1"""
    target_se = 0.01
    user_accuracy = {
        "foo": 0.7,
        "bar": 0.6,
        "baz": 0.9,
        "blob": 0.95
    }
    total_class_sizes = {
        "foo": 200000,
        "bar": 150000,
        "baz": 3200000,
        "blob": 6450000
    }
    target = 641
    out = validation.cal_total_sample_size(target_se, user_accuracy, total_class_sizes)
    assert out == target


def test_part_fixed_value_sampling():
    """Test samples are from P. Olofsson et al: Good Practices for estimating area and assessing accuracy
    of land change, table 5/section 5.1.2, 2nd paragrah"""
    # NOTE: This fails for both me and Qing, and we can't figure out why. It's close, though; within a few points.
    #-John
    class_samples = {
        "foo": 100,
        "bar": 100,
        "baz": None,
        "blob": None
    }
    total_class_sizes = {
        "foo": 200000,
        "bar": 150000,
        "baz": 3200000,
        "blob": 6450000
    }
    target_points = 641
    target = {
        "foo": 100,
        "bar": 100,
        "baz": 149,
        "blob": 292
    }
    out = validation.part_fixed_value_sampling(class_samples, total_class_sizes, target_points)
    assert sum(out.values()) == target_points
    assert target == out


def test_maths():
    class_samples = {
        "foo": 100,
        "bar": 100,
        "baz": None,
        "blob": None
    }
    total_class_sizes = {
        "foo": 200000,
        "bar": 150000,
        "baz": 3200000,
        "blob": 6450000
    }
    user_accuracies = {
        "foo": 0.7,
        "bar": 0.6,
        "baz": 0.9,
        "blob": 0.95
    }
    class_sample_counts = {
        "foo": 100,
        "bar": 100,
        "baz": 149,
        "blob": 292
    }

    sample_weights = validation.cal_w_all(total_class_sizes)
    overall_accuracy = validation.cal_sd_for_overall_accuracy(
        weight_dict=sample_weights,
        u_dict=user_accuracies,
        sample_size_dict=class_sample_counts
    )
    np.testing.assert_allclose(overall_accuracy, 0.011, atol=1e-3)
    foo_accuracy = validation.cal_sd_for_user_accuracy(
        user_accuracies["foo"],
        class_sample_counts["foo"]
    )
    baz_accuracy = validation.cal_sd_for_user_accuracy(
        user_accuracies["baz"],
        class_sample_counts["baz"]
    )
    # Values from Olafsson table 7
    np.testing.assert_allclose(foo_accuracy, 0.046, atol=1e-3)
    np.testing.assert_allclose(baz_accuracy, 0.025, atol=1e-3)
