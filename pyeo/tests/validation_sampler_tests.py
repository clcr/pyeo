from pyeo import validation
import numpy as np
import gdal
import os
import pytest


@pytest.mark.slow
def test_stratified_random_sample():
    image_path = r"test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"
    points = validation.stratified_random_sample(
        map_path = image_path,
        n_points = 100
    )
    assert len(points) == 100
    assert len(points[50]) == 2


def test_get_class_point_lists():
    image_array = np.array([0, 1, 2, 4]*6)
    image_array = np.reshape(image_array, (6, 4))
    class_point_dict = validation.build_class_dict(image_array)
    assert class_point_dict
    print(class_point_dict)


def test_convert_point_list_to_shapefile():
    image_path = r"test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"
    out_path = r"test_outputs/conversion_test/conversion_test.shp"
    point_list = [
        (1000, 1000),
        (2000, 2000),
        (3000, 3000),
        (4000, 4000)
    ]
    image = gdal.Open(image_path)
    gt = image.GetGeoTransform()
    proj = image.GetProjection()
    validation.save_point_list_to_shapefile(point_list, out_path, gt, proj)
    assert os.path.exists(out_path)