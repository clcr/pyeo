
import os

import pyeo.terrain_correction as terrain_correction
import osgeo.gdal as gdal
import pathlib
import numpy as np
import pytest
import datetime as dt
import pysolar

gdal.UseExceptions()


def setup_module():
    os.chdir(pathlib.Path(__file__).parent/"dem_tests")


def test_setup():
    assert pathlib.Path().cwd().is_absolute()
    assert pathlib.Path.cwd().stem == "dem_tests"


@pytest.mark.skip("Not implemented yet")
def test_terrain_correction():
    input_image_path = "test_data/terrain_test_before.tif"
    target_image_path = "test_data/terrain_test_after.tif"
    output_image_path = "test_outputs/terrain_test_output.tif"
    terrain_correction(input_image_path, output_image_path)
    output_image = gdal.Open(output_image_path)
    target_image = gdal.Open(target_image_path)
    assert np.all(output_image.GetVirtualMemArray() == target_image.GetVirtualMemArray())


@pytest.mark.skip("Test is too slow for dev right now")
def test_get_dem_slope_and_angle():
    test_dem_path = r"test_data/N001E109/N001E109_AVE_DSM.tif"
    slope_path = r"test_outputs/slope.dem"
    angle_path = r"test_outputs/angle.dem"
    terrain_correction.get_dem_slope_and_angle(test_dem_path, slope_path, angle_path)
    assert gdal.Open(slope_path)
    assert gdal.Open(angle_path)


def test_get_pixel_latlon():
    # Expected out for  top-left corner of test image, (0,0)
    # Test image is in EPSG 32748, QGIS says that TL corner coords are 600001.8, 9399997.9
    # epsg.io says that this is 35.6942795°, -0.0000193° in latlon
    os.chdir(pathlib.Path(__file__).parent)
    test_image_path = "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE/GRANULE/L2A_T48MXU_A011755_20170922T031450/IMG_DATA/R20m/L2A_T48MXU_20170922T025541_AOT_20m.jp2"
    target_lon = 105.9026743
    target_lat = -5.4275703
    test_image = gdal.Open(test_image_path)
    out_lat, out_lon = terrain_correction.get_pixel_latlon(test_image, 0, 0)
    np.testing.assert_allclose(out_lat, target_lat, 0.001)
    np.testing.assert_allclose(out_lon, target_lon, 0.001)


def test_calculate_illumination_raster():
    slope_ras_path = "something"
    aspect_raster_path = "something_else"
