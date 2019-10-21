
import os
from os import path as p

import pyeo.terrain_correction as terrain_correction
import osgeo.gdal as gdal
import pathlib
import numpy as np
import pytest
import datetime as dt
import pytz
import joblib
from tempfile import TemporaryDirectory

gdal.UseExceptions()


def setup_module():
    os.chdir(pathlib.Path(__file__).parent)


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
    # epsg.io says that this is 105.9026743, -5.4275703 in latlon
    os.chdir(pathlib.Path(__file__).parent)
    test_image_path = "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE/GRANULE/L2A_T48MXU_A011755_20170922T031450/IMG_DATA/R20m/L2A_T48MXU_20170922T025541_AOT_20m.jp2"
    target_lon = 105.9026743
    target_lat = -5.4275703
    test_image = gdal.Open(test_image_path)
    out_lat, out_lon = terrain_correction.get_pixel_latlon(test_image, 0, 0)
    np.testing.assert_allclose(out_lat, target_lat, 0.001)
    np.testing.assert_allclose(out_lon, target_lon, 0.001)


def test_calculate_latlon_array():
    raster_path = "test_data/dem_test_indonesia.tif"
    raster = gdal.Open(raster_path)
    array = raster.GetVirtualMemArray()
    transformer, gt = terrain_correction._generate_latlon_transformer(raster)
    latlon = terrain_correction._generate_latlon_arrays(array, transformer, gt)
    import pdb
    pdb.set_trace()


def test_calculate_illumination_raster(monkeypatch):

    # The generate latlon array function is massively time-consuming.
    # This replaces it with precomputed data.
    def mock_latlon(foo, bar, baz):
        lat = joblib.load("test_data/lat_array_indo")
        lon = joblib.load("test_data/lon_array_indo")
        return lat, lon
    monkeypatch.setattr(terrain_correction, "_generate_latlon_arrays", mock_latlon)

    os.chdir(pathlib.Path(__file__).parent)
    dem_path = "test_data/dem_test_indonesia.tif"
    raster_timezone = pytz.timezone("Asia/Jakarta")
    raster_datetime = dt.datetime(2019, 6, 1, 12, 00, 00, tzinfo=raster_timezone)
    out_path = "test_outputs/illumination_indonesia.tif"
    terrain_correction.calculate_illumination_condition_array(dem_path, raster_datetime, out_path)


@pytest.mark.filterwarnings("ignore:numeric")
def test_terrain_correction(monkeypatch):
    os.chdir(pathlib.Path(__file__).parent)

    def mock_latlon(foo, bar, baz):
        lat = joblib.load("test_data/clipped_lat")
        lon = joblib.load("test_data/clipped_lon")
        return lat, lon
    monkeypatch.setattr(terrain_correction, "_generate_latlon_arrays", mock_latlon)

    dem_path = "test_data/dem_test_indonesia.tif"
    in_path = "test_data/indonesia_s2_image.tif"
    raster_timezone = pytz.timezone("UTC")
    raster_datetime = dt.datetime(2017, 9, 22, 2, 55, 41, tzinfo=raster_timezone)
    out_path = "test_outputs/correction_indonesia.tif"
    terrain_correction.calculate_reflectance(in_path, dem_path, out_path, raster_datetime)


if __name__ == "__main__":
    test_calculate_illumination_raster()
