
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


@pytest.mark.filterwarnings("ignore:numeric")
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
    lon, lat = terrain_correction._generate_latlon_arrays(array, transformer, gt)
    test_lat = joblib.load("test_data/lat_array_indo")
    test_lon = joblib.load("test_data/lon_array_indo")
    assert np.all(lat == test_lat)
    assert np.all(lon == test_lon)


@pytest.mark.skip
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

#    def mock_latlon(foo, bar, baz):
#        lat = joblib.load("test_data/clipped_lat")
#        lon = joblib.load("test_data/clipped_lon")
#        return lat, lon
 #   monkeypatch.setattr(terrain_correction, "_generate_latlon_arrays", mock_latlon)#

    dem_path = "test_data/dem_test_indonesia.tif"
    in_path = "test_data/indonesia_s2_image_clipped.tif"
    raster_timezone = pytz.timezone("UTC")
    raster_datetime = dt.datetime(2017, 9, 22, 2, 55, 41, tzinfo=raster_timezone)
    out_path = "test_outputs/correction_indonesia.tif"
    terrain_correction.do_terrain_correction(in_path, dem_path, out_path, raster_datetime)


@pytest.mark.filterwarnings("ignore:numeric")
def test_terrain_correction_landsat(monkeypatch):
    dem_path = "test_data/dem_test_indonesia.tif"
    in_path = "test_data/landsat_stack.tif"
    raster_timezone = pytz.timezone("UTC")
    raster_datetime = dt.datetime(2015, 7, 5, 3, 5, 42, tzinfo=raster_timezone)
    out_path = "test_outputs/correction_landsat_indonesia.tif"
    terrain_correction.do_terrain_correction(in_path, dem_path, out_path, raster_datetime, is_landsat=False)
    assert gdal.Open(out_path)


@pytest.mark.filterwarnings("ignore:numeric")
def test_terrain_correction_s2():
    dem_path = "test_data/dem_test_clipped.tif"
    in_path = "test_data/S2A_MSIL1C_20200310T025541_N0209_R032_T48MXU_20200310T062815.tif"
    out_path = "test_outputs/correction_s2_indonesia.tif"
    if os.path.exists(out_path):
        os.remove(out_path)
    raster_timezone = pytz.timezone("UTC")
    raster_datetime = dt.datetime(2020, 3, 10, 2, 55, 41, tzinfo=raster_timezone) # Timezone?
    terrain_correction.do_terrain_correction(in_path, dem_path, out_path, raster_datetime)
    out = gdal.Open(out_path)
    assert out.GetVirtualMemArray().max() > 10


@pytest.mark.filterwarnings("ignore:numeric")
def test_landsat_stacking():
    from pyeo import raster_manipulation as ras
    folder_path = "test_data/landsat_from_usgs/LC08_L1TP_123064_20180102_20180104_01_T1"
    out_image_path = "test_outputs/landsat_stack.tif"
    ras.preprocess_landsat_images(folder_path, out_image_path, new_projection=32748)
    out_raster = gdal.Open(out_image_path)
    assert out_raster
