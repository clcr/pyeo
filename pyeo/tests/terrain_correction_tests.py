
import os
import pyeo.pyeo as pyeo
import pyeo.pyeo.terrain_correction as terrain_correction
import gdal
import pathlib
import numpy as np
import pytest
import datetime as dt

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
    pyeo.terrain_correction(input_image_path, output_image_path)
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


def test_calculate_fractional_year():
    test_1_dt = dt.datetime(1990, 1, 1, 12, 0, 0)
    assert terrain_correction.calculate_fractional_year(test_1_dt) == 0
    test_2_dt = dt.datetime(1990, 7, 2, 12, 0, 0)   #Halfway through the year
    np.testing.assert_allclose(terrain_correction.calculate_fractional_year(test_2_dt), np.pi, 1e-2)
    test_2_dt = dt.datetime(1990, 12, 31, 12, 0, 0)   #End of year
    np.testing.assert_allclose(terrain_correction.calculate_fractional_year(test_2_dt), 2*np.pi, 1e-2)


def test_calculate_declination_angle():
    # Declination angle for noon on the 1st Jan 1990 (fractional_year = 0)
    # From https://www.esrl.noaa.gov/gmd/grad/solcalc
    np.testing.assert_allclose(terrain_correction.calculate_declination_angle(0), -23.01, 1e-1)


def test_calculate_eqtime():
    # Equation of time for noon on the 1st Jan 1990 (fractional_year = 0)
    # Target value from https://www.esrl.noaa.gov/gmd/grad/solcalc on the 1st Jan 1990
    #NOTE: Sahid said there was a discrepency between eqtime from paper and eqtime from solarcalc
    np.testing.assert_allclose(terrain_correction.calculate_eqtime(0), -3.53, 1e-1)


def test_calculate_time_offset():
    # Time offset at 0,0 on the 1st Jan, 1990.
    # This should be degenerate, and shake out to eqtime: everything else is 0
    assert terrain_correction.calculate_time_offset(-3.53, 0, 0) == -3.53


def test_calculate_true_solar_time():
    # TST at 0,0 on 1st Jan, 1990
    # https://www.esrl.noaa.gov/gmd/grad/solcal for time_offset and toarget value (true solar noon)
    target = (21*60) + 3 + (32/60)
    test_1_dt = dt.datetime(1990, 1, 1, 12, 0, 0)
    np.testing.assert_allclose(terrain_correction.calculate_true_solar_time(test_1_dt, -3.53), target)


def test_calcuate_hour_angle():
    # Need to find a test value for this
    print(terrain_correction.calculate_hour_angle(-3.53))


def test_calculate_solar_zenith():
    # Test values from
    ha = -180.8825
    lat = 0
    dec = -23.01
    target = 90 - 66.99  # From https://www.esrl.noaa.gov/gmd/grad/solcalc

    out = terrain_correction.calculate_solar_zenith(ha, lat, dec)
    np.testing.assert_allclose(out, target, 1e-1)


def test_calculate_sun_position():
    expected_output = {
        "solar_zenith_angle": 45.627,
        "solar_azimuth_angle": 142.83,
        "solar_elevation_angle": 44.39
    }
    actual_output = terrain_correction.calculate_sun_position(
        latitude=13.0421,
        longitude=100.4726,
        timezone=7,
        local_datetime=dt.datetime(2008, 12, 18, 10, 22, 28)
    )
    assert expected_output == actual_output
