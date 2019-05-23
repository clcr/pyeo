"""Functions for implementing terrain correction algorithm (credit to Wim Nursal, LAPAN"""

import gdal
from tempfile import TemporaryDirectory
import os.path as p
import pyeo.pyeo.core as core
import numpy as np
import datetime as dt
import calendar


def do_terrain_correction(in_safe_file, out_path, dem_path):
    """Corrects for shadow effects due to terrain features.
    Takes in a L2 SAFE file and a DEM, and produces a really boring image.
    Algorithm:
    -Generate slope and aspect from DEM using gdaldem
    -Calculate solar position from datatake sensing start and location of image
    -Calculate the correction factor for that image from the sun zenith angle, azimuth angle, DEM aspect and DEM slope
    -Build a mask of green areas using NDVI
    -Perform a linear regression based on that IC calculation and the contents of the L2 image to get ground slope(?)
    -Correct pixel p in original image with following: p_out = p_in - (ground_slope*(IC-cos(sun_zenith)))
    -Write to output
    NOTE TO SELF: Watch out; Wim has recast the bands to float. Beware off-by-one and rounding errors"""
    with TemporaryDirectory() as td:
        slope_dem_path = p.join(td, "slope.tif")
        aspect_dem_path = p.join(td, "aspect.tif")

        get_dem_slope_and_angle(dem_path, slope_dem_path, aspect_dem_path)


def download_dem():
    #"""Downloads a DEM (probably JAXA) for the relevent area (maybe)"""
    # Maybe later.
    pass


def get_dem_slope_and_angle(dem_path, slope_out_path, aspect_out_path):
    """Produces two .tifs, slope and aspect, from an imput DEM.
    Assumes that the DEM is in meters."""
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(slope_out_path, dem, "slope", scale=111120)  # For DEM in meters
    gdal.DEMProcessing(aspect_out_path, dem, "aspect")


def calculate_granule_solar_position(safe_file):
    """Returns the sun zenith angle and azimuth angle from a L2 .SAFE file.
    NOTE: Only difference between this and Landsat seems to be that Landsat uses UTC"""
    sensing_dt = core.get_image_acquisition_time(p.basename(safe_file))  # Keep a close eye on the timezone
    gamma = calculate_fractional_year(sensing_dt)
    decl = calculate_declination_angle(gamma)


def calculate_fractional_year(sensing_dt):
    """Calculates the faction of the year from a datetime in radians.
    See https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF"""
    year, _, _, hour, minute, second, _, day_of_year, _ = sensing_dt.timetuple()
    A = (2 * np.pi) / days_in_year(year)
    B = (day_of_year - 1 + ((hour - 12) / 24))  # Wim was using minutes and seconds here, does it matter that I'm not?
    gamma = A*B
    return gamma


def calculate_declination_angle(gamma):
    """Given a fractional year in radians (gamma) calculates the sun declination angle in degrees.
    See https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF"""
    decl = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) \
           + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)  # radians
    decl_deg = np.rad2deg(decl)
    return decl_deg


def calculate_sun_zenith(latitude, longitude, decl_deg, eqtime, sensing_dt):
    """Returns the sun zenith angle in degrees"""
    time_offset = eqtime + 4*longitude  # minutes. Can ignore timezone as S2 stamps are in UTC
    true_solar_time = sensing_dt.hour*60 + sensing_dt.minute - sensing_dt.second/60 + time_offset
    solar_hour_angle = (true_solar_time/4) - 180
    zenit1 = np.sin(latitude) * np.sin(decl_deg) + np.cos(latitude) * np.cos(decl_deg) * np.cos(solar_hour_angle)
    zenit2 = np.arccos(zenit1)  # radians
    return np.rad2deg(zenit2)


def calculate_eqtime(gamma):
    """Given a fractional year in radians (gamma), calulates the equation of time in minutes.
    See  https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF"""
    # np trig funcs expect radians
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    return eqtime


def days_in_year(year):
    if calendar.isleap(year):
        return 366
    else:
        return 365