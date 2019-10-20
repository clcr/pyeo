"""
Terrain correction
==================

Functions for implementing terrain correction algorithm (credit to Wim Nursal, LAPAN

Original code at https://github.com/Forests2020-Indonesia/Topographic-Correction/blob/master/Topographic%20Correction.py
"""

import gdal
from tempfile import TemporaryDirectory
import os.path as p

import osr

import pyeo.filesystem_utilities as fu
import pyeo.coordinate_manipulation as cm
import pyeo.raster_manipulation as ras
import numpy as np
import datetime as dt
import calendar
from pysolar import solar
import pytz

import logging

from joblib import Parallel, delayed
log = logging.getLogger("pyeo")


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
    log.info("Calculating slope and apsect rasters from")
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(slope_out_path, dem, "slope", scale=111120)  # For DEM in meters
    gdal.DEMProcessing(aspect_out_path, dem, "aspect")
    dem=None


def calculate_solar_zenith(hour_angle, latitude, solar_declination):
    """Given the hour angle, latitude and solar declination (all in degrees), returns the solar
    zenith angle in degrees."""
    latitude = np.deg2rad(latitude)
    hour_angle = np.deg2rad(hour_angle)
    solar_declination = np.deg2rad(solar_declination)
    A = np.sin(latitude) * np.sin(solar_declination)
    B = np.cos(latitude) * np.cos(solar_declination) * np.cos(hour_angle)
    theta = np.arccos(A+B)
    return np.rad2deg(theta)


def calculate_solar_azimuth(solar_zenith, latitude, solar_declination):
    """Given the solar zenith in degrees, east-positive latitude and solar declination in degrees"""
    # Again, work in rads
    lat_rad = np.deg2rad(latitude)
    zen_rad = np.deg2rad(solar_zenith)
    dec_rad = np.deg2rad(solar_declination)
    A = (np.sin(lat_rad)*np.cos(zen_rad)) - np.sin(dec_rad)
    B = np.cos(lat_rad)*np.sin(zen_rad)
    C = -1*(A/B)
    theta = np.pi-np.arccos(C)
    out = np.rad2deg(theta)
    return out


def get_pixel_latlon(raster, x, y):
    """For a given pixel, gets the lat-lon value in EPSG 4326."""
    # TODO: Move to coordinate_manipulation
    native_projection = osr.SpatialReference()
    native_projection.ImportFromWkt(raster.GetProjection())
    latlon_projection = osr.SpatialReference()
    latlon_projection.ImportFromEPSG(4326)
    transformer = osr.CoordinateTransformation(native_projection, latlon_projection)

    geotransform = raster.GetGeoTransform()
    x_geo, y_geo = cm.pixel_to_point_coordinates([y,x], geotransform)  # Why did I do this reverse?
    lon, lat, _ = transformer.TransformPoint(x_geo, y_geo)
    return lat, lon


def _generate_latlon_transformer(raster):
    native_projection = osr.SpatialReference()
    native_projection.ImportFromWkt(raster.GetProjection())
    latlon_projection = osr.SpatialReference()
    latlon_projection.ImportFromEPSG(4326)
    geotransform = raster.GetGeoTransform()
    return osr.CoordinateTransformation(native_projection, latlon_projection), geotransform


# This is very slow.
def _generate_latlon_arrays(array, transformer, geotransform):
    lat_array = np.empty_like(array, dtype=np.float)
    lon_array = np.empty_like(array, dtype=np.float)
    iterator = np.nditer(array, flags=['multi_index'])
    while not iterator.finished:
        pixel = list(reversed(iterator.multi_index))  # pixel_to_point_coords takes y,x for some reason.
        geo_coords = cm.pixel_to_point_coordinates(pixel, geotransform)
        lat, lon, _ = transformer.TransformPoint(*geo_coords)  # U
        lat_array[pixel[1], pixel[0]] = lat
        lon_array[pixel[1], pixel[0]] = lon
        iterator.iternext()
    return lat_array, lon_array


def calculate_illumination_condition_raster(dem_raster_path, raster_datetime, ic_raster_out_path):
    """
    Given a DEM, creates a raster of the illumination conditions as specified in
    https://ieeexplore.ieee.org/document/8356797, equation 9. The Pysolar library is
    used to calculate solar position.

    Parameters
    ----------
    dem_raster_path
        The path to a raster containing the DEM in question
    raster_datetime
        The time of day _with timezone set_ for the
    ic_raster_out_path
        The path to save the output raster.


    """
    log.info("Generating illumination condition raster from {}".format(dem_raster_path))
    with TemporaryDirectory() as td:
        log.info("Calculating slope and aspect rasters")
        slope_raster_path = p.join(td, "slope.tif")
        aspect_raster_path = p.join(td, "aspect.tif")
        dem_image = gdal.Open(dem_raster_path)
        dem_array = dem_image.GetVirtualMemArray()

        get_dem_slope_and_angle(dem_raster_path, slope_raster_path, aspect_raster_path)
        slope_image = gdal.Open(slope_raster_path)
        slope_array = slope_image.GetVirtualMemArray()
        aspect_image = gdal.Open(aspect_raster_path)
        aspect_array = aspect_image.GetVirtualMemArray()

        transformer, geotransform = _generate_latlon_transformer(dem_image)
        lat_array, lon_array = _generate_latlon_arrays(dem_array, transformer, geotransform)

        print("pixels to process: {}".format(np.product(lat_array.shape)))

        ic_array = parallel_ic_calculation(lat_array, lon_array, aspect_array, slope_array, raster_datetime)

        ras.save_array_as_image(ic_array, ic_raster_out_path, dem_image.GetGeoTransform(), dem_image.GetProjection())


def calc_azimuth_array(lat_array, lon_array, raster_datetime):
    def calc_azimuth_for_datetime(lat, lon):
        return solar.get_azimuth_fast(lat, lon, raster_datetime)
    return(np.array(list(map(calc_azimuth_for_datetime, lat_array, lon_array))))


def calc_altitude_array(lat_array, lon_array, raster_datetime):
    def calc_altitude_for_datetime(lat, lon):
        return solar.get_altitude_fast(lat, lon, raster_datetime)
    return (np.array(list(map(calc_altitude_for_datetime, lat_array, lon_array))))


def parallel_ic_calculation(lat_array, lon_array, aspect_array, slope_array, raster_datetime):

    print("Precomputing -azimuth and zenith arrays")
    azimuth_array = calc_azimuth_array(lat_array, lon_array, raster_datetime)
    altitude_array = calc_altitude_array(lat_array, lon_array, raster_datetime)
    zenith_array = 90-altitude_array
    print("Beginning IC calculation.")
    ic_array = _deg_cos(np.deg2rad(zenith_array)) * _deg_cos(slope_array) + \
               _deg_sin(zenith_array) * _deg_sin(slope_array) * _deg_cos(azimuth_array - aspect_array)

    return ic_array

def _deg_sin(in_array):
    return np.rad2deg(np.sin(np.deg2rad(in_array)))

def _deg_cos(in_array):
    return np.rad2deg(np.cos(np.deg2rad(in_array)))


def calculate_ic_for_pixel(aspect, raster_datetime, slope, lat, lon, azimuth, altitude):
    zenith = 90 - altitude
    ic = np.cos(zenith) * np.cos(slope) + \
         np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
    return ic


def calculate_reflectance():
    for y in reflectance_f:  #
        val2 = reflectance_f[y]  #
        temp[y] = val2[a_true, b_true].ravel()  # masked
        IC_true = IC[a_true, b_true].ravel()  # IC masked
        slope = linregress(IC_true, temp[y])
        IC_final[y] = reflectance_f[y] - (slope[0] * (IC - cos(zenit_angle)))
