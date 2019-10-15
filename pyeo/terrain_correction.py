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
import numpy as np
import datetime as dt
import calendar
from pysolar import solar
import pytz


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


def calculate_solar_elevation(solar_declination):
    """Trivial function"""
    return 90 - solar_declination


def calculate_sun_position(latitude, longitude, timezone, local_datetime):
    """Stuff it, we're using Pysolar"""

    solar_altitude = solar.get_altitude(latitude, longitude, local_datetime)
    solar_elevation = calculate_solar_elevation(solar_declination)

    out = {
        "solar_zenith_angle": solar_zenith,
        "solar_azimuth_angle": solar_azimuth,
        "solar_elevation_angle": solar_elevation
    }

    return out


def days_in_year(year):
    if calendar.isleap(year):
        return 366
    else:
        return 365


def get_pixel_latlon(raster, x, y):
    """For a given pixel, gets the lat-lon value in EPSG 4326."""
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


def _generate_latlon_arrays(array, transformer, geotransform):
    lat_array = np.empty_like(array, dtype=np.float)
    lon_array = np.empty_like(array, dtype=np.float)
    iterator = np.nditer(lat, flags=['multi_index'])
    while not iterator.finished:
        pixel = reversed(iterator.multi_index)  # pixel_to_point_coords takes y,x for some reason.
        geo_coords = cm.pixel_to_point_coordinates(pixel, geotransform)
        lat, lon = transformer.TransformPoint(*geo_coords)  # U
        lat_array[pixel[1], pixel[0]] = lat
        lon_array[pixel[1], pixel[0]] = lon
        iterator.iternext()
    return lat_array, lon_array


def calculate_illumination_condition_raster(dem_raster_path, raster_datetime):
    with TemporaryDirectory() as td:
        slope_raster_path = p.join(td, "slope.tif")
        aspect_raster_path = p.join(td, "aspect.tif")
        dem_image = gdal.Open(dem_raster_path)
        dem_array = dem_image.GetVirtualMemArray()

        get_dem_slope_and_angle(dem_raster_path, slope_raster_path, aspect_raster_path)
        slope_image = gdal.Open(slope_raster_path)
        slope_array = slope_image.GetVirtualMemArray()
        aspect_image = gdal.Open(aspect_raster_path)
        aspect_array = aspect_image.GetVirtualMemArray()

        transformer, geotrasform = _generate_latlon_transformer(slope_image)
        lat_array, lon_array = _generate_latlon_arrays(slope_array, transformer, geotrasform)

        # Quick meatball vectorisation of pysolar functions
        get_azimuth_array = np.vectorize(
            lambda lat, lon, elevation: solar.get_azimuth(lat, lon, raster_datetime, elevation)
        )
        get_altitude_array = np.vectorize(
            lambda lat, lon, elevation: solar.get_altitude(lat, lon, raster_datetime, elevation)
        )

        azimuth_array = get_azimuth_array(lat_array, lon_array, dem_array)
        altitude_array = get_altitude_array(lat_array, lon_array, dem_array)
        zenith_array = 90 - altitude_array

        #OK, if we're cool we can do this all as array operations.
        IC_array = np.cos(zenith_array)*np.cos(slope_array)+\
                   np.sin(zenith_array)*np.sin(slope_array)*np.cos(azimuth_array-aspect_array)



def calculate_reflectance():
    for y in reflectance_f:  #
        val2 = reflectance_f[y]  #
        temp[y] = val2[a_true, b_true].ravel()  # masked
        IC_true = IC[a_true, b_true].ravel()  # IC masked
        slope = linregress(IC_true, temp[y])
        IC_final[y] = reflectance_f[y] - (slope[0] * (IC - cos(zenit_angle)))
