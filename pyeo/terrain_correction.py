"""
Terrain correction
==================

Functions for implementing terrain correction algorithm (credit to Wim Nursal and team, LAPAN)

Original code at https://github.com/Forests2020-Indonesia/Topographic-Correction/blob/master/Topographic%20Correction.py
Method b: https://ieeexplore.ieee.org/document/8356797
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
from scipy import stats

import logging

log = logging.getLogger("pyeo")


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
    dem = None


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


def generate_latlon(x, y,geotransform, transformer):
    x_geo, y_geo = cm.pixel_to_point_coordinates([y,x], geotransform)
    lon, lat, _ = transformer.TransformPoint(x_geo, y_geo)
    return np.fromiter((lat, lon),np.float)

# This is very slow.
def _generate_latlon_arrays(array, transformer, geotransform):
    
    def generate_latlon_for_here(x,y):
        return generate_latlon(x,y,geotransform, transformer)

    latlon_array = np.empty((array.size, 2))
    x_list = np.arange(array.shape[0])
    y_list = np.arange(array.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_list, y_list)
    x_mesh = x_mesh.ravel()
    y_mesh = y_mesh.ravel()
    latlon_array[...,:] = list(map(generate_latlon_for_here, x_mesh, y_mesh)) 
    lat_array = latlon_array[:, 0]
    lon_array = latlon_array[:, 1]
    lat_array = np.reshape(lat_array, array.shape).T
    lon_array = np.reshape(lon_array, array.shape).T
    return lat_array, lon_array

   

def calculate_illumination_condition_array(dem_raster_path, raster_datetime, ic_raster_out_path=None):
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

    Returns
    -------



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

        print("Calculating latlon arrays (this takes a while, for some reason.")
        transformer, geotransform = _generate_latlon_transformer(dem_image)
        lat_array, lon_array = _generate_latlon_arrays(dem_array, transformer, geotransform)

        print("pixels to process: {}".format(np.product(lat_array.shape)))
        ic_array, zenith_array = parallel_ic_calculation(lat_array, lon_array, aspect_array, slope_array, raster_datetime)
        
        if ic_raster_out_path:
            ras.save_array_as_image(ic_array, ic_raster_out_path, dem_image.GetGeoTransform(), dem_image.GetProjection())

        return ic_array, zenith_array


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
    ic_array = _deg_cos(zenith_array) * _deg_cos(slope_array) + \
               _deg_sin(zenith_array) * _deg_sin(slope_array) * _deg_cos(azimuth_array - aspect_array)

    return ic_array, zenith_array


def _deg_sin(in_array):
    return np.sin(np.deg2rad(in_array))


def _deg_cos(in_array):
    return np.cos(np.deg2rad(in_array))


def calculate_reflectance(raster_path, dem_path, out_raster_path, raster_datetime):

    """
    Corrects for shadow effects due to terrain features.
    Algorithm:

    * Generate slope and aspect from DEM using gdaldem
    * Calculate solar position from datatake sensing start and location of image
    * Calculate the correction factor for that image from the sun zenith angle, azimuth angle, DEM aspect and DEM slope
    * Build a mask of green areas using NDVI
    * Perform a linear regression based on that IC calculation and the contents of the L2 image to get ground slope(?)
    * Correct pixel p in original image with following: p_out = p_in - (ground_slope*(IC-cos(sun_zenith)))
    * Write to output

    Parameters
    ----------
    raster_path
        A path to the raster to correct.
    dem_path
        The path to the DEM
    out_raster_path
        The path to the output.
    raster_datetime
        A datetime.DateTime object *with timezone set*


    """
   
    with TemporaryDirectory() as td:

        ref_raster = gdal.Open(raster_path)
        ref_array = ref_raster.GetVirtualMemArray()
        out_raster = ras.create_matching_dataset(ref_raster, out_raster_path, bands=ref_raster.RasterCount)
        out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)

        print("Preprocessing DEM")
        clipped_dem_path = p.join(td, "clipped_dem.tif")
        reproj_dem_path = p.join(td, "reproj_dem.tif")
        ras.reproject_image(dem_path, reproj_dem_path, ref_raster.GetProjection(), do_post_resample=False)
        ras.resample_image_in_place(reproj_dem_path, ref_raster.GetGeoTransform()[1])  # Assuming square pixels
        ras.clip_raster_to_intersection(reproj_dem_path, raster_path, clipped_dem_path)

        ic_array, zenith_array = calculate_illumination_condition_array(clipped_dem_path, raster_datetime)

        if len(ref_array.shape) == 2:
            ref_array = np.expand_dims(ref_array, 0)
        
        import pdb
        print("Beginning linear regression")
        for i, band in enumerate(ref_array[:, ...]):
            print("Processing band {} of {}".format(i+1, ref_array.shape[0]))
            slope, _, _, _, _ = stats.linregress(ic_array.ravel(), band.ravel())
            out_array[i, ...] = (band - (slope*(ic_array - _deg_cos(zenith_array)))).reshape(band.shape)

    out_array = None
    out_raster = None
    ref_array = None
    ref_raster = None
