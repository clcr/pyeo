"""
Terrain correction
==================

Functions for implementing terrain correction algorithm (credit to Wim Nursal and team, LAPAN)

Original code at https://github.com/Forests2020-Indonesia/Topographic-Correction/blob/master/Topographic%20Correction.py
Method b: https://ieeexplore.ieee.org/document/8356797
"""

from osgeo import gdal
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

import pyeo.windows_compatability

import pdb

gdal.UseExceptions()

DEBUG_PROJECTION = None
DEBUG_GT = None


def download_dem():
    #"""Downloads a DEM (probably JAXA) for the relevent area (maybe)"""
    # Maybe later.
    pass


def get_dem_slope_and_angle(dem_path, slope_out_path, aspect_out_path):
    """Produces two .tifs, slope and aspect, from an imput DEM.
    Assumes that the DEM is in meters."""
    log.info("Calculating slope and apsect rasters from {}".format(dem_path))
    dem = gdal.Open(dem_path)
    slope_options = gdal.DEMProcessingOptions(slopeFormat="degree", scale=111139)  # Scale from latlon to meters
    gdal.DEMProcessing(slope_out_path, dem, "slope", options=slope_options)  # For DEM in latlon
    aspect_options = gdal.DEMProcessingOptions(zeroForFlat=True, scale=111139)
    gdal.DEMProcessing(aspect_out_path, dem, "aspect", options=aspect_options)
    dem = None


def get_pixel_latlon(raster, x, y):
    """For a given pixel in raster, gets the lat-lon value in EPSG 4326."""
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
    return np.fromiter((lat, lon), np.half)


def _generate_latlon_arrays(array, transformer, geotransform):
    
    def generate_latlon_for_here(x, y):
        return generate_latlon(x, y, geotransform, transformer)

    # The crude way. Ask someone if it is doable.
    # Turns out it's not, at least not this way.
    # Time for more linspace!


    latlon_array = np.empty((array.size, 2))
    x_list = np.arange(array.shape[0])
    y_list = np.arange(array.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_list, y_list)
    x_mesh = x_mesh.ravel()
    y_mesh = y_mesh.ravel()
    latlon_array[...,:] = list(map(generate_latlon_for_here, x_mesh, y_mesh))
    return latlon_array[0], latlon_array[1]


def generate_slope_and_aspect_rasters(dem_raster_path, out_directory):
    """Using GDAL, splits a DEM into slope and aspect rasters"""
    #global slope_raster_path, aspect_raster_path, dem_image, dem_array    # Get rid of this later
    log.info("Calculating slope and aspect rasters")
    slope_raster_path = p.join(out_directory, 'slope.tif')
    aspect_raster_path = p.join(out_directory, 'aspect.tif')
    get_dem_slope_and_angle(dem_raster_path, slope_raster_path, aspect_raster_path)
    return slope_raster_path, aspect_raster_path


def calculate_ic_array(slope_raster_path, aspect_raster_path, raster_datetime, ic_raster_out_path = None):
    """
    Given a slope and an aspect raster, creates an array of the illumination conditions as specified in
    https://ieeexplore.ieee.org/document/8356797, equation 9. The Pysolar library is
    used to calculate solar position.

    Parameters
    ----------
    dem_raster_path
        The path to a raster containing the DEM in question
    raster_datetime
        The time of day _with timezone set_ for the
    ic_raster_out_path
        If present, saves a raster of the illumination condition

    Returns
    -------
    An array of illuminatiton conditions for every pixel in the input DEM.
    Each pixel is a value between -1 and 1

    """

    with TemporaryDirectory() as td:
        slope_image = gdal.Open(slope_raster_path)
        slope_array = slope_image.ReadAsArray()   # This is returned, so we can't use GetVirtualMemArray()
        aspect_image = gdal.Open(aspect_raster_path)
        aspect_array = aspect_image.GetVirtualMemArray()

        print("Calculating latlon arrays (this takes a while, for some reason.")
        transformer, geotransform = _generate_latlon_transformer(slope_image)
        lat_array, lon_array = _generate_latlon_arrays(aspect_array, transformer, geotransform)

        print("pixels to process: {}".format(np.product(lat_array.shape)))
        ic_array, zenith_array = ic_calculation(lat_array, lon_array, aspect_array, slope_array, raster_datetime)
        
        if ic_raster_out_path:
            ras.save_array_as_image(ic_array, ic_raster_out_path, aspect_image.GetGeoTransform(), aspect_image.GetProjection())
        return ic_array, zenith_array, slope_array


def calc_azimuth_array(lat_array, lon_array, raster_datetime):
    def calc_azimuth_for_datetime(lat, lon):
        return solar.get_azimuth_fast(lat, lon, raster_datetime).astype(np.dtype('float32'))
    return np.array(list(map(calc_azimuth_for_datetime, lat_array, lon_array)))


def calc_altitude_array(lat_array, lon_array, raster_datetime):
    def calc_altitude_for_datetime(lat, lon):
        return solar.get_altitude_fast(lat, lon, raster_datetime).astype(np.dtype('float32'))
    return np.array(list(map(calc_altitude_for_datetime, lat_array, lon_array)))


def ic_calculation(lat_array, lon_array, aspect_array, slope_array, raster_datetime):
    print("Precomputing azimuth, altitude and and zenith arrays")
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


def build_sample_array(raster_array, slope_array, red_band_index, ir_band_index):
    """
    Returns a set of pixels in raster with slope > 18deg + ndvi > 0.5
    """

    red_band = raster_array[red_band_index, ...]
    ir_band = raster_array[ir_band_index, ...]
    ndvi_array = (ir_band - red_band)/(ir_band + red_band)
    np.nan_to_num(ndvi_array, copy=False)
    mask_array = np.logical_and(ndvi_array > 0.5, slope_array > 18)
    return ras.apply_array_image_mask(raster_array, mask_array, fill_value=0)


def do_terrain_correction(raster_path, dem_path, out_raster_path, raster_datetime, is_landsat=False):

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
        A datetime.DateTime object **with timezone set**

    """
   
    with TemporaryDirectory() as td:

        in_raster = gdal.Open(raster_path)
        in_array = in_raster.GetVirtualMemArray()
        out_raster = ras.create_matching_dataset(in_raster, out_raster_path, bands=in_raster.RasterCount,
                                                 datatype=gdal.GDT_Float32)
        out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)

        # These are being saved for convenience of ras.save_array_as_image
        # Using globals like this is a Bad Idea in real code
        global DEBUG_PROJECTION
        global DEBUG_GT
        DEBUG_PROJECTION = in_raster.GetProjection()
        DEBUG_GT = in_raster.GetGeoTransform()

        print("Preprocessing DEM")
        # If we resample then extract slope and angle, then they have Weird Holes in them that correspond to the centre
        # of pixels. So we need to extract then preprocess -_-
        slope_raster_path, angle_raster_path = generate_slope_and_aspect_rasters(dem_path, td)
        slope_raster_path = preprocess_dem(slope_raster_path, raster_path, td)
        angle_raster_path = preprocess_dem(angle_raster_path, raster_path, td)
        ic_array, zenith_array, slope_array = calculate_ic_array(slope_raster_path, angle_raster_path, raster_datetime,
                                                                 "ic_array.tif")
        
        if is_landsat:
            in_array = in_array.T

        if len(in_array.shape) == 2:
            in_array = np.expand_dims(in_array, 0)

        if is_landsat:
            print("Calculating reflectance array")
            # Oh no, magic numbers. I think these were from the original paper? Are they for Landsat?
        #ref_multi_this_band = 2.0e-5
        #ref_add_this_band = -0.1
        #ref_array = (ref_multi_this_band * in_array + ref_add_this_band) / _deg_cos(zenith_array.T)
        else:
            print("Meatball reflectance test")
            ref_array = np.divide(in_array,10000, dtype=np.half)   # This number straight from Sahid.
            #ref_array = in_array

        print("Calculating sample array")
        sample_array = build_sample_array(ref_array, slope_array, red_band_index=2, ir_band_index=3)
        ras.save_array_as_image(sample_array.astype(np.short), 'sample_array.tif', DEBUG_GT, DEBUG_PROJECTION)
        band_indicies = sample_array[0, ...].nonzero()

        print("Beginning linear regression")
        for i, band in enumerate(sample_array[:, ...]):
            print("Processing band {} of {}".format(i+1, ref_array.shape[0]))
            out_array[i, ...] = correct_reflectance(band, band_indicies, i, ic_array, ref_array, zenith_array)

    out_array = None
    out_raster = None
    ref_array = None
    in_raster = None


def preprocess_dem(dem_path, raster_path, out_directory):
    in_raster = gdal.Open(raster_path)
    dem_name = p.basename(dem_path).partition('.')[0]
    clipped_dem_path = p.join(out_directory, "{}_clipped.tif".format(dem_name))
    reproj_dem_path = p.join(out_directory, "{}_reproj.tif".format(dem_name))
    ras.reproject_image(dem_path, reproj_dem_path, in_raster.GetProjection(), do_post_resample=False)
    ras.resample_image_in_place(reproj_dem_path, in_raster.GetGeoTransform()[1])  # Assuming square pixels
    ras.align_image_in_place(reproj_dem_path, raster_path)
    ras.clip_raster_to_intersection(reproj_dem_path, raster_path, clipped_dem_path)
    return clipped_dem_path


def correct_reflectance(band, band_indicies, i, ic_array, ref_array, zenith_array):
    import joblib
    ic_for_linregress = ic_array[band_indicies[0], band_indicies[1]].ravel()
    band_for_linregress = band[band_indicies[0], band_indicies[1]].ravel()
    slope, _, _, _, _ = stats.linregress(ic_for_linregress, band_for_linregress)
    corrected_band = (band - (slope * (ic_array - _deg_cos(zenith_array))))
    joblib.dump(ic_for_linregress, f"{i}_ic")
    joblib.dump(band_for_linregress, f"{i}_band")
    return np.where(band > 0, corrected_band, ref_array[i, ...])
