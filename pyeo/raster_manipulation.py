"""
pyeo.raster_manipulation
========================
Functions for working with raster data, including masks and platform-specific processing functions.

Key functions
-------------

    :py:func:`create_matching_dataset` Creates an empty raster of the same shape as a source, ready for writing.

    :py:func:`stack_rasters` Stacks a list of rasters into a single raster.

    :py:func:`preprocess_sen2_images` Preprocesses a set of of Sentinel-2 images into single raster files.

    :py:func:`clip_raster` Clips a raster to a shapefile

Rasters
-------

When working with raster data (geotiff, .jp2, ect) using this module, the following assumptions have been made:

* Any function that reads a raster can read from any gdal-readable format
* All interim rasters are stored internally as a geotiff
* All internal rasters have a .tif extension in the filename
* Unless otherwise stated, **all rasters are assumed to be in a projected coordinate system** - i.e. in meters.
  Functions may fail if passed a raster in lat-long projection

Timestamps
----------

Pyeo uses the same timestamp convention as ESA: `yyyymmddThhmmss`; for example, 1PM on 27th December 2020 would be
`20201227T130000`. All timestamps are in UTC

Supported datatypes
-------------------

When a function in this library asks for a datatype, it can take one of the following

.. code:: python

    gdal.GDT_Unknown
    gdal.GDT_Byte
    gdal.GDT_UInt16
    gdal.GDT_Int16
    gdal.GDT_UInt32
    gdal.GDT_Int32
    gdal.GDT_Float32
    gdal.GDT_Float64
    gdal.GDT_CInt16
    gdal.GDT_CInt32
    gdal.GDT_CFloat32
    gdal.GDT_CFloat64




Geotransforms
-------------
Every gdal raster has a geotransform associated with it; this defines it's top-left hand corner in the projection
and pixel size.
In Pyeo, it takes the form of a 6-element tuple; for north-up images, these are the following.

.. code:: python

    geotransform[0] = top_left_x
    geotransfrom[1] = pixel_width
    geotransform[2] = 0
    geotransform[3] = top_left_y
    geotransform[4] = 0
    geotransform[5] = pixel_height

A projection can be obtained from a raster with the following snippet:

.. code:: python

    image = gdal.Open("my_raster.tif")
    gt = image.GetGeoTransform()

For more information, see the following: See the following: https://gdal.org/user/raster_data_model.html#affine-geotransform

Projections
-----------
Each Gdal raster also has a projection, defining (among other things) the unit of the geotransform.
Projections in Pyeo are referred to either by EPSG number or passed around as a wkt( well-known text) string.
You can look up EPSG values at https://epsg.io
You can use the following snippet to extract the well-known text of a raster

.. code:: python

    projection = image.GetProjection()

Masks
-----
Some functions in this module include options for creating and applying masks.

* A raster may have an associated mask

* A mask is a geotif with an identical name as the raster it's masking with a .msk extension

   * For example, the mask for my_sat_image.tif is my_sat_image.msk

* A mask is

   * a single band raster
   * of identical height, width and resolution of the related image
   * contains values 0 or 1

* A mask is applied by multiplying it with each band of its raster

   * So any pixel with a 0 in its mask will be removed, and a 1 will be kept

Function reference
------------------
"""
import sys
import datetime
import glob
import logging
import os
import shutil
import subprocess
import re
from tempfile import TemporaryDirectory, NamedTemporaryFile

import gdal
import numpy as np
from osgeo import gdal_array, osr, ogr
from osgeo.gdal_array import NumericTypeCodeToGDALTypeCode, GDALTypeCodeToNumericTypeCode
from skimage import morphology as morph

import pdb
import faulthandler

from pyeo.coordinate_manipulation import get_combined_polygon, pixel_bounds_from_polygon, write_geometry, \
    get_aoi_intersection, get_raster_bounds, align_bounds_to_whole_number, get_poly_bounding_rect, reproject_vector, \
    get_local_top_left
from pyeo.array_utilities import project_array
from pyeo.filesystem_utilities import sort_by_timestamp, get_sen_2_tiles, get_l1_safe_file, get_sen_2_image_timestamp, \
    get_sen_2_image_tile, get_sen_2_granule_id, check_for_invalid_l2_data, get_mask_path, get_sen_2_baseline, \
    get_safe_product_type
from pyeo.exceptions import CreateNewStacksException, StackImagesException, BadS2Exception, NonSquarePixelException

log = logging.getLogger("pyeo")

import pyeo.windows_compatability
faulthandler.enable()




def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype = None):
    """
    Creates an empty gdal dataset with the same dimensions, projection and geotransform as in_dataset.
    Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified

    Parameters
    ----------
    in_dataset : gdal.Dataset
        A gdal.Dataset object
    out_path : str
        The path to save the copied dataset to
    format : str, optional
        The Ggal image format. Defaults to geotiff ("GTiff"); for a full list, see https://gdal.org/drivers/raster/index.html
    bands : int, optional
        The number of bands in the dataset. Defaults to 1.
    datatype : gdal constant, optional
        The datatype of the returned dataset. See the introduction for this module. Defaults to in_dataset's datatype
        if not supplied.

    Returns
    -------
    new_dataset : gdal.Dataset
        An gdal.Dataset of the new, empty dataset that is ready for writing.

    """
    driver = gdal.GetDriverByName(format)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    out_dataset = driver.Create(out_path,
                                xsize=in_dataset.RasterXSize,
                                ysize=in_dataset.RasterYSize,
                                bands=bands,
                                eType=datatype)
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


def save_array_as_image(array, path, geotransform, projection, format = "GTiff"):
    """
    Saves a given array as a geospatial image to disk in the format 'format'. The datatype will be of one corresponding
    to Array must be gdal format: [bands, y, x].

    Parameters
    ----------
    array : array_like
        A Numpy array containing the values to be saved to a raster
    path : str
        The path to the location to save the output raster to
    geotransform : list
        The geotransform of the image to be saved. See note.
    projection : str
        The projection, as wkt, of the image to be saved. See note.
    format : str, optional
        The image format. Defaults to 'GTiff'; see note for other types.

    Returns
    -------
    path_to_image : str
        The path to the image

    """
    driver = gdal.GetDriverByName(format)
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    # If array is 2d, give it an extra dimension.
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)
    out_dataset = driver.Create(
        path,
        xsize=array.shape[2],
        ysize=array.shape[1],
        bands=array.shape[0],
        eType=type_code
    )
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)
    out_array = out_dataset.GetVirtualMemArray(eAccess=gdal.GA_Update).squeeze()
    out_array[...] = array
    out_array = None
    out_dataset = None
    return path


def create_new_stacks(image_dir, stack_dir):
    """
    For each granule present in image_dir Saves the result in stacked_dir.
    Assumes that each image in image_dir is saved with a Sentinel-2 identifiter name - see merge_raster.

    Parameters
    ----------
    image_dir : str
        A path to the directory containing the images to be stacked, all named as Sentinel 2 identifiers
    stack_dir : str
        A path to a directory to save the stacked images to.
    Returns
    -------
    new_stacks : list of str
        A list of paths to the new stacks

    Notes
    -----
    The pairing algorithm is as follows:
    Step 1: Group directory by tile number
    Step 2: For each tile number, sort by time
    Step 3: For each image in the sorted list, stack each image with it's next oldest image.

    Raises
    ------
    CreateNewStacksException
        If the image directory is empty

    """
    new_images = []
    tiles = get_sen_2_tiles(image_dir)
    tiles = list(set(tiles)) # eliminate duplicates
    n_tiles = len(tiles)
    log.info("Found {} unique tile IDs for stacking:".format(n_tiles))
    for tile in tiles:
        log.info("   {}".format(tile))  # Why in its own loop?
    for tile in tiles:
        log.info("Tile ID for stacking: {}".format(tile))
        safe_files = glob.glob(os.path.join(image_dir, "*" + tile + "*.tif")) # choose all files with that tile ID
        if len(safe_files) == 0:
            raise CreateNewStacksException("Image_dir is empty: {}".format(os.path.join(image_dir, tile + "*.tif")))
        else:
            safe_files = sort_by_timestamp(safe_files)
            log.info("Image file list for pairwise stacking for this tile:")
            for file in safe_files:
                log.info("   {}".format(file))
            # For each image in the list, stack the oldest with the next oldest. Then set oldest to next oldest
            # and repeat
            latest_image_path = safe_files[0]
            for image in safe_files[1:]:
                new_images.append(stack_old_and_new_images(image, latest_image_path, stack_dir))
                latest_image_path = image
    return new_images


def stack_image_with_composite(image_path, composite_path, out_dir, create_combined_mask=True, skip_if_exists=True,
                               invert_stack = False):
    """
    Creates a single 8-band geotif image with a cloud-free composite, and saves the result in out_dir. Images are named
    "composite_tile_timestamp-of-composite_timestamp-of-image". Bands 1,2,3 and 4 are the B,G,R and I bands of the
    composite, and bands 5,6,7 and 8 are the B,G,R and I bands of the image.

    Parameters
    ----------
    image_path : str
        Path to the image to be stacked
    composite_path : str
        Path to the composite to stack the image with
    out_dir : str
        The directory to save the resulting composite to
    create_combined_mask : bool, optional
        If true, combines the cloud mask files associated with the images into a single mask. The mask will mask out
        clouds that exist in either image. Defaults to True.
    skip_if_exists : bool, optional
        If true, skip stacking if a file with the same name is found. Defaults to True.
    invert_stack : bool, optional.
        If true, changes the ordering of the bands to image BGRI - composite BGRI. Included to permit compatibility
        with older models - you can usually leave this alone.

    Returns
    -------
    out_path : str
        The path to the new composite.

    """
    log.info("Stacking {} with composite {}".format(image_path, composite_path))
    composite_timestamp = get_sen_2_image_timestamp(composite_path)
    image_timestamp = get_sen_2_image_timestamp(image_path)
    tile = get_sen_2_image_tile(image_path)
    out_filename = "composite_{}_{}_{}.tif".format(tile, composite_timestamp, image_timestamp)
    out_path = os.path.join(out_dir, out_filename)
    out_mask_path = out_path.rsplit('.')[0] + ".msk"
    if os.path.exists(out_path) and os.path.exists(out_mask_path) and skip_if_exists:
        log.info("{} and mask exists, skipping".format(out_path))
        return out_path
    to_be_stacked = [composite_path, image_path]
    if invert_stack:
        to_be_stacked.reverse()
    stack_images(to_be_stacked, out_path, geometry_mode="intersect")
    if create_combined_mask:
        image_mask_path = get_mask_path(image_path)
        comp_mask_path = get_mask_path(composite_path)
        combine_masks([comp_mask_path, image_mask_path], out_mask_path, combination_func="and", geometry_func="intersect")
    return out_path


def stack_images(raster_paths, out_raster_path,
                 geometry_mode="intersect", format="GTiff", datatype=gdal.GDT_Int32):
    """
    When provided with a list of rasters, will stack them into a single raster. The nunmber of
    bands in the output is equal to the total number of bands in the input. Geotransform and projection
    are taken from the first raster in the list; there may be unexpected behavior if multiple differing
    proejctions are provided.

    Parameters
    ----------
    raster_paths : list of str
        A list of paths to the rasters to be stacked, in order.
    out_raster_path : str
        The path to the saved output raster.
    geometry_mode : {'intersect' or 'union'}
        Can be either 'instersect' or 'union'.
        - If 'intersect', then the output raster will only contain the pixels of the input rasters that overlap.
        - If 'union', then the output raster will contain every pixel in the outputs. Layers without data will
          have their pixel values set to 0.
    format : str, optional
        The GDAL image format for the output. Defaults to 'GTiff'
    datatype : gdal datatype, optional
        The datatype of the gdal array - see introduction. Defaults to gdal.GDT_Int32.

    """
    #TODO: Confirm the union works, and confirm that nondata defaults to 0.
    log.info("Stacking images {}".format(raster_paths))
    if len(raster_paths) <= 1:
        raise StackImagesException("stack_images requires at least two input images")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    total_layers = sum(raster.RasterCount for raster in rasters)
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5]*-1   # Y resolution in affine geotransform is -ve for Maths reasons
    combined_polygons = get_combined_polygon(rasters, geometry_mode)

    # Creating a new gdal object
    out_raster = create_new_image_from_polygon(combined_polygons, out_raster_path, x_res, y_res,
                                               total_layers, projection, format, datatype)

    # I've done some magic here. GetVirtualMemArray lets you change a raster directly without copying
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    present_layer = 0
    for i, in_raster in enumerate(rasters):
        log.info("Stacking image {}".format(i))
        in_raster_array = in_raster.GetVirtualMemArray()
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, combined_polygons)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_raster, combined_polygons)
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        # Gdal does band, y, x
        out_raster_view = out_raster_array[
                      present_layer:  present_layer + in_raster.RasterCount,
                      out_y_min: out_y_max,
                      out_x_min: out_x_max
                      ]
        in_raster_view = in_raster_array[
                    0:in_raster.RasterCount,
                    in_y_min: in_y_max,
                    in_x_min: in_x_max
                    ]
        out_raster_view[...] = in_raster_view
        out_raster_view = None
        in_raster_view = None
        present_layer += in_raster.RasterCount
        in_raster_array = None
        in_raster = None
    out_raster_array = None
    out_raster = None


def strip_bands(in_raster_path, out_raster_path, bands_to_strip):
    """
    Removes bands from a raster and saves a copy.
    Parameters
    ----------
    in_raster_path : str
        Path to the raster
    out_raster_path : str
        Path to the output
    bands_to_strip : list of int
        0-indexed list of bands to remove

    Returns
    -------
    out_path : str
        The path to the output

    """
    in_raster = gdal.Open(in_raster_path)
    out_raster_band_count = in_raster.RasterCount-len(bands_to_strip)
    out_raster = create_matching_dataset(in_raster, out_raster_path, bands=out_raster_band_count)
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    in_raster_array = in_raster.GetVirtualMemArray()

    bands_to_copy = [band for band in range(in_raster_array.shape[0]) if band not in bands_to_strip]

    out_raster_array[...] = in_raster_array[bands_to_copy, :,:]

    out_raster_array = None
    in_raster_array = None
    out_raster = None
    in_raster = None

    return out_raster_path


def average_images(raster_paths, out_raster_path,
                 geometry_mode="intersect", format="GTiff", datatype=gdal.GDT_Int32):
    """
    When provided with a list of rasters, will stack them into a single raster. The nunmber of
    bands in the output is equal to the total number of bands in the input. Geotransform and projection
    are taken from the first raster in the list; there may be unexpected behavior if multiple differing
    proejctions are provided.

    Parameters
    ----------
    raster_paths : list of str
        A list of paths to the rasters to be stacked, in order.
    out_raster_path : str
        The path to the saved output raster.
    geometry_mode : {'intersect' or 'union'}, optional
        Can be either 'intersect' or 'union'. Defaults to 'intersect'.
        - If 'intersect', then the output raster will only contain the pixels of the input rasters that overlap.
        - If 'union', then the output raster will contain every pixel in the outputs. Layers without data will
          have their pixel values set to 0.
    format : str
        The GDAL image format for the output. Defaults to 'GTiff'
    datatype : gdal datatype
        The datatype of the gdal array - see note. Defaults to gdal.GDT_Int32

    """
    # TODO: Confirm the union works, and confirm that nondata defaults to 0.
    log.info("Stacking images {}".format(raster_paths))
    if len(raster_paths) <= 1:
        raise StackImagesException("stack_images requires at least two input images")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    most_rasters = max(raster.RasterCount for raster in rasters)
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1  # Y resolution in affine geotransform is -ve for Maths reasons
    combined_polygons = get_combined_polygon(rasters, geometry_mode)

    # Creating a new gdal object
    out_raster = create_new_image_from_polygon(combined_polygons, out_raster_path, x_res, y_res,
                                               most_rasters, projection, format, datatype)

    # I've done some magic here. GetVirtualMemArray lets you change a raster directly without copying
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    if len(out_raster_array.shape == 2):
        out_raster_array = np.expand_dims(out_raster_array, 3)
    present_layer = 0
    for i, in_raster in enumerate(rasters):
        log.info("Stacking image {}".format(i))
        in_raster_array = in_raster.GetVirtualMemArray()
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, combined_polygons)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_raster, combined_polygons)
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        # Gdal does band, y, x
        out_raster_view = out_raster_array[
                          :
                          out_y_min: out_y_max,
                          out_x_min: out_x_max
                          ]
        in_raster_view = in_raster_array[
                         :
                         in_y_min: in_y_max,
                         in_x_min: in_x_max
                         ]
        out_raster_view[...] = (out_raster_view+in_raster_view)/2   # Sequential mean
        out_raster_view = None
        in_raster_view = None
        present_layer += in_raster.RasterCount
        in_raster_array = None
        in_raster = None
    out_raster_array = None
    out_raster = None


def trim_image(in_raster_path, out_raster_path, polygon, format="GTiff"):
    """
    Trims a raster to a polygon.

    Parameters
    ----------
    in_raster_path : str
        Path to the imput raster
    out_raster_path : str
        Path of the output raster
    polygon : ogr.Geometry
        A ogr.Geometry containing a single polygon
    format : str
        Image format of the output raster. Defaults to 'GTiff'.
    """
    with TemporaryDirectory() as td:
        in_raster = gdal.Open(in_raster_path)
        in_gt = in_raster.GetGeoTransform()
        x_res = in_gt[1]
        y_res = in_gt[5] * -1
        temp_band = in_raster.GetRasterBand(1)
        datatype = temp_band.DataType
        out_raster = create_new_image_from_polygon(polygon, out_raster_path, x_res, y_res,
                                                  in_raster.RasterCount, in_raster.GetProjection(),
                                                  format, datatype)
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, polygon)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_raster, polygon)
        out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
        in_raster_array = in_raster.GetVirtualMemArray()
        out_raster_view = out_raster_array[
                      :,
                      out_y_min: out_y_max,
                      out_x_min: out_x_max
                      ]
        in_raster_view = in_raster_array[
                    :,
                    in_y_min: in_y_max,
                    in_x_min: in_x_max
                    ]
        out_raster_view[...] = in_raster_view
        out_raster_view = None
        in_raster_view = None
        out_raster_array = None
        in_raster_array = None
        out_raster = None
        in_raster = None


def mosaic_images(raster_paths, out_raster_file, format="GTiff", datatype=gdal.GDT_Int32, nodata = 0):
    """
    Mosaics multiple images with the same number of layers into one single image. Overwrites
    overlapping pixels with the value furthest down raster_paths. Takes projection from the first
    raster.

    Parameters
    ----------
    raster_paths : str
        A list of paths of raster to be mosaiced
    out_raster_file : str
        The path to the output file
    format : str
        The image format of the output raster. Defaults to 'GTiff'
    datatype : gdal datatype
        The datatype of the output raster. Defaults to gdal.GDT_Int32
    nodata : number
        The input nodata value; any pixels in raster_paths with this value will be ignored. Defaults to 0.

    """

    # This, again, is very similar to stack_rasters
    log = logging.getLogger(__name__)
    log.info("Beginning mosaic")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1  # Y resolution in agt is -ve for Maths reasons
    combined_polygon = align_bounds_to_whole_number(get_combined_polygon(rasters, geometry_mode='union'))
    layers = rasters[0].RasterCount
    out_raster = create_new_image_from_polygon(combined_polygon, out_raster_file, x_res, y_res, layers,
                                               projection, format, datatype)
    log.info("New empty image created at {}".format(out_raster_file))
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    for i, raster in enumerate(rasters):
        log.info("Now mosaicking raster no. {}".format(i))
        in_raster_array = raster.GetVirtualMemArray()
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        in_bounds = get_raster_bounds(raster)
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_raster, in_bounds)
        out_raster_view = out_raster_array[:, out_y_min: out_y_max, out_x_min: out_x_max]
        np.copyto(out_raster_view, in_raster_array, where=in_raster_array != nodata)
        in_raster_array = None
        out_raster_view = None
    log.info("Raster mosaicking done")
    out_raster_array = None


def composite_images_with_mask(in_raster_path_list, composite_out_path, format="GTiff", generate_date_image=False):
    """
    Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Will also create a mask and
    (optionally) a date image in the same directory.

    Parameters
    ----------
    in_raster_path_list : list of str
        A list of paths to rasters.
    composite_out_path : str
        The path of the output image
    format : str, optional
        The gdal format of the image. Defaults to "GTiff"
    generate_date_image : bool, optional
        If true, generates a single-layer raster containing the dates of each image detected - see below.

    Returns
    -------
    composite_path : str
        The path to the composite.

    Notes
    -----
    Masks are assumed to be a multiplicative .msk file with the same path as their corresponding image; see REFERENCE.
    All images must have the same number of layers and resolution, but do not have to be perfectly on top of each
    other. If it does not exist, composite_out_path will be created. Takes projection, resolution, ect from first band
    of first raster in list. Will reproject images and masks if they do not match initial raster.

    If generate_date_images is True, an raster ending with the suffix .date will be created; each pixel will contain the
    timestamp (yyyymmdd) of the date that pixel was last seen in the composite.

    """

    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(format)
    in_raster_list = [gdal.Open(raster) for raster in in_raster_path_list]
    projection = in_raster_list[0].GetProjection()
    in_gt = in_raster_list[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1
    n_bands = in_raster_list[0].RasterCount
    temp_band = in_raster_list[0].GetRasterBand(1)
    datatype = temp_band.DataType
    temp_band = None

    # Creating output image + array
    log.info("Creating composite at {}".format(composite_out_path))
    log.info("Composite info: x_res: {}, y_res: {}, {} bands, datatype: {}, projection: {}"
             .format(x_res, y_res, n_bands, datatype, projection))
    out_bounds = align_bounds_to_whole_number(get_poly_bounding_rect(get_combined_polygon(in_raster_list,
                                                                                          geometry_mode="union")))
    composite_image = create_new_image_from_polygon(out_bounds, composite_out_path, x_res, y_res, n_bands,
                                                    projection, format, datatype)

    if generate_date_image:
        time_out_path = composite_out_path.rsplit('.')[0]+".dates"
        dates_image = create_matching_dataset(composite_image, time_out_path, bands=1, datatype=gdal.GDT_UInt32)
        dates_array = dates_image.GetVirtualMemArray(eAccess=gdal.gdalconst.GF_Write).squeeze()

    output_array = composite_image.GetVirtualMemArray(eAccess=gdal.gdalconst.GF_Write)
    if len(output_array.shape) == 2:
        output_array = np.expand_dims(output_array, 0)

    mask_paths = []

    for i, in_raster in enumerate(in_raster_list):
        mask_paths.append(get_mask_path(in_raster_path_list[i]))

        # Get a view of in_raster according to output_array
        log.info("Adding {} to composite".format(in_raster_path_list[i]))
        in_bounds = align_bounds_to_whole_number(get_raster_bounds(in_raster))
        x_min, x_max, y_min, y_max = pixel_bounds_from_polygon(composite_image, in_bounds)
        output_view = output_array[:, y_min:y_max, x_min:x_max]

        # Move every unmasked pixel in in_raster to output_view
        log.info("Mask for {} at {}".format(in_raster_path_list[i], mask_paths[i]))
        in_masked = get_masked_array(in_raster, mask_paths[i])
        np.copyto(output_view, in_masked, where=np.logical_not(in_masked.mask))

        # Save dates in date_image if needed
        if generate_date_image:
            dates_view = dates_array[y_min: y_max, x_min: x_max]
            # Gets timestamp as integer in form yyyymmdd
            date = np.uint32(get_sen_2_image_timestamp(in_raster.GetFileList()[0]).split("T")[0])
            dates_view[np.logical_not(in_masked.mask[0, ...])] = date
            dates_view = None

        # Deallocate
        output_view = None
        in_masked = None

    output_array = None
    dates_array = None
    dates_image = None
    composite_image = None

    log.info("Composite done")
    log.info("Creating composite mask at {}".format(composite_out_path.rsplit(".")[0]+".msk"))
    combine_masks(mask_paths, composite_out_path.rsplit(".")[0]+".msk", combination_func='or', geometry_func="union")
    return composite_out_path


def reproject_directory(in_dir, out_dir, new_projection, extension = '.tif'):
    """
    Reprojects every file ending with extension to new_projection and saves in out_dir

    Parameters
    ----------
    in_dir : str
        A directory containing the rasters to be reprojected/
    out_dir : str
        The directory to save the output files to. Output files will be saved in out_dir, with the same
        filenames.
    new_projection : str
        The new projection in wkt.
    extension : str, optional
        The file extension to reproject. Default is '.tif'

    """
    log = logging.getLogger(__name__)
    image_paths = [os.path.join(in_dir, image_path) for image_path in os.listdir(in_dir) if image_path.endswith(extension)]
    for image_path in image_paths:
        reproj_path = os.path.join(out_dir, os.path.basename(image_path))
        log.info("Reprojecting {} to {}, storing in {}".format(image_path, reproj_path, new_projection))
        reproject_image(image_path, reproj_path, new_projection)


def reproject_image(in_raster, out_raster_path, new_projection,  driver = "GTiff",  memory = 2e3, do_post_resample=True):
    """
    Creates a new, reprojected image from in_raster using the gdal.ReprojectImage function.

    Parameters
    ----------
    in_raster : str or gdal.Dataset
        Either a gdal.Dataset object or a path to a raster
    out_raster_path : str
        The path to the new output raster.
    new_projection : str or int
        The new projection in .wkt or as an EPSG number
    driver : str, optional
        The format of the output raster.
    memory : float, optional
        The amount of memory to give to the reprojection. Defaults to 2e3
    do_post_resample : bool, optional
        If set to false, do not resample the image back to the original projection. Defaults to True

    Notes
    -----
    The GDAL reprojection routine changes the size of the pixels by a very small amount; for example, a 10m pixel image
    can become a 10.002m pixel resolution image. To stop alignment issues, by default this function resamples the images
    back to their original resolution. If you are reprojecting from latlon to meters and get an outofmemory error from
    Gdal, set do_post_resample to False.


    """
    if type(new_projection) is int:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(new_projection)
        new_projection = proj.ExportToWkt()
    log = logging.getLogger(__name__)
    log.info("Reprojecting {} to {}".format(in_raster, new_projection))
    if type(in_raster) is str:
        in_raster = gdal.Open(in_raster)
    res = in_raster.GetGeoTransform()[1]
    gdal.Warp(out_raster_path, in_raster, dstSRS=new_projection, warpMemoryLimit=memory, format=driver)
    # After warping, image has irregular gt; resample back to previous pixel size
    # TODO: Make this an option
    if do_post_resample:
        resample_image_in_place(out_raster_path, res)
    return out_raster_path


def composite_directory(image_dir, composite_out_dir, format="GTiff", generate_date_images=False):
    """
    Using composite_images_with_mask, creates a composite containing every image in image_dir. This will
     place a file named composite_[last image date].tif inside composite_out_dir

    Parameters
    ----------
    image_dir : str
        The directory containing the rasters and associated .msk files to be composited.
    composite_out_dir : str
        The directory that will contain the final composite
    format : str, optional
        The raster format of the output image. Defaults to 'GTiff'
    generate_date_images : bool, optional
        If true, generates a corresponding date image for the composite. See docs for composite_images_with_mask.
        Defaults to False.

    Returns
    -------
    composite_out_path : str
        The path to the new composite

    """
    log = logging.getLogger(__name__)
    log.info("Compositing {}".format(image_dir))
    sorted_image_paths = [os.path.join(image_dir, image_name) for image_name
                          in sort_by_timestamp(os.listdir(image_dir), recent_first=False)  # Let's think about this
                          if image_name.endswith(".tif")]
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(composite_out_dir, "composite_{}.tif".format(last_timestamp))
    composite_images_with_mask(sorted_image_paths, composite_out_path, format, generate_date_image=generate_date_images)
    return composite_out_path


def flatten_probability_image(prob_image, out_path):
    """
    Takes a probability output from classify_image and flattens it into a single layer containing only the maximum
    value from each pixel.

    Parameters
    ----------
    prob_image : str
        The path to a probability image.
    out_path : str
        The place to save the flattened image.

    """
    prob_raster = gdal.Open(prob_image)
    out_raster = create_matching_dataset(prob_raster, out_path, bands=1)
    prob_array = prob_raster.GetVirtualMemArray()
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_array[:, :] = prob_array.max(axis=0)
    out_array = None
    prob_array = None
    out_raster = None
    prob_raster = None


def get_masked_array(raster, mask_path):
    """
    Returns a numpy.mask masked array for the raster.
    Masked pixels are FALSE in the mask image (multiplicateive map),
    but TRUE in the masked_array (nodata pixels). If the raster is multi-band and
    the mask is single-band, the mask will be applied to every raster.
    Parameters
    ----------
    raster : gdal.Dataset
        A gdal.Dataset object
    mask_path : str
        The path to the mask to use

    Returns
    -------
    masked_array : numpy.masked
        A numpy.masked array of the raster.

    """
    mask = gdal.Open(mask_path)
    mask_array = mask.GetVirtualMemArray().squeeze()
    raster_array = raster.GetVirtualMemArray()
    # If the shapes do not match, assume single-band mask for multi-band raster
    if len(mask_array.shape) == 2 and len(raster_array.shape) == 3:
        mask_array = project_array(np.asarray(mask_array), raster_array.shape[0], 0)
    return np.ma.array(raster_array, mask=np.logical_not(mask_array))


def stack_and_trim_images(old_image_path, new_image_path, aoi_path, out_image):
    """
    Stacks an old and new S2 image and trims to within an aoi.
    Parameters
    ----------
    old_image_path : str
        Path to the image that will be the first set of bands in the output image
    new_image_path : str
        Path to the image that will be the second set of bands in the output image
    aoi_path : str
        Path to a shapefile containing the AOI
    out_image : str
        Path to the location of the clipped and stacked image.

    """
    log = logging.getLogger(__name__)
    if os.path.exists(out_image):
        log.warning("{} exists, skipping.")
        return
    with TemporaryDirectory() as td:
        old_clipped_image_path = os.path.join(td, "old.tif")
        new_clipped_image_path = os.path.join(td, "new.tif")
        clip_raster(old_image_path, aoi_path, old_clipped_image_path)
        clip_raster(new_image_path, aoi_path, new_clipped_image_path)
        stack_images([old_clipped_image_path, new_clipped_image_path],
                     out_image, geometry_mode="intersect")


def clip_raster(raster_path, aoi_path, out_path, srs_id=4326, flip_x_y = False, dest_nodata = 0):
    """
    Clips a raster at raster_path to a shapefile given by aoi_path. Assumes a shapefile only has one polygon.
    Will np.floor() when converting from geo to pixel units and np.absolute() y resolution form geotransform.
    Will also reproject the shapefile to the same projection as the raster if needed.

    Parameters
    ----------
    raster_path : str
        Path to the raster to be clipped.
    aoi_path : str
        Path to a shapefile containing a single polygon
    out_path : str
        Path to a location to save the final output raster
    flip_x_y : bool, optional
        If True, swaps the x and y axis of the raster image before clipping. For compatability with Landsat.
        Default is False.
    dest_nodata : number, optional
        The fill value for outside of the clipped area. Deafults to 0.

    """
    # TODO: Set values outside clip to 0 or to NaN - in irregular polygons
    # https://gis.stackexchange.com/questions/257257/how-to-use-gdal-warp-cutline-option
    with TemporaryDirectory() as td:
        log.info("Clipping {} with {}".format(raster_path, aoi_path))
        raster = gdal.Open(raster_path)
        in_gt = raster.GetGeoTransform()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(raster.GetProjection())
        intersection_path = os.path.join(td, 'intersection')
        aoi = ogr.Open(aoi_path)
        if aoi.GetLayer(0).GetSpatialRef().ExportToWkt() != srs.ExportToWkt():
            log.info("Non-matching projections, reprojecting.")
            aoi = None
            tmp_aoi_path = os.path.join(td, "tmp_aoi.shp")
            reproject_vector(aoi_path, tmp_aoi_path, srs)
            aoi = ogr.Open(tmp_aoi_path)
        intersection = get_aoi_intersection(raster, aoi)
        min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
        if flip_x_y:
            min_x_geo, min_y_geo = min_y_geo, min_x_geo
            max_x_geo, max_y_geo = max_y_geo, max_x_geo
        width_pix = int(np.floor(max_x_geo - min_x_geo)/in_gt[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo)/np.absolute(in_gt[5]))
        new_geotransform = (min_x_geo, in_gt[1], 0, max_y_geo, 0, in_gt[5])   # OK, time for hacking
        write_geometry(intersection, intersection_path, srs_id=srs.ExportToWkt())
        clip_spec = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=intersection_path+r"/geometry.shp",
            cropToCutline=True,
            width=width_pix,
            height=height_pix,
            dstSRS=srs,
            dstNodata=dest_nodata
        )
        out = gdal.Warp(out_path, raster, options=clip_spec)
        out.SetGeoTransform(new_geotransform)
        out = None


def clip_raster_to_intersection(raster_to_clip_path, extent_raster_path, out_raster_path, is_landsat=False):
    """
    Clips one raster to the extent proivded by the other raster, and saves the result at temp_file.
    Assumes both raster_to_clip and extent_raster are in the same projection.
    Parameters
    ----------
    raster_to_clip_path : str
        The location of the raster to be clipped.
    extent_raster_path : str
        The location of the raster that will provide the extent to clip to
    out_raster_path : str
        A location for the finished raster
    """

    with TemporaryDirectory() as td:
        temp_aoi_path = os.path.join(td, "temp_clip.shp")
        get_extent_as_shp(extent_raster_path, temp_aoi_path)
        ext_ras = gdal.Open(extent_raster_path)
        proj = osr.SpatialReference(wkt=ext_ras.GetProjection())
        srs_id = int(proj.GetAttrValue('AUTHORITY', 1))
        clip_raster(raster_to_clip_path, temp_aoi_path, out_raster_path, srs_id, flip_x_y = is_landsat)


def create_new_image_from_polygon(polygon, out_path, x_res, y_res, bands,
                           projection, format="GTiff", datatype = gdal.GDT_Int32, nodata = -9999):
    """
    Returns an empty image that covers the extent of the imput polygon.

    Parameters
    ----------
    polygon : ogr.Geometry
        An OGR.Geometry object of a single polygon
    out_path : str
        The path to save the new image to
    x_res : number
        Pixel width in the new image
    y_res : number
        Pixel height in the new image
    bands : int
        Number of bands in the new image.
    projection : str
        The projection, in wkt, of the output image.
    format : str, optional
        The gdal raster format of the output image. Defaults to "Gtiff"
    datatype : gdal datatype, optional
        The gdal datatype of the output image. Defaults to gdal.GDT_Int32

    Returns
    -------
    A gdal.Image object

    """
    # TODO: Implement nodata
    bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max = polygon.GetEnvelope()
    if bounds_x_min >= bounds_x_max:
        bounds_x_min, bounds_x_max = bounds_x_max, bounds_x_min
    if bounds_y_min >= bounds_y_max:
        bounds_y_min, bounds_y_max = bounds_y_max, bounds_y_min
    final_width_pixels = int(np.abs(bounds_x_max - bounds_x_min) / x_res)
    final_height_pixels = int(np.abs(bounds_y_max - bounds_y_min) / y_res)
    driver = gdal.GetDriverByName(format)
    out_raster = driver.Create(
        out_path, xsize=final_width_pixels, ysize=final_height_pixels,
        bands=bands, eType=datatype
    )
    out_raster.SetGeoTransform([
        bounds_x_min, x_res, 0,
        bounds_y_max, 0, y_res * -1
    ])
    out_raster.SetProjection(projection)
    return out_raster


def resample_image_in_place(image_path, new_res):
    """
    Resamples an image in-place using gdalwarp to new_res in metres.
    WARNING: This will make a permanent change to an image! Use with care.

    Parameters
    ----------
    image_path : str
        Path to the image to be resampled

    new_res : number
        Pixel edge size in meters

    """
    # I don't like using a second object here, but hey.
    with TemporaryDirectory() as td:
        # Remember this is used for masks, so any averging resample strat will cock things up.
        args = gdal.WarpOptions(
            xRes=new_res,
            yRes=new_res
        )
        temp_image = os.path.join(td, "temp_image.tif")
        gdal.Warp(temp_image, image_path, options=args)

        # Urrrgh. Stupid Windows permissions.
        if sys.platform.startswith("win"):
            os.remove(image_path)
            shutil.copy(temp_image, image_path)
        else:
            shutil.move(temp_image, image_path)


def align_image_in_place(image_path, target_path):
    """
    Adjusts the geotransform of the image at image_path with that of the one at target_path, so they align neatly with
    the smallest magnitude of movement

    Parameters
    ----------
    image_path : str
        The path to the image to be adjusted.
    target_path : str
        The image to align the image at image_path to.

    Raises
    ------
    NonSquarePixelException
        Raised if either image does not have square pixels

    """
    log.info("Aligning {} with{}")
    # Right, this is actually a lot more complicated than I thought
    # Step 1
    target = gdal.Open(target_path)
    target_gt = target.GetGeoTransform()
    if target_gt[1] != np.abs(target_gt[5]):
        raise NonSquarePixelException("Target pixel resolution is not uniform")

    image = gdal.Open(image_path, gdal.GA_Update)
    image_gt = image.GetGeoTransform()
    if image_gt[1] != np.abs(image_gt[5]):
        raise NonSquarePixelException("Image pixel resolution is not uniform")

    # need to find the nearest grid intersection to image tl
    pixel_index_x, pixel_index_y = get_local_top_left(image, target)

    target_res = target_gt[1]

    image_x = image_gt[0]
    image_y = image_gt[3]
    image_res = image_gt[1]

    if target_res%image_res != 0:
        log.warning("Target and image resolutions are not divisible, grids will not align. Consider resampling.")

    # First, do we want to move the image left or right and up or down?
    x_offset = image_x%target_res
    y_offset = image_y%target_res

    if x_offset == 0 and y_offset == 0:
        log.info("Images are already aligned")
        return

    # If x is nearest to pixel line
    if -1*(image_res/2) <= x_offset <= image_res/2:
        new_x = image_x - x_offset
    else:
        new_x = image_x + (image_res - x_offset)
        
    # Likewise with y
    if -1 * (image_res / 2) <= y_offset <= image_res / 2:
        new_y = image_y - y_offset
    else:
        new_y = image_y + (image_res - y_offset)

    new_gt = list(image_gt)
    new_gt[0] = new_x
    new_gt[3] = new_y
    image.SetGeoTransform(new_gt)
    image = None


def raster_to_array(rst_pth):
    """Reads in a raster file and returns a N-dimensional array.

    Parameters
    ----------
    rst_pth : str
        Path to input raster.
    Returns
    -------
    out_array : array_like
        An N-dimensional array.
    """
    log = logging.getLogger(__name__)
    in_ds = gdal.Open(rst_pth)
    out_array = in_ds.ReadAsArray()

    return out_array

def get_extent_as_shp(in_ras_path, out_shp_path):
    """
    Gets the extent of a raster as a shapefile
    Parameters
    ----------
    in_ras_path : str
        The raster to get
    out_shp_path : str
        The shape path
    Returns
    -------
    out_shp_path : str
        The path to the new shapefile

    """
    #By Qing
    os.system('gdaltindex ' + out_shp_path + ' ' + in_ras_path)
    return out_shp_path


def calc_ndvi(raster_path, output_path):
    """
    Creates a raster of NDVI from the input raster at output_path
    Parameters
    ----------
    raster_path : str
        Path to a raster with blue, green, red and infrared bands (in that order)
    output_path : str
        Path to a location to save the output raster

    """
    raster = gdal.Open(raster_path)
    out_raster = create_matching_dataset(raster, output_path, datatype=gdal.GDT_Float32)
    array = raster.GetVirtualMemArray()
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    R = array[2, ...]
    I = array[3, ...]
    out_array[...] = (R-I)/(R+I)

    out_array[...] = np.where(out_array == -2147483648, 0, out_array)

    R = None
    I = None
    array = None
    out_array = None
    raster = None
    out_raster = None


def apply_band_function(in_path, function, bands, out_path, out_datatype = gdal.GDT_Int32):
    """
    Applys an arbitrary band mathemtics function to an image at in_path and saves the result at out_map.
    Function should be a function object of the form f(band_input_A, band_input_B, ...)

    Parameters
    ----------
    in_path : str
        The image to process
    function : Func
    bands
    out_path
    out_datatype

    Examples
    --------
    Calculating the NDVI of an image with red band at band 0 and IR band at 4
    First, define the function to be run across each pixel:

    >>> def ndvi_function(r, i):
    ...     return (r-i)/(r+i)



    >>> apply_band_function("my_raster.tif", ndvi_function, [0,1], "my_ndvi.tif")

    """
    raster = gdal.Open(in_path)
    out_raster = create_matching_dataset(raster, out_path=out_path, datatype=out_datatype)
    array = raster.GetVirtualMemArray()
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    band_views = [array[band, ...] for band in bands]
    out_array[...] = function(*band_views)
    out_array = None
    for view in band_views:
        view = None
    raster = None
    out_raster = None


def ndvi_function(r, i):
    """
    :meta private:
    Parameters
    ----------
    r
    i

    Returns
    -------

    """
    return (r-i)/(r+i)


def apply_image_function(in_paths, out_path, function, out_datatype = gdal.GDT_Int32):
    """
    Applies a pixel-wise function across every image. Assumes each image is exactly contiguous and, for now,
    single-banded. function() should take a list of values and return a single value.

    Parameters
    ----------
    in_paths : list of str
        list of raster paths to process
    out_path : str
        The path to the
    function : function
        The function to apply to the list of images. Must take a list of numbers as an input and return a value.
    out_datatype : gdal datatype, optional
        The datatype of the final raster. Defaults to gdal.gdt_Int32

    Examples
    --------
    Producing a raster where each pixel contains the sum of the corresponding pixels in a list of other rasters

    >>> def sum_function(pixels_in):
    ...     return np.sum(pixels_in)
    >>> in_paths = os.listdir("my_raster_dir")
    >>> apply_image_function(in_paths, "sum_raster.tif", sum_function)

    """
    rasters = [gdal.Open(in_path) for in_path in in_paths]
    raster_arrays = [raster.GetVirtualMemArray() for raster in rasters]
    in_array = np.stack(raster_arrays, axis=0)

    out_raster = create_matching_dataset(rasters[0], out_path=out_path, datatype=out_datatype)
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_array[...] = np.apply_along_axis(function, 0, in_array)

    # Deallocating. Not taking any chances here.
    out_array = None
    out_raster = None
    in_array = None
    for raster_array, raster in zip(raster_arrays, rasters):
        raster_array = None
        raster = None


def sum_function(pixels_in):
    return np.sum(pixels_in)


def raster_sum(inRstList, outFn, outFmt='GTiff'):
    """Creates a raster stack from a list of rasters. Adapted from Chris Gerard's
    book 'Geoprocessing with Python'. The out put data type is the same as the input data type.

    Parameters
    ----------

    inRstList : list of str
        List of rasters to stack.
    outFn : str
        Filename output as str including directory else image will be
        written to current working directory.
    outFmt : str, optional.
        String specifying the input data format e.g. 'GTiff' or 'VRT'. Defaults to GTiff.


    """
    log = logging.getLogger(__name__)
    log.info('Starting raster sum function.')

    # open 1st band to get info
    in_ds = gdal.Open(inRstList[0])
    in_band = in_ds.GetRasterBand(1)

    # Get raster shape
    rst_dim = (in_band.YSize, in_band.XSize)

    # initiate empty array
    empty_arr = np.empty(rst_dim, dtype=np.uint8)

    for i, rst in enumerate(inRstList):
        # Todo: Check that dimensions and shape of both arrays are the same in the first loop.
        ds = gdal.Open(rst)
        bnd = ds.GetRasterBand(1)
        arr = bnd.ReadAsArray()
        empty_arr = empty_arr + arr

    # Create a 1 band GeoTiff with the same properties as the input raster
    driver = gdal.GetDriverByName(outFmt)
    out_ds = driver.Create(outFn, in_band.XSize, in_band.YSize, 1,
                           in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    out_ds.GetRasterBand(1).WriteArray(empty_arr)

    # write the data to disk
    out_ds.FlushCache()

    # Compute statistics on each output band setting ComputeStatistics to false calculates stats on all pixels
    # not estimates
    out_ds.GetRasterBand(1).ComputeStatistics(False)

    out_ds.BuildOverviews("average", [2, 4, 8, 16, 32])

    out_ds = None

    log.info('Finished summing up of raster layers.')


def filter_by_class_map(image_path, class_map_path, out_map_path, classes_of_interest, out_resolution=10):
    """
    Filters a raster with a set of classes for corresponding for pixels in filter_map_path containing only
    classes_of_interest. Assumes that filter_map_path and class_map_path are same resolution and projection.


    Parameters
    ----------
    image_path : str
        Path to the raster to be filtered.
    class_map_path : str
        Path to the map to use as the filter. Assumes a raster of integer class labels.
    out_map_path : str
        Path to the filtered map
    classes_of_interest : list of int
        The classes in class_map_path to keep present in the raster to be filtered
    out_resolution : number, optional
        The resolution of the output image

    Returns
    -------
    out_map_path : str
        The path to the new map

    """
    # TODO: Include nodata value
    log = logging.getLogger(__name__)
    log.info("Filtering {} using classes{} from map {}".format(class_map_path, classes_of_interest, image_path))
    with TemporaryDirectory() as td:

        binary_mask_path = os.path.join(td, "binary_mask.tif")
        create_mask_from_class_map(class_map_path, binary_mask_path, classes_of_interest, out_resolution=out_resolution)

        log.info("Mask created at {}, applying...".format(binary_mask_path))
        class_map = gdal.Open(binary_mask_path)
        class_array = class_map.GetVirtualMemArray()

        image_map = gdal.Open(image_path)
        image_array = image_map.GetVirtualMemArray()
        out_map = create_matching_dataset(image_map, out_map_path)
        out_array = out_map.GetVirtualMemArray(eAccess=gdal.GA_Update)
        class_bounds = get_raster_bounds(class_map)
        image_bounds = get_raster_bounds(image_map)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(image_map, class_bounds)
        image_view = image_array[in_y_min: in_y_max, in_x_min: in_x_max]
        class_x_min, class_x_max, class_y_min, class_y_max = pixel_bounds_from_polygon(class_map, image_bounds)
        class_view = class_array[class_y_min: class_y_max, class_x_min: class_x_max]
        filtered_array = apply_array_image_mask(class_view, image_view)

        np.copyto(out_array, filtered_array)
        out_array = None
        out_map = None
        class_array = None
        class_map = None

    log.info("Map filtered")
    return out_map_path


def open_dataset_from_safe(safe_file_path, band, resolution = "10m"):
    """
    Opens a dataset given a level 2 .SAFE file.

    Parameters
    ----------
    safe_file_path : str
        The path to the .SAFE file
    band : int
        The band to open
    resolution : {'10m', '20m', '60m'}, optional
        The resolution of imagery to open. Defaults to "10m".

    Returns
    -------
    band_raster : gdal.Dataset
        A Gdal dataset contining the band

    """
    image_glob = r"GRANULE/*/IMG_DATA/R{}/*_{}_{}.jp2".format(resolution, band, resolution)
    # edited by hb91
    #image_glob = r"GRANULE/*/IMG_DATA/*_{}.jp2".format(band)
    fp_glob = os.path.join(safe_file_path, image_glob)
    image_file_path = glob.glob(fp_glob)
    out = gdal.Open(image_file_path[0])
    return out


def preprocess_sen2_images(l2_dir, out_dir, l1_dir, cloud_threshold=60, buffer_size=0, epsg=None,
                           bands=("B02", "B03", "B04", "B08"), out_resolution=10):
    """
    For every .SAFE folder in l2_dir and L1_dir, stacks band 2,3,4 and 8  bands into a single geotif, creates a cloudmask from
    the combined fmask and sen2cor cloudmasks and reprojects to a given EPSG if provided.

    Parameters
    ----------
    l2_dir : str
        The directory containing a set of L2 .SAFE folders to preprocess
    out_dir : str
        The directory to store the preprocessed files
    l1_dir : str
        The directory containing a set of L1 .SAFE files, corresponding to the L2 files in l2_dir
    cloud_threshold : number
        DEPRECIATED; in for backwards compatibility.
    buffer_size : int, optional
        The buffer to apply to the sen2cor mask - defaults to 0
    epsg : int, optional
        If present, the EPSG number to reproject the final images to.
    bands : list of str, optional
        List of names of bands to include in the final rasters. Defaults to ("B02", "B03", "B04", "B08")
    out_resolution : number, optional
        Resolution to resample every image to - units are defined by the image projection. Default is 10.

    Warnings
    --------
    This functions' augment list is likely to be changed in the near future to (l1_dir, l2_dir, out_dir) - please
    be aware - September 2020.


    """
    safe_file_path_list = [os.path.join(l2_dir, safe_file_path) for safe_file_path in os.listdir(l2_dir)]
    for l2_safe_file in safe_file_path_list:
        with TemporaryDirectory() as temp_dir:
            log.info("----------------------------------------------------")
            log.info("Merging 10m bands in SAFE dir: {}".format(l2_safe_file))
            temp_path = os.path.join(temp_dir, get_sen_2_granule_id(l2_safe_file)) + ".tif"
            log.info("Output file: {}".format(temp_path))
            stack_sentinel_2_bands(l2_safe_file, temp_path, bands=bands, out_resolution=out_resolution)

            log.info("Creating cloudmask for {}".format(temp_path))
            l1_safe_file = get_l1_safe_file(l2_safe_file, l1_dir)
            mask_path = get_mask_path(temp_path)
            create_mask_from_sen2cor_and_fmask(l1_safe_file, l2_safe_file, mask_path, buffer_size=buffer_size)
            log.info("Cloudmask created")

            out_path = os.path.join(out_dir, os.path.basename(temp_path))
            out_mask_path = os.path.join(out_dir, os.path.basename(mask_path))

            if epsg:
                log.info("Reprojecting images to {}".format(epsg))
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(epsg)
                wkt = proj.ExportToWkt()
                reproject_image(temp_path, out_path, wkt)
                reproject_image(mask_path, out_mask_path, wkt)
                resample_image_in_place(out_mask_path, out_resolution)
            else:
                log.info("Moving images to {}".format(out_dir))
                shutil.move(temp_path, out_path)
                shutil.move(mask_path, out_mask_path)
                resample_image_in_place(out_mask_path, out_resolution)


def preprocess_landsat_images(image_dir, out_image_path, new_projection = None, bands_to_stack=("B2","B3","B4")):
    """


    Stacks a set of Landsat images into a single raster and reorders the bands into
    [bands, y, x] - by default, Landsat uses [x,y] and bands are in seperate rasters.
    If given, will also reproject to an EPSG or .wkt

    Parameters
    ----------
    image_dir : str
        The directory containing the Landsat images
    out_image_path : str
        The path to the stacked image
    new_projection : int or str, optional
        An EPSG number or a .wkt containing a projection. Defaults to None
    bands_to_stack : list of str, optional
        The Landsat bands to put into the stacked

    """
    log.info("Stacking Landsat rasters in folder {}".format(image_dir))
    band_path_list = []     # This still feels like a Python antipattern, but hey.
    for band_id in bands_to_stack:
        band_glob = os.path.join(image_dir, "LC08_*_{}.TIF".format(band_id))
        band_path_list.append(glob.glob(band_glob)[0])

    n_bands = len(band_path_list)
    driver = gdal.GetDriverByName("GTiff")
    first_ls_raster = gdal.Open(band_path_list[0])
    first_ls_array = first_ls_raster.GetVirtualMemArray()
    out_image = driver.Create(out_image_path,
            xsize = first_ls_array.shape[1],
            ysize = first_ls_array.shape[0],
            bands = n_bands,
            eType = first_ls_raster.GetRasterBand(1).DataType
            )
    out_image.SetGeoTransform(first_ls_raster.GetGeoTransform())
    out_image.SetProjection(first_ls_raster.GetProjection())
    out_array = out_image.GetVirtualMemArray(eAccess = gdal.GA_Update)
    first_ls_array = None
    first_ls_raster = None
    for ii, ls_raster_path in enumerate(band_path_list):
        log.info("Stacking {} to raster layer {}".format(ls_raster_path, ii))
        ls_raster = gdal.Open(ls_raster_path)
        ls_array = ls_raster.GetVirtualMemArray()
        out_array[ii, ...] = ls_array[...]
        ls_array = None
        ls_raster = None
    out_array = None
    out_image = None
    if new_projection:
        with TemporaryDirectory() as td:
            log.info("Reprojecting to {}")
            temp_path = os.path.join(td, "reproj_temp.tif")
            log.info("Temporary image path at {}".format(temp_path))
            reproject_image(out_image_path, temp_path, new_projection, do_post_resample = False)
            os.remove(out_image_path)
            os.rename(temp_path, out_image_path)
            resample_image_in_place(out_image_path, 30)
    log.info("Stacked image at {}".format(out_image_path))


def stack_sentinel_2_bands(safe_dir, out_image_path, bands=("B02", "B03", "B04", "B08"), out_resolution=10):
    """
    Stacks the specified bands of a .SAFE granule directory into a single geotiff

    Parameters
    ----------
    safe_dir : str
        Path to the .SAFE file to stack
    out_image_path : str
        Location of the new image
    bands : list of str, optional
        The band IDs to be stacked
    out_resolution
        The final resolution of the geotif- bands will be resampled if needed.

    Returns
    -------
    out_image_path : str
        The path to the new image

    """

    band_paths = [get_sen_2_band_path(safe_dir, band, out_resolution) for band in bands]

    # Move every image NOT in the requested resolution to resample_dir and resample
    with TemporaryDirectory() as resample_dir:
        new_band_paths = []
        for band_path in band_paths:
            if get_image_resolution(band_path) != out_resolution:
                log.info("Resampling {} to {}m".format(band_path, out_resolution))
                resample_path = os.path.join(resample_dir, os.path.basename(band_path))
                shutil.copy(band_path, resample_path)
                resample_image_in_place(resample_path, out_resolution)  # why did I make this the only in-place function?
                new_band_paths.append(resample_path)
            else:
                new_band_paths.append(band_path)

        stack_images(new_band_paths, out_image_path, geometry_mode="intersect")

    # Saving band labels in images
    new_raster = gdal.Open(out_image_path)
    for band_index, band_label in enumerate(bands):
        band = new_raster.GetRasterBand(band_index+1)
        band.SetDescription(band_label)

    return out_image_path


def get_sen_2_band_path(safe_dir, band, resolution=None):
    """
    Returns the path to the raster of the specified band in the specified safe_dir.

    Parameters
    ----------
    safe_dir : str
        Path to the directory containing the raster
    band : str
        The band identifier ('B01', 'B02', ect)
    resolution : int, optional
        If given, tries to get that band in that image - if not, tries for the highest resolution

    Returns
    -------
    band_path : str
        The path to the raster containing the band.

    """
    if resolution == 10:
        res_string = "10m"
    elif resolution == 20:
        res_string = "20m"
    elif resolution == 60:
        res_string = "60m"
    else:
        res_string = None

    if get_safe_product_type(safe_dir) == "MSIL1C":
        band_glob = "GRANULE/*/IMG_DATA/*_{}*.*".format(band)
        band_glob = os.path.join(safe_dir, band_glob)
        band_paths = glob.glob(band_glob)
        if not band_paths:
            raise FileNotFoundError("Band {} not found for safe file {}".format(band, safe_dir))
        band_path = band_paths[0]

    else:
        if res_string in ["10m", "20m", "60m"]:  # If resolution is given, then find the band of that resolution
            band_glob = "GRANULE/*/IMG_DATA/R{}/*_{}_*.*".format(res_string, band)
            band_glob = os.path.join(safe_dir, band_glob)
            try:
                band_path = glob.glob(band_glob)[0]
            except IndexError:
                log.warning("Band {} not found of specified resolution, searching in other available resolutions".format(band))

        if res_string is None or 'band_path' not in locals():  # Else use the highest resolution available for that band
            band_glob = "GRANULE/*/IMG_DATA/R*/*_{}_*.*".format(band)
            band_glob = os.path.join(safe_dir, band_glob)
            band_paths = glob.glob(band_glob)
            try:
                band_path = sorted(band_paths)[0] # Sorting alphabetically gives the highest resolution first
            except TypeError:
                raise FileNotFoundError("Band {} not found for safe file {}". format(band, safe_dir))
    return band_path


def get_image_resolution(image_path):
    """
    Returns the resolution of the image in its native projection. Assumes square pixels.

    Parameters
    ----------
    image_path : str
        Path to a raster

    Returns
    -------
    resolution : number
        The size of each pixel in the image, in the units of its native projection.

    """
    image= gdal.Open(image_path)
    if image is None:
        raise FileNotFoundError("Image not found at {}".format(image_path))
    gt = image.GetGeoTransform()
    if gt[1] != gt[5]*-1:
        raise NonSquarePixelException("Image at {} has non-square pixels - this is currently not implemented in Pyeo")
    return gt[1]



def stack_old_and_new_images(old_image_path, new_image_path, out_dir, create_combined_mask=True):
    """
    Stacks two images that cover the same tile into a single multi-band raster, old_image_path being the first set of
    bands and new_image_path being the second. The produced image will have the name `{tile}_{old_date}_{new_date}.tif`.

    Parameters
    ----------
    old_image_path : str
        Path to the older image
    new_image_path : str
        Path to the newer image
    out_dir : str
        Directory to place the new stacked raster into
    create_combined_mask : bool, optional
        If True, finds and combines the associated mask files between

    Returns
    -------
    out_image_path : str
        The path to the new image

    """
    # First, decompose the granule ID into its components:
    # e.g. S2A, MSIL2A, 20180301, T162211, N0206, R040, T15PXT, 20180301, T194348
    # are the mission ID(S2A/S2B), product level(L2A), datatake sensing start date (YYYYMMDD) and time(THHMMSS),
    # the Processing Baseline number (N0206), Relative Orbit number (RO4O), Tile Number field (T15PXT),
    # followed by processing run date and then time
    log = logging.getLogger(__name__)
    tile_old = get_sen_2_image_tile(old_image_path)
    tile_new = get_sen_2_image_tile(new_image_path)
    if (tile_old == tile_new):
        log.info("Stacking {} and".format(old_image_path))
        log.info("         {}".format(new_image_path))
        old_timestamp = get_sen_2_image_timestamp(os.path.basename(old_image_path))
        new_timestamp = get_sen_2_image_timestamp(os.path.basename(new_image_path))
        out_path = os.path.join(out_dir, tile_new + '_' + old_timestamp + '_' + new_timestamp)
        log.info("Output stacked file: {}".format(out_path + ".tif"))
        stack_images([old_image_path, new_image_path], out_path + ".tif")
        if create_combined_mask:
            out_mask_path = out_path + ".msk"
            old_mask_path = get_mask_path(old_image_path)
            new_mask_path = get_mask_path(new_image_path)
            combine_masks([old_mask_path, new_mask_path], out_mask_path, combination_func="and", geometry_func="intersect")
        return out_path + ".tif"
    else:
        log.error("Tiles  of the two images do not match. Aborted.")


def apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=False):
    """
    Applies sen2cor to the SAFE file at image_path. Returns the path to the new product.

    Parameters
    ----------
    image_path : str
        Path to the L1 Sentinel 2 .SAFE file
    sen2cor_path : str
        Path to the l2a_process script (Linux) or l2a_process.exe (Windows)
    delete_unprocessed_image : bool, optional
        If True, delete the unprocessed image after processing is done. Defaults to False.

    Returns
    -------
    out_path : str
        The path to the new L2 .SAFE file

    """
    # Here be OS magic. Since sen2cor runs in its own process, Python has to spin around and wait
    # for it; since it's doing that, it may as well be logging the output from sen2cor. This
    # approach can be multithreaded in future to process multiple image (1 per core) but that
    # will take some work to make sure they all finish before the program moves on.
    # added sen2cor_path by hb91
    out_dir = os.path.dirname(image_path)
    log.info("calling subprocess: {}".format([sen2cor_path, image_path, '--output_dir', os.path.dirname(image_path)]))
    now_time = datetime.datetime.now()   # I can't think of a better way of geting the new outpath from sen2cor
    timestamp = now_time.strftime(r"%Y%m%dT%H%M%S")
    sen2cor_proc = subprocess.Popen([sen2cor_path, image_path, '--output_dir', os.path.dirname(image_path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
    while True:
        nextline = sen2cor_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and sen2cor_proc.poll() is not None:
            break
        if "CRITICAL" in nextline:
            #log.error(nextline)
            raise subprocess.CalledProcessError(-1, "L2A_Process")

    log.info("sen2cor processing finished for {}".format(image_path))
    log.info("Validating:")
    version = get_sen2cor_version(sen2cor_path)
    out_path = build_sen2cor_output_path(image_path, timestamp, version)
    if not check_for_invalid_l2_data(out_path):
        log.error("10m imagery not present in {}".format(out_path))
        raise BadS2Exception
    if delete_unprocessed_image:
        log.info("removing {}".format(image_path))
        shutil.rmtree(image_path)
    return out_path


def build_sen2cor_output_path(image_path, timestamp, version):
    """
    Creates a sen2cor output path dependent on the version ofr sen2cor
    Parameters
    ----------
    image_path : str
        Path to the image
    timestamp : str
        Timestamp that processing was started (required for v >= 2.8)
    version : str
        String of the version of sen2cor (ex "2.05.05")

    Returns
    -------
    new_path : str
        The path of the finished sen2cor file

    """
    # Accounting for sen2cors ever-shifting filename format
    if version >= "2.08.00":
        out_path = image_path.replace("MSIL1C", "MSIL2A")
        baseline = get_sen_2_baseline(image_path)
        out_path = out_path.replace(baseline, "N9999")
        out_path = out_path.rpartition("_")[0] + "_" + timestamp + ".SAFE"
    else:
        out_path = image_path.replace("MSIL1C", "MSIL2A")
    return out_path


def get_sen2cor_version(sen2cor_path):
    """
    Gets the version number of sen2cor from the help string.
    Parameters
    ----------
    sen2cor_path : str
        Path the the sen2cor executable

    Returns
    -------
    version : str
        A string of the version of sen2cor at sen2cor_path

    """
    proc = subprocess.run([sen2cor_path, "--help"], stdout=subprocess.PIPE)
    help_string = proc.stdout.decode("utf-8")

    # Looks for the string "Version: " followed by three sets of digits separated by period characters.
    # Returns the three character string as group 1.
    version_regex = r"Version: (\d+.\d+.\d+)"
    match = re.search(version_regex, help_string)
    if match:
        return match.group(1)
    else:
        raise FileNotFoundError("Version information not found; please check your sen2cor path.")


def atmospheric_correction(in_directory, out_directory, sen2cor_path, delete_unprocessed_image=False):
    """
    Applies Sen2cor atmospheric correction to each L1 image in in_directory

    Parameters
    ----------
    in_directory : str
        Path to the directory containing the L1 images
    out_directory : str
        Path to the directory that will containg the new L2 images
    sen2cor_path : str
        Path to the l2a_process script (Linux) or l2a_process.exe (Windows)
    delete_unprocessed_image : bool, optional
        If True, delete the unprocessed image after processing is done. Defaults to False.

    """
    log = logging.getLogger(__name__)
    images = [image for image in os.listdir(in_directory)
              if image.startswith('MSIL1C', 4)]
    # Opportunity for multithreading here
    for image in images:
        log.info("Atmospheric correction of {}".format(image))
        image_path = os.path.join(in_directory, image)
        image_timestamp = datetime.datetime.now().strftime(r"%Y%m%dT%H%M%S")
        out_name = build_sen2cor_output_path(image, image_timestamp, get_sen2cor_version(sen2cor_path))
        out_path = os.path.join(out_directory, out_name)
        out_glob = out_path.rpartition("_")[0] + "*"
        if glob.glob(out_glob):
            log.warning("{} exists. Skipping.".format(out_path))
            continue
        try:
            l2_path = apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=delete_unprocessed_image)
        except (subprocess.CalledProcessError, BadS2Exception):
            log.error("Atmospheric correction failed for {}. Moving on to next image.".format(image))
            pass
        else:
            l2_name = os.path.basename(l2_path)
            log.info("L2  path: {}".format(l2_path))
            log.info("New path: {}".format(os.path.join(out_directory, l2_name)))
            os.rename(l2_path, os.path.join(out_directory, l2_name))


def create_mask_from_model(image_path, model_path, model_clear=0, num_chunks=10, buffer_size=0):
    """
    Returns a multiplicative mask (0 for cloud, shadow or haze, 1 for clear) built from the ML at model_path.

    Parameters
    ----------
    image_path : str
        Path to the image
    model_path : str
        Path to a pickled scikit-learn classification model
    model_clear : int, optional
        The class from the model corresponding to non-cloudy pixels. Defaults to 0
    num_chunks : int, optional
        The number of chunks to break the processing into. See :py:func:`classification.classify_image`
    buffer_size : int, optional
        If present, will apply a buffer of this many pixels to the mask, expanding

    Returns
    -------
    mask_path : str
        The path to the new mask

    """
    from pyeo.classification import classify_image  # Deferred import to deal with circular reference
    with TemporaryDirectory() as td:
        log = logging.getLogger(__name__)
        log.info("Building cloud mask for {} with model {}".format(image_path, model_path))
        temp_mask_path = os.path.join(td, "cat_mask.tif")
        classify_image(image_path, model_path, temp_mask_path, num_chunks=num_chunks)
        temp_mask = gdal.Open(temp_mask_path, gdal.GA_Update)
        temp_mask_array = temp_mask.GetVirtualMemArray()
        mask_path = get_mask_path(image_path)
        mask = create_matching_dataset(temp_mask, mask_path, datatype=gdal.GDT_Byte)
        mask_array = mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
        mask_array[:, :] = np.where(temp_mask_array != model_clear, 0, 1)
        temp_mask_array = None
        mask_array = None
        temp_mask = None
        mask = None
        if buffer_size:
            buffer_mask_in_place(mask_path, buffer_size)
        log.info("Cloud mask for {} saved in {}".format(image_path, mask_path))
        return mask_path


def create_mask_from_confidence_layer(l2_safe_path, out_path, cloud_conf_threshold=0, buffer_size=3):
    """
    Creates a multiplicative binary mask where cloudy pixels are 0 and non-cloudy pixels are 1. If
    cloud_conf_threshold = 0, use scl mask else use confidence image

    Parameters
    ----------
    l2_safe_path : str
        Path to the L1
    out_path : str
        Path to the new path
    cloud_conf_threshold : int, optional
        If 0, uses the sen2cor classification raster as the base for the mask - else uses the cloud confidence image.
        Defaults to 0
    buffer_size : int, optional
        The size of the buffer to apply to the cloudy pixel classes

    Returns
    -------
    out_path : str
        The path to the mask

    """
    log = logging.getLogger(__name__)
    log.info("Creating mask for {} with {} confidence threshold".format(l2_safe_path, cloud_conf_threshold))
    if cloud_conf_threshold:
        cloud_glob = "GRANULE/*/QI_DATA/*CLD*_20m.jp2"  # This should match both old and new mask formats
        cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
        cloud_image = gdal.Open(cloud_path)
        cloud_confidence_array = cloud_image.GetVirtualMemArray()
        mask_array = (cloud_confidence_array < cloud_conf_threshold)
        cloud_confidence_array = None
    else:
        cloud_glob = "GRANULE/*/IMG_DATA/R20m/*SCL*_20m.jp2"  # This should match both old and new mask formats
        cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
        cloud_image = gdal.Open(cloud_path)
        scl_array = cloud_image.GetVirtualMemArray()
        mask_array = np.isin(scl_array, (4, 5, 6))

    mask_image = create_matching_dataset(cloud_image, out_path)
    mask_image_array = mask_image.GetVirtualMemArray(eAccess=gdal.GF_Write)
    np.copyto(mask_image_array, mask_array)
    mask_image_array = None
    cloud_image = None
    mask_image = None
    resample_image_in_place(out_path, 10)
    if buffer_size:
        buffer_mask_in_place(out_path, buffer_size)
    log.info("Mask created at {}".format(out_path))
    return out_path


def create_mask_from_class_map(class_map_path, out_path, classes_of_interest, buffer_size=0, out_resolution=None):
    """
    Creates a multiplicative mask from a classification mask: 1 for each pixel containing one of classes_of_interest,
    otherwise 0

    Parameters
    ----------
    class_map_path : str
        Path to the classification map to build the mask from
    out_path : str
        Path to the new mask
    classes_of_interest : list of int
        The list of classes to count as clear pixels
    buffer_size : int
        If greater than 0, applies a buffer to the masked pixels of this size. Defaults to 0.
    out_resolution : int or None, optional
        If present, resamples the mask to this resoltion. Applied before buffering. Defaults to 0.

    Returns
    -------
    out_path : str
        The path to the new mask.

    """
    # TODO: pull this out of the above function
    class_image = gdal.Open(class_map_path)
    class_array = class_image.GetVirtualMemArray()
    mask_array = np.isin(class_array, classes_of_interest)
    out_mask = create_matching_dataset(class_image, out_path, datatype=gdal.GDT_Byte)
    out_array = out_mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    np.copyto(out_array, mask_array)
    class_array = None
    class_image = None
    out_array = None
    out_mask = None
    if out_resolution:
        resample_image_in_place(out_path, out_resolution)
    if buffer_size:
        buffer_mask_in_place(out_path, buffer_size)
    return out_path


def combine_masks(mask_paths, out_path, combination_func = 'and', geometry_func ="intersect"):
    """
    ORs or ANDs several masks. Gets metadata from top mask. Assumes that masks are a
    Python true or false. Also assumes that all masks are the same projection for now.

    Parameters
    ----------
    mask_paths : list of str
        A list of paths to the masks to combine
    out_path : str
        The path to the new mask
    combination_func : {'and' or 'or}, optional
        Whether the a pixel in the final mask will be masked if
        - any pixel ('or') is masked
        - or all pixels ('and') are masked
        ..in the corresponding pixels in the list of masks. Defaults to 'and'

    geometry_func : {'intersect' or 'union'}
        How to handle non-overlapping masks. Defaults to 'intersect'

    Returns
    -------
    out_path : str
        The path to the new mask

    """
    log = logging.getLogger(__name__)
    log.info("Combining masks {}:\n   combination function: '{}'\n   geometry function:'{}'".format(
        mask_paths, combination_func, geometry_func))
    masks = [gdal.Open(mask_path) for mask_path in mask_paths]
    combined_polygon = align_bounds_to_whole_number(get_combined_polygon(masks, geometry_func))
    gt = masks[0].GetGeoTransform()
    x_res = gt[1]
    y_res = gt[5]*-1  # Y res is -ve in geotransform
    bands = 1
    projection = masks[0].GetProjection()
    out_mask = create_new_image_from_polygon(combined_polygon, out_path, x_res, y_res,
                                             bands, projection, datatype=gdal.GDT_Byte, nodata=0)

    # This bit here is similar to stack_raster, but different enough to not be worth spinning into a combination_func
    # I might reconsider this later, but I think it'll overcomplicate things.
    out_mask_array = out_mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
    out_mask_array = out_mask_array.squeeze() # This here to account for unaccountable extra dimension Windows patch adds
    out_mask_array[:, :] = 1
    for mask_index, in_mask in enumerate(masks):
        in_mask_array = in_mask.GetVirtualMemArray()
        in_mask_array = in_mask_array.squeeze()  # See previous comment
        if geometry_func == "intersect":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_mask, combined_polygon)
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_mask, combined_polygon)
        elif geometry_func == "union":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(out_mask, get_raster_bounds(in_mask))
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(in_mask, get_raster_bounds(in_mask))
        else:
            raise Exception("Invalid geometry_func; can be 'intersect' or 'union'")
        out_mask_view = out_mask_array[out_y_min: out_y_max, out_x_min: out_x_max]
        in_mask_view = in_mask_array[in_y_min: in_y_max, in_x_min: in_x_max]
        if mask_index is 0:
            out_mask_view[:,:] = in_mask_view
        else:
            if combination_func is 'or':
                out_mask_view[:, :] = np.bitwise_or(out_mask_view, in_mask_view, dtype=np.uint8)
            elif combination_func is 'and':
                out_mask_view[:, :] = np.bitwise_and(out_mask_view, in_mask_view, dtype=np.uint8)
            elif combination_func is 'nor':
                out_mask_view[:, :] = np.bitwise_not(np.bitwise_or(out_mask_view, in_mask_view, dtype=np.uint8), dtype=np.uint8)
            else:
                raise Exception("Invalid combination_func; valid values are 'or', 'and', and 'nor'")
        in_mask_view = None
        out_mask_view = None
        in_mask_array = None
        in_mask = None
    out_mask_array = None
    out_mask = None
    return out_path


def buffer_mask_in_place(mask_path, buffer_size):
    """
    Expands a mask in-place, overwriting the previous mask
    Parameters
    ----------
    mask_path : str
        Path to a multiplicative mask (0; masked, 1; unmasked)
    buffer_size : int
        The radius of the buffer, in pixel units of the mask

    """
    log = logging.getLogger(__name__)
    log.info("Buffering {} with buffer size {}".format(mask_path, buffer_size))
    mask = gdal.Open(mask_path, gdal.GA_Update)
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    cache = morph.binary_erosion(mask_array.squeeze(), selem=morph.disk(buffer_size))
    np.copyto(mask_array, cache)
    mask_array = None
    mask = None


def apply_array_image_mask(array, mask, fill_value=0):
    """
    Applies a mask of (y,x) to an image array of (bands, y, x). Replaces any masked pixels with fill_value
    Mask is an a 2 dimensional array of 1 (unmasked) and 0 (masked)

    Parameters
    ----------
    array : array_like
        The array containing the raster data
    mask : array_like
        The array containing the mask
    fill_value : number, optional
        The value to replace any masked pixels with. Defaults to 0.

    Returns
    -------
    masked_array : array_like
        The array with masked pixels replaced with fill_value

    """
    stacked_mask = np.broadcast_to(mask, array.shape)
    return np.where(stacked_mask == 1, array, fill_value)


def create_mask_from_sen2cor_and_fmask(l1_safe_file, l2_safe_file, out_mask_path, buffer_size=0):
    """
    Creates a cloud mask from a combination of the sen2cor thematic mask and the fmask method. Requires corresponding
    level 1 and level 2 .SAFE files.

    Parameters
    ----------
    l1_safe_file : str
        Path to the level 1 .SAFE file
    l2_safe_file : str
        Path to the level 2 .SAFE file
    out_mask_path : str
        Path to the new mask
    buffer_size : int, optional
        If greater than 0, the buffer to apply to the Sentinel 2 thematic map

    """
    with TemporaryDirectory() as td:
        s2c_mask_path = os.path.join(td, "s2_mask.tif")
        fmask_mask_path = os.path.join(td, "fmask.tif")
        create_mask_from_confidence_layer(l2_safe_file, s2c_mask_path, buffer_size=buffer_size)
        create_mask_from_fmask(l1_safe_file, fmask_mask_path)
        combine_masks([s2c_mask_path, fmask_mask_path], out_mask_path, combination_func="and", geometry_func="union")


def create_mask_from_fmask(in_l1_dir, out_path):
    """
    Creates a cloud mask from a level 1 Sentinel-2 product using fmask
    Parameters
    ----------
    in_l1_dir : str
        The path to the l1 .SAFE folder
    out_path : str
        The path to new mask

    """
    log = logging.getLogger(__name__)
    log.info("Creating fmask for {}".format(in_l1_dir))
    with TemporaryDirectory() as td:
        temp_fmask_path = os.path.join(td, "fmask.tif")
        apply_fmask(in_l1_dir, temp_fmask_path)
        fmask_image = gdal.Open(temp_fmask_path)
        fmask_array = fmask_image.GetVirtualMemArray()
        out_image = create_matching_dataset(fmask_image, out_path, datatype=gdal.GDT_Byte)
        out_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
        log.info("fmask created, converting to binary cloud/shadow mask")
        out_array[:,:] = np.isin(fmask_array, (2, 3, 4), invert=True)
        out_array = None
        out_image = None
        fmask_array = None
        fmask_image = None
        resample_image_in_place(out_path, 10)


def apply_fmask(in_safe_dir, out_file, fmask_command="fmask_sentinel2Stacked.py"):
    """
    :meta private:
    Calls fmask to create a new mask for L1 data
    Parameters
    ----------
    in_safe_dir
    out_file
    fmask_command

    Returns
    -------

    """
    # For reasons known only to the spirits, calling subprocess.run from within this function on a HPC cause the PATH
    # to be prepended with a Windows "eoenv\Library\bin;" that breaks the environment. What follows is a large kludge.
    if "torque" in os.getenv("PATH"):  # Are we on a HPC? If so, give explicit path to fmask
        fmask_command = "/data/clcr/shared/miniconda3/envs/eoenv/bin/fmask_sentinel2Stacked.py"
    if sys.platform.startswith("win"):
        fmask_command = subprocess.check_output(["where", fmask_command], text=True).strip()
    log = logging.getLogger(__name__)
    args = [
        fmask_command,
        "-o", out_file,
        "--safedir", in_safe_dir
    ]
    log.info("Creating fmask from {}, output at {}".format(in_safe_dir, out_file))
    fmask_proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    while True:
        nextline = fmask_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and fmask_proc.poll() is not None:
            break
