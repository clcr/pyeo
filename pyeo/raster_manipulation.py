"""
Functions for working with raster data, including masks and platform-specific processing functions.

Key functions
-------------

:py:func:`create_matching_dataset`  : Creates an empty raster of the same shape as a source, ready for writing.

:py:func:`stack_images`  : Stacks a list of rasters into a single raster.

:py:func:`n2_images`  : Preprocesses a set of of Sentinel-2 images into single raster files.

:py:func:`clip_raster`  : Clips a raster to a shapefile

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

pyeo uses the same timestamp convention as ESA: `yyyymmddThhmmss`; for example, 1PM on 27th December 2020 would be
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
In pyeo, it takes the form of a 6-element tuple; for north-up images, these are the following.

.. code:: python

    geotransform[0] = top_left_x
    geotransfrom[1] = pixel_width
    geotransform[2] = 0
    geotransform[3] = top_left_y
    geotransform[4] = 0
    geotransform[5] = pixel_height

A geotransform can be obtained from a raster with the following snippet:

.. code:: python

    image = gdal.Open("my_raster.tif")
    gt = image.GetGeoTransform()

For more information, see the following: See the following: https://gdal.org/user/raster_data_model.html#affine-geotransform

Projections
-----------
Each Gdal raster also has a projection, defining (among other things) the unit of the geotransform.
Projections in pyeo are referred to either by EPSG number or passed around as a wkt( well-known text) string.
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
import faulthandler
import glob
import logging
import numpy as np

import os
from osgeo import gdal
from osgeo import gdalconst
from osgeo import gdal_array, osr, ogr
from osgeo.gdal_array import (
    NumericTypeCodeToGDALTypeCode,
    GDALTypeCodeToNumericTypeCode,
)

#import pdb
import re
import shutil
import subprocess
from skimage import morphology as morph
from tempfile import TemporaryDirectory
import scipy.ndimage as ndimage
#import itertools as iterate
#import matplotlib.pylab as pl
from matplotlib import cm
#from lxml import etree
import warnings

from pyeo.coordinate_manipulation import (
    get_combined_polygon,
    pixel_bounds_from_polygon,
    write_geometry,
    get_aoi_intersection,
    get_raster_bounds,
    align_bounds_to_whole_number,
    get_poly_bounding_rect,
    reproject_vector,
    get_local_top_left,
)
from pyeo.array_utilities import project_array
from pyeo.filesystem_utilities import (
    sort_by_timestamp,
    get_sen_2_tiles,
    get_l1_safe_file,
    get_sen_2_image_timestamp,
    get_sen_2_image_tile,
    get_sen_2_granule_id,
    check_for_invalid_l2_data,
    get_mask_path,
    get_sen_2_baseline,
    get_safe_product_type,
    get_change_detection_dates,
    get_filenames,
    get_raster_paths,
    get_image_acquisition_time,
    serial_date_to_string,
)
from pyeo.exceptions import (
    CreateNewStacksException,
    StackImagesException,
    BadS2Exception,
    NonSquarePixelException,
)

gdal.UseExceptions()

log = logging.getLogger("pyeo")

import pyeo.windows_compatability

faulthandler.enable()


def create_matching_dataset(
    in_dataset: gdal.Dataset,
    out_path: str,
    format: str = "GTiff",
    bands: int =1,
    datatype=None
):
    """
    Creates an empty gdal dataset with the same dimensions, projection and geotransform as in_dataset.
    Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified.

    Parameters
    ----------
    in_dataset : gdal.Dataset
        A gdal.Dataset object
    out_path : str
        The path to save the copied dataset to
    format : str, optional
        The Gdal image format. Defaults to geotiff ("GTiff"); for a full list, see https://gdal.org/drivers/raster/index.html
    bands : int, optional
        The number of bands in the dataset. Defaults to 1.
    datatype : gdal constant, optional
        The datatype of the returned dataset. See the introduction for this module. Defaults to in_dataset's datatype
        if not supplied.

    Returns
    -------
    new_dataset : gdal.Dataset
        An gdal.Dataset of the new, empty dataset that is ready for writing.

    Warnings
    --------
    The default of bands=1 will be changed to match the input dataset in the next release of pyeo

    """
    try:
        driver = gdal.GetDriverByName(format)
    except RunTimeError as e:
        log.error("GDAL.GetDriverByName error: {}".format(e))
        sys.exit(1)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    try:
        out_dataset = driver.Create(
            out_path,
            xsize=in_dataset.RasterXSize,
            ysize=in_dataset.RasterYSize,
            bands=bands,
            eType=datatype,
            options=["BigTIFF=IF_NEEDED"],
        )
    except:
        log.warning(
            "GDAL option BigTIFF could not be set in create_matching_dataset. Trying without it."
        )
        out_dataset = driver.Create(
            out_path,
            xsize=in_dataset.RasterXSize,
            ysize=in_dataset.RasterYSize,
            bands=bands,
            eType=datatype,
        )
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


def save_array_as_image(array, path, geotransform, projection, format="GTiff"):
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
    driver = gdal.GetDriverByName(str(format))
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    # If array is 2d, give it an extra dimension.
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)
    out_dataset = driver.Create(
        path,
        xsize=array.shape[2],
        ysize=array.shape[1],
        bands=array.shape[0],
        eType=type_code,
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
    Assumes that each image in image_dir is saved with a Sentinel-2 identifier name - see merge_raster.

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
    tiles = list(set(tiles))  # eliminate duplicates
    n_tiles = len(tiles)
    log.info("Found {} unique tile IDs for stacking:".format(n_tiles))
    for tile in tiles:
        log.info("   {}".format(tile))
    for tile in tiles:
        log.info("Tile ID for stacking: {}".format(tile))
        safe_files = glob.glob(
            os.path.join(image_dir, "*" + tile + "*.tif")
        )  # choose all files with that tile ID
        if len(safe_files) == 0:
            raise CreateNewStacksException(
                "Image_dir is empty: {}".format(os.path.join(image_dir, tile + "*.tif"))
            )
        else:
            safe_files = sort_by_timestamp(safe_files)
            log.info("Image file list for pairwise stacking for this tile:")
            for file in safe_files:
                log.info("   {}".format(file))
            # For each image in the list, stack the oldest with the next oldest. Then set oldest to next oldest
            # and repeat
            latest_image_path = safe_files[0]
            for image in safe_files[1:]:
                new_images.append(
                    stack_old_and_new_images(image, latest_image_path, stack_dir)
                )
                latest_image_path = image
    return new_images


def stack_image_with_composite(
    image_path,
    composite_path,
    out_dir,
    create_combined_mask=True,
    skip_if_exists=True,
    invert_stack=False,
):
    """
    Creates a single 8-band geotif image with a cloud-free composite and an image, and saves the result in out_dir. Output images are named
    "composite_tile_timestamp-of-composite_timestamp-of-image". Bands 1,2,3 and 4 are the B,G,R and NIR bands of the
    composite, and bands 5,6,7 and 8 are the B,G,R and NIR bands of the image.

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
    out_filename = "composite_{}_{}_{}.tif".format(
        tile, composite_timestamp, image_timestamp
    )
    out_path = os.path.join(out_dir, out_filename)
    out_mask_path = out_path.rsplit(".")[0] + ".msk"
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
        combine_masks(
            [comp_mask_path, image_mask_path],
            out_mask_path,
            combination_func="and",
            geometry_func="intersect",
        )
    return out_path


def stack_images(
    raster_paths,
    out_raster_path,
    geometry_mode="intersect",
    format="GTiff",
    datatype=gdal.GDT_Int32,
    nodata_value=0,
):
    """
    When provided with a list of rasters, will stack them into a single raster. The number of
    bands in the output is equal to the total number of bands in the input raster. Geotransform and projection
    are taken from the first raster in the list; there may be unexpected behaviour if multiple differing
    projections are provided.

    Parameters
    ----------
    raster_paths : list of str
        A list of paths to the rasters to be stacked, in order.
    out_raster_path : str
        The path to the saved output raster.
    geometry_mode : {'intersect' or 'union'}
        Can be either 'intersect' or 'union'.
    - If 'intersect', then the output raster will only contain the pixels of the input rasters that overlap.
    - If 'union', then the output raster will contain every pixel in the outputs. Layers without data will
        have their pixel values set to 0.
    format : str, optional
        The GDAL image format for the output. Defaults to 'GTiff'
    datatype : gdal datatype, optional
        The datatype of the gdal array - see introduction. Defaults to gdal.GDT_Int32.

    """

    log.info("Merging band rasters into a single file:")
    if len(raster_paths) <= 1:
        raise StackImagesException("stack_images requires at least two input images")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    total_layers = sum(raster.RasterCount for raster in rasters)
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = (
        in_gt[5] * -1
    )  # Y resolution in affine geotransform is -ve for Maths reasons
    combined_polygons = get_combined_polygon(rasters, geometry_mode)

    # Creating a new gdal object
    out_raster = create_new_image_from_polygon(
        combined_polygons,
        out_raster_path,
        x_res,
        y_res,
        total_layers,
        projection,
        format,
        datatype,
    )

    # I've done some magic here. GetVirtualMemArray lets you change a raster directly without copying
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    out_raster_array[...] = nodata_value
    present_layer = 0
    for i, in_raster in enumerate(rasters):
        log.info("  {}".format(raster_paths[i]))
        in_raster_array = in_raster.GetVirtualMemArray()
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
            out_raster, combined_polygons
        )
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
            in_raster, combined_polygons
        )
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        # Gdal does band, y, x
        out_raster_view = out_raster_array[
            present_layer : present_layer + in_raster.RasterCount,
            out_y_min:out_y_max,
            out_x_min:out_x_max,
        ]
        in_raster_view = in_raster_array[
            0 : in_raster.RasterCount, in_y_min:in_y_max, in_x_min:in_x_max
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
    out_raster_band_count = in_raster.RasterCount - len(bands_to_strip)
    out_raster = create_matching_dataset(
        in_raster, out_raster_path, bands=out_raster_band_count
    )
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    in_raster_array = in_raster.GetVirtualMemArray()

    bands_to_copy = [
        band for band in range(in_raster_array.shape[0]) if band not in bands_to_strip
    ]

    out_raster_array[...] = in_raster_array[bands_to_copy, :, :]

    out_raster_array = None
    in_raster_array = None
    out_raster = None
    in_raster = None

    return out_raster_path


def average_images(
    raster_paths,
    out_raster_path,
    geometry_mode="intersect",
    format="GTiff",
    datatype=gdal.GDT_Int32):
    """
    When provided with a list of rasters, will stack them into a single raster. The number of
    bands in the output is equal to the total number of bands in the input. Geotransform and projection
    are taken from the first raster in the list; there may be unexpected behavior if multiple differing
    projections are provided.

    Parameters
    ----------
    raster_paths : list of str
        A list of paths to the rasters to be stacked, in order.
    out_raster_path : str
        The path to the saved output raster.
    geometry_mode : {'intersect' or 'union'}, optional
        Can be either 'intersect' or 'union'. Defaults to 'intersect'.
    - If 'intersect', then the output raster will only contain the pixels of the input rasters that overlap.
    - If 'union', then the output raster will contain every pixel in the outputs. Layers without data will have their pixel values set to 0.
    format : str
        The GDAL image format for the output. Defaults to 'GTiff'
    datatype : gdal datatype
        The datatype of the gdal array - see note. Defaults to gdal.GDT_Int32

    """

    log.info("Stacking images {}".format(raster_paths))
    if len(raster_paths) <= 1:
        raise StackImagesException("stack_images requires at least two input images")
    rasters = [gdal.Open(raster_path) for raster_path in raster_paths]
    most_rasters = max(raster.RasterCount for raster in rasters)
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = (
        in_gt[5] * -1
    )  # Y resolution in affine geotransform is -ve for Maths reasons
    combined_polygons = get_combined_polygon(rasters, geometry_mode)

    # Creating a new gdal object
    out_raster = create_new_image_from_polygon(
        combined_polygons,
        out_raster_path,
        x_res,
        y_res,
        most_rasters,
        projection,
        format,
        datatype,
    )
    if out_raster is None:
        log.error("Could not create: {}".format(out_raster_path))

    # I've done some magic here. GetVirtualMemArray lets you change a raster directly without copying
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
    if len(out_raster_array.shape == 2):
        out_raster_array = np.expand_dims(out_raster_array, 3)
    present_layer = 0
    for i, in_raster in enumerate(rasters):
        log.info("Stacking image {}".format(i))
        in_raster_array = in_raster.GetVirtualMemArray()
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
            out_raster, combined_polygons
        )
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
            in_raster, combined_polygons
        )
        if len(in_raster_array.shape) == 2:
            in_raster_array = np.expand_dims(in_raster_array, 0)
        # Gdal does band, y, x
        out_raster_view = out_raster_array[:out_y_min:out_y_max, out_x_min:out_x_max]
        in_raster_view = in_raster_array[:in_y_min:in_y_max, in_x_min:in_x_max]
        out_raster_view[...] = out_raster_view + in_raster_view  # add up
        out_raster_view = None
        in_raster_view = None
        present_layer += in_raster.RasterCount
        in_raster_array = None
        in_raster = None
    out_raster_view[...] = out_raster_view / len(rasters)  # calculate average
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
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        in_raster = gdal.Open(in_raster_path)
        in_gt = in_raster.GetGeoTransform()
        x_res = in_gt[1]
        y_res = in_gt[5] * -1
        temp_band = in_raster.GetRasterBand(1)
        datatype = temp_band.DataType
        out_raster = create_new_image_from_polygon(
            polygon,
            out_raster_path,
            x_res,
            y_res,
            in_raster.RasterCount,
            in_raster.GetProjection(),
            format,
            datatype,
        )
        out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
            out_raster, polygon
        )
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
            in_raster, polygon
        )
        out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
        in_raster_array = in_raster.GetVirtualMemArray()
        out_raster_view = out_raster_array[:, out_y_min:out_y_max, out_x_min:out_x_max]
        in_raster_view = in_raster_array[:, in_y_min:in_y_max, in_x_min:in_x_max]
        out_raster_view[...] = in_raster_view
        out_raster_view = None
        in_raster_view = None
        out_raster_array = None
        in_raster_array = None
        out_raster = None
        in_raster = None


def mosaic_images(
    raster_path, out_raster_path, format="GTiff", datatype=gdal.GDT_Int32, nodata=0
):
    """
    Mosaics multiple images in the directory raster_path with the same number of layers into one single image.
    Overwrites overlapping pixels with the value furthest down raster_paths. Takes projection from the first raster.
    The output mosaic file will have a name that contains all unique tile IDs and the earliest and latest
    acquisition date and time from the raster file names.

    Parameters
    ----------
    raster_path : str
        The directory path containing all Geotiff rasters to be mosaiced (all having the same number of bands)
    out_raster_path : str
        The path to the output directory for the new mosaic file
    format : str
        The image format of the output raster. Defaults to 'GTiff'
    datatype : gdal datatype
        The datatype of the output raster. Defaults to gdal.GDT_Int32
    nodata : number
        The input nodata value; any pixels in raster_paths with this value will be ignored. Defaults to 0.

    """

    # This, again, is very similar to stack_rasters
    log = logging.getLogger(__name__)
    log.info("--------------------------------------")
    log.info("Beginning mosaicking of all tiff images in {}".format(raster_path))
    raster_files = [
        raster_file
        for raster_file in os.listdir(raster_path)
        if raster_file.endswith(".tif") or raster_file.endswith(".tiff")
    ]
    rasters = [
        gdal.Open(os.path.join(raster_path, raster_file))
        for raster_file in raster_files
    ]
    log.info(
        "Found {} tiff files in directory {}".format(len(raster_files), raster_path)
    )
    projection = rasters[0].GetProjection()
    in_gt = rasters[0].GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1  # Y resolution in agt is -ve for Maths reasons
    combined_polygon = align_bounds_to_whole_number(
        get_combined_polygon(rasters, geometry_mode="union")
    )
    layers = rasters[0].RasterCount
    tiles = out_raster_path + "/mosaic_"
    all_tiles = get_sen_2_tiles(raster_path)
    all_tiles_as_set = set(
        all_tiles
    )  # a trick to remove duplicate strings from a list is to convert them to a set and back
    all_tiles = list(all_tiles_as_set)
    log.info("All tiles without duplicates: {}".format(all_tiles))
    for t, tile in enumerate(all_tiles):
        tiles = tiles + tile + "_"
    out_raster_file = tiles
    dt_str = []
    for raster_file in raster_files:
        date_time = get_change_detection_dates(raster_file)
        for i, dt in enumerate(date_time):
            dt_str.append(dt.strftime("%Y%m%d%H%M%S"))
            if i > 1:
                log.warning(
                    "More than two acquisition dates / times found in raster file name: {}".format(
                        raster_file
                    )
                )
            log.info(
                "File {} contains imagery from acquisition date / time of {}".format(
                    raster_file, dt.strftime("%Y%m%d%H%M%S")
                )
            )
    dt_str = sorted(dt_str)
    dt_min = dt_str[0]
    dt_max = dt_str[-1]
    out_raster_file = out_raster_file + dt_min + "_" + dt_max + ".tif"
    if os.path.isfile(out_raster_file):
        log.info(
            "Mosaic output file already exists. Skipping the mosaicking step. {}".format(
                out_raster_file
            )
        )
    else:
        out_raster = create_new_image_from_polygon(
            combined_polygon,
            out_raster_file,
            x_res,
            y_res,
            layers,
            projection,
            format,
            datatype,
            nodata=nodata,
        )
        log.info("New empty image mosaic created at {}".format(out_raster_file))
        out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)
        out_raster_array[...] = nodata
        old_nodata = nodata
        for i, raster in enumerate(rasters):
            log.info(
                "Now mosaicking raster no. {} out of {}".format(i + 1, len(rasters))
            )
            in_raster_array = raster.GetVirtualMemArray()
            if len(in_raster_array.shape) == 2:
                in_raster_array = np.expand_dims(in_raster_array, 0)
            if np.isnan(old_nodata):
                nodata = (
                    np.nan_to_num(in_raster_array).max() + 1
                )  # This is turning into a _real_ hack now
                in_raster_array = np.nan_to_num(in_raster_array, nan=nodata, copy=True)
            in_bounds = get_raster_bounds(raster)
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
                out_raster, in_bounds
            )
            out_raster_view = out_raster_array[
                :, out_y_min:out_y_max, out_x_min:out_x_max
            ]
            np.copyto(out_raster_view, in_raster_array, where=in_raster_array != nodata)
            in_raster_array = None
            out_raster_view = None
        log.info("Raster mosaicking done")
        out_raster_array = None
        out_raster = None


def update_composite_with_images(
    composite_in_path,
    in_raster_path_list,
    composite_out_path,
    format="GTiff",
    generate_date_image=True,
    missing=0,
):
    """
    Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Will also create a mask and
    (optionally) a date image in the same directory.

    Parameters
    ----------
    composite_in_path : str
        The path of the input composite image to be updated
    in_raster_path_list : list of str
        A list of paths to rasters.
    composite_out_path : str
        The path of the output image
    format : str, optional
        The gdal format of the image. Defaults to "GTiff"
    generate_date_image : bool, optional
        If true, generates a single-layer raster containing the dates of each image detected - see below.
    missing : missing value to be ignored, 0 by default

    Returns
    -------
    composite_path : str
        The path to the composite.

    Notes:

    If generate_date_images is True, an raster ending with the suffix .date will be created; each pixel will contain the
    timestamp (yyyymmdd) of the date that pixel was last seen in the composite.

    """
    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(str(format))
    in_raster_list = [gdal.Open(raster) for raster in in_raster_path_list]
    in_composite = gdal.Open(composite_in_path)
    projection = in_composite.GetProjection()
    in_gt = in_composite.GetGeoTransform()
    x_res = in_gt[1]
    y_res = in_gt[5] * -1
    n_bands = in_composite.RasterCount
    temp_band = in_composite.GetRasterBand(1)
    datatype = temp_band.DataType
    temp_band = None

    # Creating output image + array
    log.info("Creating updated composite at {}".format(composite_out_path))
    log.info("  based on previous composite {}".format(composite_in_path))
    log.info(
        "  x_res: {}, y_res: {}, {} bands, datatype: {}, projection: {}".format(
            x_res, y_res, n_bands, datatype, projection
        )
    )
    out_bounds = align_bounds_to_whole_number(
        get_poly_bounding_rect(
            get_combined_polygon(in_raster_list + [in_composite], geometry_mode="union")
        )
    )
    composite_image = create_new_image_from_polygon(
        out_bounds,
        composite_out_path,
        x_res,
        y_res,
        n_bands,
        projection,
        format,
        datatype,
    )
    if generate_date_image:
        time_out_path = composite_out_path.rsplit(".")[0] + ".dates"
        dates_image = create_matching_dataset(
            composite_image, time_out_path, bands=1, datatype=gdal.GDT_UInt32
        )
        dates_array = dates_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()

    output_array = composite_image.GetVirtualMemArray(
        eAccess=gdal.gdalconst.GF_Write, datatype=gdal.GDT_Float32
    )
    if len(output_array.shape) == 2:
        output_array = np.expand_dims(output_array, 0)

    in_composite_array = get_array(in_composite)
    np.copyto(output_array, in_composite_array)

    for i, in_raster in enumerate(in_raster_list):
        # Get a view of in_raster according to output_array
        log.info("Adding {} to composite".format(in_raster_path_list[i]))
        in_bounds = align_bounds_to_whole_number(get_raster_bounds(in_raster))
        x_min, x_max, y_min, y_max = pixel_bounds_from_polygon(
            composite_image, in_bounds
        )
        output_view = output_array[:, y_min:y_max, x_min:x_max]

        # Move every pixel except missing values in in_raster to output_view
        in_array = get_array(in_raster)
        log.info("Type of in_array: {}".format(in_array.dtype))
        log.info("Type of output_view: {}".format(output_view.dtype))
        np.copyto(
            output_view, in_array, where=np.where(in_array == missing, False, True)
        )

        # Save dates in date_image if needed
        if generate_date_image:
            dates_view = dates_array[y_min:y_max, x_min:x_max]
            # Gets timestamp as integer in form yyyymmdd
            date = np.uint32(
                get_sen_2_image_timestamp(in_raster.GetFileList()[0]).split("T")[0]
            )
            dates_view[np.logical_not(in_masked.mask[0, ...])] = date
            dates_view = None

        # Deallocate
        output_view = None
        in_masked = None

    output_array = None
    dates_array = None
    dates_image = None
    composite_image = None

    log.info("Composite update done.")
    return composite_out_path


def composite_images_with_mask(
    in_raster_path_list, composite_out_path, format="GTiff", generate_date_image=True
):
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
    other. If it does not exist, composite_out_path will be created. Takes projection, resolution, etc. from first band
    of first raster in list. Will reproject images and masks if they do not match initial raster.

    If generate_date_images is True, an raster ending with the suffix .date will be created; each pixel will contain the
    timestamp (yyyymmdd) of the date that pixel was last seen in the composite.

    """

    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(str(format))
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
    log.info(
        "Composite info: x_res: {}, y_res: {}, {} bands, datatype: {}, projection: {}".format(
            x_res, y_res, n_bands, datatype, projection
        )
    )
    out_bounds = align_bounds_to_whole_number(
        get_poly_bounding_rect(
            get_combined_polygon(in_raster_list, geometry_mode="union")
        )
    )
    composite_image = create_new_image_from_polygon(
        out_bounds,
        composite_out_path,
        x_res,
        y_res,
        n_bands,
        projection,
        format,
        datatype,
    )

    if generate_date_image:
        time_out_path = composite_out_path.rsplit(".")[0] + ".dates"
        dates_image = create_matching_dataset(
            composite_image, time_out_path, bands=1, datatype=gdal.GDT_UInt32
        )
        dates_array = dates_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()

    output_array = composite_image.GetVirtualMemArray(eAccess=gdal.gdalconst.GF_Write)
    if len(output_array.shape) == 2:
        output_array = np.expand_dims(output_array, 0)

    mask_paths = []

    for i, in_raster in enumerate(in_raster_list):
        mask_paths.append(get_mask_path(in_raster_path_list[i]))

        # Get a view of in_raster according to output_array
        log.info("Adding {} to composite".format(in_raster_path_list[i]))
        in_bounds = align_bounds_to_whole_number(get_raster_bounds(in_raster))
        x_min, x_max, y_min, y_max = pixel_bounds_from_polygon(
            composite_image, in_bounds
        )
        output_view = output_array[:, y_min:y_max, x_min:x_max]

        # Move every unmasked pixel in in_raster to output_view
        log.info("Mask for {} at {}".format(in_raster_path_list[i], mask_paths[i]))
        in_masked = get_masked_array(in_raster, mask_paths[i])
        np.copyto(output_view, in_masked, where=np.logical_not(in_masked.mask))

        # Save dates in date_image if needed
        if generate_date_image:
            dates_view = dates_array[y_min:y_max, x_min:x_max]
            # Gets timestamp as integer in form yyyymmdd
            date = np.uint32(
                get_sen_2_image_timestamp(in_raster.GetFileList()[0]).split("T")[0]
            )
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
    log.info(
        "Creating composite mask at {}".format(
            composite_out_path.rsplit(".")[0] + ".msk"
        )
    )
    combine_masks(
        mask_paths,
        composite_out_path.rsplit(".")[0] + ".msk",
        combination_func="or",
        geometry_func="union",
    )
    return composite_out_path


def get_stats_from_raster_file(in_raster_path, format="GTiff", missing_data_value=0):
    """
    Gets simple statistics from a raster file and prints them to the log file

    Parameters
    ----------
    in_raster_path : str
        A path to a raster file

    format : str, optional
        The gdal format of the image. Defaults to "GTiff"

    missing_data_value : int, optional
        The encoding of missing values in the raster that will be omitted from the calculations

    Returns
    --------
    result : dictionary
        Dictionary containing the results of the function
    """

    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(str(format))
    in_raster = gdal.Open(in_raster_path)
    n_bands = in_raster.RasterCount
    result = {}
    for band in range(n_bands):
        # Read into NumPy array
        raster_band = in_raster.GetRasterBand(band + 1)
        in_array = raster_band.ReadAsArray()
        if missing_data_value is not None:
            in_array = np.ma.masked_equal(in_array, missing_data_value)
        if in_array.count() == 0:
            result.update(
                {"band_{}".format(band + 1): " contains only missing values."}
            )
        else:
            result.update(
                {
                    "band_{}".format(band + 1): "min=%.3f, max=%3f, mean=%3f, stdev=%3f"
                    % (
                        np.nanmin(in_array[~in_array.mask]),
                        np.nanmax(in_array[~in_array.mask]),
                        np.nanmean(in_array[~in_array.mask]),
                        np.nanstd(in_array[~in_array.mask]),
                    )
                }
            )
    log.info("Raster file stats for {}".format(in_raster_path))
    for key, item in result.items():
        log.info("   {} : {}".format(key, item))
    in_raster = None
    return result


def get_dir_size(path="."):
    """
    Gets the size of all contents of a directory.
    """
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def find_small_safe_dirs(path, threshold=600 * 1024 * 1024):
    """
    Quickly finds all subdirectories ending with ".SAFE" or ".safe" and logs a warning if the
    directory size is less than a threshold, 600 MB by default. This indicates incomplete downloads

    Returns a list of all paths to the SAFE directories that are smaller than the threshold and a list of all sizes.
    """
    dir_paths = [
        os.path.join(p, d)
        for p, ds, f in os.walk(path)
        for d in ds
        if os.path.isdir(os.path.join(p, d))
        and (d.endswith(".SAFE") or d.endswith(".safe"))
    ]
    if len(dir_paths) == 0:
        # log.info("No .SAFE directories found in {}.".format(path))
        return [], []
    small_dirs = []
    sizes = []
    for index, dir_path in enumerate(dir_paths):
        size = get_dir_size(dir_path)
        if size < threshold:
            log.warning(
                "Incomplete download likely: {} MB: {}".format(
                    str(round(size / 1024 / 1024)), dir_path
                )
            )
            small_dirs = small_dirs + [dir_path]
            sizes = sizes + [size]
    return small_dirs, sizes


def get_file_sizes(dir_path):
    """
    Gets all file sizes in bytes from the files contained in dir_path and its subdirs.

    Parameters
    ----------
    dir_path : str
        Paths to the files.

    Returns
    -------
    results : dictionary
        Dictionary containing the results of the function.
    """
    log = logging.getLogger(__name__)
    results = {}
    file_paths = [f.path for f in os.scandir(dir_path) if f.is_file()]
    if len(file_paths) == 0:
        log.warning("No files found in {} and its subdirectories.".format(dir_path))
    for f in file_paths:
        size = os.path.getsize(f)
        results.update({"{}".format(f): "{}".format(str(size))})
        log.info("File size of {} is {} MB.".format(f, str(size / 1024 / 2024)))
    for key, item in results.items():
        log.info("   {} : {}".format(key, item))
    return results


def clever_composite_images(
    in_raster_path_list,
    composite_out_path,
    format="GTiff",
    chunks=10,
    generate_date_image=True,
    missing_data_value=0,
    log=logging.getLogger(__name__),
):
    """
    Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Will also create
    (optionally) a date image in the same directory. Processes raster stacks by splitting them into a number of chunks
    to avoid memory allocation errors.

    Parameters
    ----------
    in_raster_path_list : list[str]
        A list of paths to rasters.
    composite_out_path : str
        The path of the output image
    format : str, optional
        The gdal format of the image. Defaults to "GTiff"
    generate_date_image : bool, optional
        If true, generates a single-layer raster containing the dates of each image detected - see below.
    missing_data_value : int, optional
        Value for no data encoding, will be ignored in calculating the median
    log : logger object

    Returns
    -------
    composite_path : str
        The path to the composite.


    Notes
    -----

    If generate_date_images is True, an raster ending with the suffix .date will be created; each pixel will contain the
    timestamp (yyyymmdd) of the date that pixel was last seen in the composite.

    """

    def median_of_raster_list(
        in_raster_path_list,
        out_raster_path,
        band=1,
        chunks=10,
        missing_data_value=0,
        format="GTiff",
        log=logging.getLogger(__name__),

    ):
        """
        Calculates the median of each pixel in a list of rasters with one band of the same dimensions and map projection.
        Excludes missing data values from the calculations. Processes raster stacks by splitting them into a number of chunks
        to avoid memory allocation errors.

        Parameters
        ----------
        in_raster_path_list : list[str]
            A list of paths to rasters
        out_raster_path : str
            The path of the output raster
        band : int, optional
            The number of the band to be processed, starting at 1. Defaults to band 1.
        missing_data_value : number, optional
            Value for no data encoding, will be ignored in calculating the median
        format : str, optional
            Raster format for GDAL
        log : logger object
        """

        in_raster = gdal.Open(in_raster_path_list[0])
        driver = gdal.GetDriverByName(str(format))
        projection = in_raster.GetProjection()
        in_gt = in_raster.GetGeoTransform()
        #log.info(
        #    f"Clever_composite_images() in_raster_path_list[0].geotransform = {in_raster.GetGeoTransform()}"
        #)
        n_bands = in_raster.RasterCount
        temp_band = in_raster.GetRasterBand(1)
        datatype = temp_band.DataType
        xsize = in_raster.RasterXSize
        ysize = in_raster.RasterYSize
        temp_band = None
        in_raster = None
        # check that all input rasters have the same size
        good_rasters = []
        for f in in_raster_path_list:
            in_raster = gdal.Open(f)
            #log.info(
            #    f"Clever_composite_images() in_raster_path_list.geotransform = {in_raster.GetGeoTransform()}"
            #)
            f_n_bands = in_raster.RasterCount
            f_xsize = in_raster.RasterXSize
            f_ysize = in_raster.RasterYSize
            in_raster = None
            if n_bands != f_n_bands:
                log.error("Raster band numbers are different. Skipping {}".format(f))
                continue
            if xsize != f_xsize:
                log.error("Raster x sizes are different. Skipping {}".format(f))
                continue
            if ysize != f_ysize:
                log.error("Raster y sizes are different. Skipping {}".format(f))
                continue
            good_rasters = good_rasters + [f]
        in_raster_path_list = good_rasters.copy()

        # determine chunk size
        chunksize = int(np.ceil(ysize / chunks))
        # create output raster file and copy metadata from first raster in the list
        result = driver.Create(
            out_raster_path, xsize=xsize, ysize=ysize, bands=1, eType=datatype
        )
        result.SetGeoTransform(in_gt)
        result.SetProjection(projection)
        log.info(
            "Chunk processing. {} chunks of size {}, {}.".format(
                chunks, chunksize, xsize
            )
        )
        for ch in range(chunks):
            log.info("Processing chunk {}".format(ch))
            xoff = 0
            yoff = ch * chunksize
            xs = xsize
            # work out the size of the last chunk (residual)
            if ch < chunks - 1:
                ys = chunksize
            else:
                ys = ysize - ch * chunksize
            # log.info("xoff, yoff, xsize, ysize: {}, {}, {}, {}".format(xoff,yoff,xs,ys))
            res = []  # reset the list of arrays that will contain the band rasters
            for f in in_raster_path_list:
                #log.info("Opening band {} from raster {}".format(band, f))
                ds = gdal.Open(f)
                b = ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xs, ys)
                if ds == None:
                    log.error(
                        "Opening band {} from raster {} failed. Skipping.".format(
                            band, f
                        )
                    )
                else:
                    res.append(b)
                ds = None
            # for i, this_res in enumerate(res):
            #    log.info("raster {} is {} and is of type {}".format(i, this_res, type(this_res)))
            #    log.info("raster {} has dimensions {}".format(i, this_res.shape))
            stacked = np.dstack(res)
            if missing_data_value is not None:
                ma = np.ma.masked_equal(stacked, missing_data_value)
                stacked[ma.mask] = np.nan
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
            try:
                median_raster = np.nanmedian(stacked, axis=-1)
            except ValueError as e:
                log.warning(
                    "Likely all NaN slice encountered in chunk processing: {}".format(e)
                )
                median_raster = np.full_like(stacked[:, :, 0], missing_data_value)
            # catch pixels where all raster layers have NaN values and set them to missing_data_value
            all_nan_locations = np.isnan(stacked).all(axis=-1)
            median_raster[all_nan_locations] = missing_data_value
            b = result.GetRasterBand(1).ReadAsArray()
            # log.info("Broadcasting from [{}:{}, {}:{}]".format(0, median_raster.shape[0], 0, median_raster.shape[1]))
            # log.info("             into [{}:{}, {}:{}]".format(yoff, yoff+ys, xoff, xoff+xs))
            b[yoff : (yoff + ys), xoff : (xoff + xs)] = median_raster
            result.GetRasterBand(1).WriteArray(b)
        result = None
        res = None
        b = None
        median_raster = None
        return

    in_raster = gdal.Open(in_raster_path_list[0])
    n_bands = in_raster.RasterCount
    in_raster = None
    #log.info("Creating median composite at {}".format(composite_out_path))
    #log.info("Using {} input raster files:".format(len(in_raster_path_list)))
    for i in in_raster_path_list:
        log.info("   {}".format(i))
    # use raster stacking to calculate the median over all masked raster files and all 4 bands
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        tmpfiles = []
        for band in range(n_bands):
            '''
            log.info(
                "Band {} - calculating median composite across all images using raster stacking".format(
                    band + 1
                )
            )
            log.info("   Ignoring missing data value of {}".format(missing_data_value))
            '''
            tmpfile = os.path.join(
                td,
                os.path.basename("tmp_" + composite_out_path).split(".")[0]
                + "_band"
                + str(band + 1)
                + ".tif",
            )
            # tmpfile = os.path.abspath(composite_out_path.split('.')[0] + '_tmp_band' + str(band+1) + '.tif')
            tmpfiles.append(tmpfile)
            median_of_raster_list(
                in_raster_path_list,
                tmpfile,
                band=band + 1,
                chunks=chunks,
                missing_data_value=missing_data_value,
                log=log
            )
            # log some image stats
            get_stats_from_raster_file(tmpfile)
        #log.info("Finished median calculations.")
        # Aggregating band composites into a single tiff file
        #log.info("Aggretating band composites into a single raster file.")
        stack_images(tmpfiles, composite_out_path, geometry_mode="intersect")
        get_stats_from_raster_file(composite_out_path)

    #log.info("Clever_composite_images() Fixing composite geotransform "+
    #   "to ensure it matches first of contributing images")
    in_raster = gdal.Open(in_raster_path_list[0])
    in_gt = in_raster.GetGeoTransform()
    composite_raster = gdal.Open(composite_out_path)
    composite_raster.SetGeoTransform(in_gt)
    in_raster = None
    composite_raster = None

    #log.info("-------------------------------------------------")
    #log.info("Median composite done")
    #log.info("-------------------------------------------------")
    return composite_out_path


def clever_composite_images_with_mask(
    in_raster_path_list,
    composite_out_path,
    format="GTiff",
    chunks=10,
    generate_date_image=True,
    missing_data_value=0,
    log=logging.getLogger(__name__),
):
    """
    Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Will also create a mask and
    (optionally) a date image in the same directory. Processes raster stacks by splitting them into a number of chunks
    to avoid memory allocation errors.

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
    missing_data_value : int, optional
        Value for no data encoding, will be ignored in calculating the median
    log : logger object

    Returns
    -------
    composite_path : str
        The path to the composite.

    Notes
    -----
    Masks are assumed to be a multiplicative .msk file with the same path as their corresponding image.
    All images must have the same number of layers and resolution, but do not have to be perfectly co-registered.
    If it does not exist, composite_out_path will be created. Takes projection, resolution, etc. from first band
    of first raster in list. Will reproject images and masks if they do not match initial raster.

    If generate_date_images is True, an raster ending with the suffix .date will be created; each pixel will contain the
    timestamp (yyyymmdd) of the date that pixel was last seen in the composite.

    #TODO: Change the date image to contain Julian date numbers.

    """

    def median_of_raster_list(
        in_raster_path_list,
        out_raster_path,
        band=1,
        chunks=10,
        missing_data_value=0,
        format="GTiff",
        log=logging.getLogger(__name__),
    ):
        """
        Calculates the median of each pixel in a list of rasters with one band of the same dimensions and map projection.
        Excludes missing data values from the calculations. Processes raster stacks by splitting them into a number of chunks
        to avoid memory allocation errors.

        Parameters
        ----------
        in_raster_path_list : list of str
            A list of paths to rasters
        out_raster_path : str
            The path of the output raster
        band : int, optional
            The number of the band to be processed, starting at 1. Defaults to band 1.
        missing_data_value : number, optional
            Value for no data encoding, will be ignored in calculating the median
        format : str, optional
            Raster format for GDAL
        log : logger object
        """

        in_raster = gdal.Open(in_raster_path_list[0])
        driver = gdal.GetDriverByName(str(format))
        projection = in_raster.GetProjection()
        in_gt = in_raster.GetGeoTransform()
        temp_band = in_raster.GetRasterBand(1)
        datatype = temp_band.DataType
        xsize = in_raster.RasterXSize
        ysize = in_raster.RasterYSize
        temp_band = None
        in_raster = None
        chunksize = int(np.ceil(ysize / chunks))
        # create output raster file and copy metadata from first raster in the list
        result = driver.Create(
            out_raster_path, xsize=xsize, ysize=ysize, bands=1, eType=datatype
        )
        result.SetGeoTransform(in_gt)
        result.SetProjection(projection)
        log.info(
            "Chunk processing. {} chunks of size {}, {}.".format(
                chunks, chunksize, xsize
            )
        )
        for ch in range(chunks):
            log.info("Processing chunk {}".format(ch))
            xoff = 0
            yoff = ch * chunksize
            xs = xsize
            # work out the size of the last chunk (residual)
            if ch < chunks - 1:
                ys = chunksize
            else:
                ys = ysize - ch * chunksize
            # log.info("xoff, yoff, xsize, ysize: {}, {}, {}, {}".format(xoff,yoff,xs,ys))
            res = []  # reset the stack of band rasters from all time slices
            for f in in_raster_path_list:
                # log.info("Opening raster {}".format(f))
                ds = gdal.Open(f)
                b = ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xs, ys)
                res.append(b)
                ds = None
            stacked = np.dstack(res)
            if missing_data_value is not None:
                ma = np.ma.masked_equal(stacked, missing_data_value)
                stacked[ma.mask] = np.nan
            median_raster = np.nanmedian(stacked, axis=-1)
            # catch pixels where all rasters have NaN values and set them to missing_data_value
            all_nan_locations = np.isnan(stacked).all(axis=-1)
            median_raster[all_nan_locations] = missing_data_value
            b = result.GetRasterBand(1).ReadAsArray()
            # log.info("Broadcasting from [{}:{}, {}:{}]".format(0, median_raster.shape[0], 0, median_raster.shape[1]))
            # log.info("             into [{}:{}, {}:{}]".format(yoff, yoff+ys, xoff, xoff+xs))
            b[yoff : (yoff + ys), xoff : (xoff + xs)] = median_raster
            result.GetRasterBand(1).WriteArray(b)
        result = None
        res = None
        b = None
        median_raster = None

    log = logging.getLogger(__name__)
    driver = gdal.GetDriverByName(str(format))
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
    log.info("-------------------------------------------------")
    log.info("Creating median composite at {}".format(composite_out_path))
    log.info("Using {} input raster files:".format(len(in_raster_path_list)))
    for i in in_raster_path_list:
        log.info("   {}".format(i))

    mask_paths = []
    for i, in_raster in enumerate(in_raster_path_list):
        mask_paths.append(get_mask_path(in_raster_path_list[i]))
    mask_paths = [
        f for f in mask_paths if "_masked" not in os.path.basename(f)
    ]  # remove already masked tiff files to avoid double-processing
    log.info("Mask paths for input raster files:")
    for i in mask_paths:
        log.info("   {}".format(i))

    # apply mask to each raster file
    masked_image_paths = []
    image_files = [
        f for f in in_raster_path_list if "_masked" not in os.path.basename(f)
    ]  # remove already masked tiff files to avoid double-processing
    for i, in_raster in enumerate(image_files):
        masked_image_path = in_raster.split(".")[0] + "_masked.tif"
        masked_image_paths.append(masked_image_path)
        if os.path.exists(masked_image_path):
            log.info(
                "Output file {} already exists, skipping the masking step.".format(
                    masked_image_path
                )
            )
        else:
            log.info("Producing cloud-masked image file: {}".format(masked_image_path))
            # log.info('   from mask: {}'.format(mask_paths[i]))
            # log.info('   and raster: {}'.format(in_raster))
            apply_mask_to_image(mask_paths[i], in_raster, masked_image_path)
            # log some image stats
            get_stats_from_raster_file(in_raster)
            get_stats_from_raster_file(masked_image_path)
            log.info("Finished application of masks to rasters.")

    # use raster stacking to calculate the median over all masked raster files and all 4 bands
    log.info("Beginning median calculations.")
    tmpfiles = []
    for band in range(n_bands):
        log.info(
            "Band {} - calculating median composite across all images using raster stacking".format(
                band + 1
            )
        )
        log.info("   Ignoring missing data value of {}".format(missing_data_value))
        tmpfile = os.path.abspath(
            composite_out_path.split(".")[0] + "_tmp_band" + str(band + 1) + ".tif"
        )
        tmpfiles.append(tmpfile)
        median_of_raster_list(
            masked_image_paths,
            tmpfile,
            band=band + 1,
            chunks=chunks,
            missing_data_value=missing_data_value,
            log=log
        )
        # log some image stats
        get_stats_from_raster_file(tmpfile)
    log.info("Finished median calculations.")

    # Aggretating band composites into a single tiff file
    log.info("Aggretating band composites into a single raster file.")
    stack_images(tmpfiles, composite_out_path, geometry_mode="intersect")
    get_stats_from_raster_file(composite_out_path)
    for tmpfile in tmpfiles:
        os.remove(tmpfile)
    log.info("Median composite done")
    log.info(
        "Creating composite mask at {}".format(
            composite_out_path.rsplit(".")[0] + ".msk"
        )
    )
    combine_masks(
        mask_paths,
        composite_out_path.rsplit(".")[0] + ".msk",
        combination_func="or",
        geometry_func="union",
    )
    return composite_out_path


def reproject_directory(in_dir, out_dir, new_projection, extension=".tif"):
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
    image_paths = [
        os.path.join(in_dir, image_path)
        for image_path in os.listdir(in_dir)
        if image_path.endswith(extension)
    ]
    for image_path in image_paths:
        reproj_path = os.path.join(out_dir, os.path.basename(image_path))
        # log.info("Reprojecting {} to projection with EPSG code {}, storing in {}".format(image_path, reproj_path, new_projection))
        reproject_image(image_path, reproj_path, new_projection)


def reproject_image(
    in_raster,
    out_raster_path,
    new_projection,
    driver="GTiff",
    memory=2e3,
    do_post_resample=True,
):
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
    log = logging.getLogger(__name__)

    log.info("Reprojecting {}".format(in_raster))
    if type(new_projection) is int:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(new_projection)
        new_projection = proj.ExportToWkt()
        # epsg = proj.GetAttrValue('AUTHORITY',1)
        # log.info("Reprojecting {} to EPSG code {}".format(in_raster, epsg))
    else:
        pass
        # log.info("Reprojecting {} to {}".format(in_raster, new_projection))
    if type(in_raster) is str:
        in_raster = gdal.Open(in_raster)
    res = in_raster.GetGeoTransform()[1]
    gdal.Warp(
        out_raster_path,
        in_raster,
        dstSRS=new_projection,
        warpMemoryLimit=memory,
        format=driver,
    )
    # After warping, image has irregular gt; resample back to previous pixel size
    if do_post_resample:
        resample_image_in_place(out_raster_path, res)
    return out_raster_path


def composite_directory(
    image_dir, composite_out_dir, format="GTiff", generate_date_images=False
):
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
    log.info("Compositing all images in directory {}".format(image_dir))
    sorted_image_paths = [
        os.path.join(image_dir, image_name)
        for image_name in sort_by_timestamp(
            os.listdir(image_dir), recent_first=False
        )  # Let's think about this
        if image_name.endswith(".tif")
    ]
    log.info(sorted_image_paths)
    log.info(len(sorted_image_paths))
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(
        composite_out_dir, "composite_{}.tif".format(last_timestamp)
    )
    composite_images_with_mask(
        sorted_image_paths,
        composite_out_path,
        format,
        generate_date_image=generate_date_images,
    )
    return composite_out_path


def clever_composite_directory(
    image_dir,
    composite_out_dir,
    format="GTiff",
    chunks=10,
    generate_date_images=False,
    missing_data_value=0,
    log=logging.getLogger(__name__),
):
    """
    Using clever_composite_images, creates a composite containing every image in image_dir. This will
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
    missing_data_value : int, optional
        Value for no data encoding, will be ignored in calculating the median
    log : logger object

    Returns
    -------
    composite_out_path : str
        The path to the new composite

    """
    log.info(
        "Cleverly compositing all images in directory into a median composite: {}".format(
            image_dir
        )
    )
    image_paths = [
        os.path.join(image_dir, image_name)
        for image_name in sort_by_timestamp(os.listdir(image_dir), recent_first=False)
        if image_name.endswith(".tif")
    ]
    # log.info("Sorted image paths: {}".format(image_paths))

    sorted_image_paths = [
        f for f in image_paths if "_masked" not in os.path.basename(f)
    ]  # remove already masked tiff files to avoid double-processing
    # log.info("Sorted image paths, no masks: {}".format(sorted_image_paths))

    # get timestamp of most recent image in the directory
    for i in range(len(sorted_image_paths)):
        timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[i]))
        log.info("Image number {} has time stamp {}".format(i + 1, timestamp))
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    tile = get_sen_2_image_tile(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(
        composite_out_dir, "composite_{}_{}.tif".format(tile, last_timestamp)
    )
    clever_composite_images(
        sorted_image_paths,
        composite_out_path,
        format,
        chunks=chunks,
        generate_date_image=generate_date_images,
        missing_data_value=missing_data_value,
        log = log
    )
    return composite_out_path


def clever_composite_directory_with_masks(
    image_dir,
    composite_out_dir,
    format="GTiff",
    chunks=10,
    generate_date_images=False,
    missing_data_value=0,
    log=logging.getLogger(__name__),
):
    """
    Using clever_composite_images_with_mask, creates a composite containing every image in image_dir. This will
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
    missing_data_value : int, optional
        Value for no data encoding, will be ignored in calculating the median
    log : logger object
    
    Returns
    -------
    composite_out_path : str
        The path to the new composite

    """

    log.info(
        "Cleverly compositing all images in directory into a median composite: {}".format(
            image_dir
        )
    )
    image_paths = [
        os.path.join(image_dir, image_name)
        for image_name in sort_by_timestamp(os.listdir(image_dir), recent_first=False)
        if image_name.endswith(".tif")
    ]
    # log.info("Sorted image paths: {}".format(image_paths))

    sorted_image_paths = [
        f for f in image_paths if "_masked" not in os.path.basename(f)
    ]  # remove already masked tiff files to avoid double-processing
    # log.info("Sorted image paths, no masks: {}".format(sorted_image_paths))

    # get timestamp of most recent image in the directory
    for i in range(len(sorted_image_paths)):
        timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[i]))
        log.info("Image number {} has time stamp {}".format(i + 1, timestamp))
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(
        composite_out_dir, "composite_{}.tif".format(last_timestamp)
    )
    clever_composite_images_with_mask(
        sorted_image_paths,
        composite_out_path,
        format,
        chunks=chunks,
        generate_date_image=generate_date_images,
        missing_data_value=missing_data_value,
        log=log
    )
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


def get_array(raster):
    """
    Returns a numpy array for the raster.

    Parameters
    ----------
    raster : gdal.Dataset
        A gdal.Dataset object

    Returns
    -------
    array : numpy array
        A numpy array of the raster

    """
    raster_array = raster.GetVirtualMemArray()
    return np.array(raster_array)


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
        log.info("{} exists, skipping.")
        return
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        old_clipped_image_path = os.path.join(td, "old.tif")
        new_clipped_image_path = os.path.join(td, "new.tif")
        clip_raster(old_image_path, aoi_path, old_clipped_image_path)
        clip_raster(new_image_path, aoi_path, new_clipped_image_path)
        stack_images(
            [old_clipped_image_path, new_clipped_image_path],
            out_image,
            geometry_mode="intersect",
        )


def clip_raster(
    raster_path, aoi_path, out_path, srs_id=4326, flip_x_y=False, dest_nodata=0
):
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
        The fill value for outside of the clipped area. Defaults to 0.
    """

    #TODO: Set values outside clip to 0 or to NaN - in irregular polygons
    # https://gis.stackexchange.com/questions/257257/how-to-use-gdal-warp-cutline-option
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        log.info("Clipping {} with {} to {}".format(raster_path, aoi_path, out_path))
        # log.info("Making temp dir {}".format(td))
        raster = gdal.Open(raster_path)
        in_gt = raster.GetGeoTransform()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(raster.GetProjection())
        intersection_path = os.path.join(td, "intersection")
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
        width_pix = int(np.floor(max_x_geo - min_x_geo) / in_gt[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo) / np.absolute(in_gt[5]))
        new_geotransform = (min_x_geo, in_gt[1], 0, max_y_geo, 0, in_gt[5])
        write_geometry(intersection, intersection_path, srs_id=srs.ExportToWkt())
        aoi = None
        clip_spec = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=intersection_path + r"/geometry.shp",
            cropToCutline=True,
            width=width_pix,
            height=height_pix,
            dstSRS=srs,
            dstNodata=dest_nodata,
        )
        out = gdal.Warp(out_path, raster, options=clip_spec)
        out.SetGeoTransform(new_geotransform)
        out = None


def clip_raster_to_intersection(
    raster_to_clip_path, extent_raster_path, out_raster_path, is_landsat=False
):
    """
    Clips one raster to the extent provided by the other raster, and saves the result at temp_file.
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

    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        temp_aoi_path = os.path.join(td, "temp_clip.shp")
        get_extent_as_shp(extent_raster_path, temp_aoi_path)
        ext_ras = gdal.Open(extent_raster_path)
        proj = osr.SpatialReference(wkt=ext_ras.GetProjection())
        srs_id = int(proj.GetAttrValue("AUTHORITY", 1))
        clip_raster(
            raster_to_clip_path,
            temp_aoi_path,
            out_raster_path,
            srs_id,
            flip_x_y=is_landsat,
        )


def create_new_image_from_polygon(
    polygon,
    out_path,
    x_res,
    y_res,
    bands,
    projection,
    format="GTiff",
    datatype=gdal.GDT_Int32,
    nodata=0,
):
    """
    Returns an empty image that covers the extent of the input polygon.

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
    bounds_x_min, bounds_x_max, bounds_y_min, bounds_y_max = polygon.GetEnvelope()
    if bounds_x_min >= bounds_x_max:
        bounds_x_min, bounds_x_max = bounds_x_max, bounds_x_min
    if bounds_y_min >= bounds_y_max:
        bounds_y_min, bounds_y_max = bounds_y_max, bounds_y_min
    final_width_pixels = int(np.abs(bounds_x_max - bounds_x_min) / x_res)
    final_height_pixels = int(np.abs(bounds_y_max - bounds_y_min) / y_res)
    driver = gdal.GetDriverByName(str(format))
    out_raster = driver.Create(
        out_path,
        xsize=final_width_pixels,
        ysize=final_height_pixels,
        bands=bands,
        eType=datatype,
    )
    out_raster.SetGeoTransform([bounds_x_min, x_res, 0, bounds_y_max, 0, y_res * -1])
    out_raster.SetProjection(projection)
    for band_index in range(1, bands):
        band = out_raster.GetRasterBand(band_index)
        band.SetNoDataValue(nodata)
        band = None
    arr = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    arr[...] = nodata
    arr = None
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
    image = gdal.Open(image_path, gdal.GA_ReadOnly)
    if image.RasterXSize == new_res and image.RasterYSize == new_res:
        log.info("Image already has {}m resolution: {}".format(new_res, image_path))
        image = None
        return
    image = None
    # log.info("Resampling to {}m resolution: {}".format(new_res, image_path))
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # Remember this is used for masks, so any averaging resample strat will cock things up.
        args = gdal.WarpOptions(xRes=new_res, yRes=new_res)
        temp_image = os.path.join(td, "temp_image.tif")
        gdal.Warp(temp_image, image_path, options=args)

        # Windows permissions.
        if sys.platform.startswith("win"):
            os.remove(image_path)
            shutil.copy(temp_image, image_path)
        else:
            shutil.move(temp_image, image_path)
    return


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

    if target_res % image_res != 0:
        log.warning(
            "Target and image resolutions are not divisible, grids will not align. Consider resampling."
        )

    # First, do we want to move the image left or right and up or down?
    x_offset = image_x % target_res
    y_offset = image_y % target_res

    if x_offset == 0 and y_offset == 0:
        log.info("Images are already aligned")
        return

    # If x is nearest to pixel line
    if -1 * (image_res / 2) <= x_offset <= image_res / 2:
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
    # By Qing
    os.system("gdaltindex " + out_shp_path + " " + in_ras_path)
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
    out_array[...] = (R - I) / (R + I)

    out_array[...] = np.where(out_array == -2147483648, 0, out_array)

    R = None
    I = None
    array = None
    out_array = None
    raster = None
    out_raster = None


def apply_band_function(
    in_path, function, bands, out_path, out_datatype=gdal.GDT_Int32
):
    """
    Applys an arbitrary band mathematics function to an image at in_path and saves the result at out_map.
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
    out_raster = create_matching_dataset(
        raster, out_path=out_path, datatype=out_datatype
    )
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
    return (r - i) / (r + i)


def apply_image_function(in_paths, out_path, function, out_datatype=gdal.GDT_Int32):
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

    out_raster = create_matching_dataset(
        rasters[0], out_path=out_path, datatype=out_datatype
    )
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


def raster_sum(inRstList, outFn, outFmt="GTiff"):
    """Creates a raster stack from a list of rasters. Adapted from Chris Gerard's
    book 'Geoprocessing with Python'. The output data type is the same as the input data type.

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
    log.info("Starting raster sum function.")

    # open 1st band to get info
    in_ds = gdal.Open(inRstList[0])
    in_band = in_ds.GetRasterBand(1)

    # Get raster shape
    rst_dim = (in_band.YSize, in_band.XSize)

    # initiate empty array
    empty_arr = np.empty(rst_dim, dtype=np.uint8)

    for i, rst in enumerate(inRstList):
        #TODO: Check that dimensions and shape of both arrays are the same in 
        #      the first loop.
        ds = gdal.Open(rst)
        bnd = ds.GetRasterBand(1)
        arr = bnd.ReadAsArray()
        empty_arr = empty_arr + arr

    # Create a 1 band GeoTiff with the same properties as the input raster
    driver = gdal.GetDriverByName(str(outFmt))
    out_ds = driver.Create(outFn, in_band.XSize, in_band.YSize, 1, in_band.DataType)
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

    log.info("Finished summing up of raster layers.")


def filter_by_class_map(
    image_path,
    class_map_path,
    out_map_path,
    classes_of_interest,
    out_resolution=10,
    invert=False,
):
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
    invert : bool, optional
        If present, invert mask (ie filter out classes_of_interest)

    Returns
    -------
    out_map_path : str
        The path to the new map

    """
    
    # TODO: Include nodata value
    log = logging.getLogger(__name__)
    log.info(
        "Filtering {} using classes{} from map {}".format(
            class_map_path, classes_of_interest, image_path
        )
    )
    # with TemporaryDirectory(dir=os.getcwd(dir=os.getcwd())) as td:
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        binary_mask_path = os.path.join(td, "binary_mask.tif")
        create_mask_from_class_map(
            class_map_path,
            binary_mask_path,
            classes_of_interest,
            out_resolution=out_resolution,
        )

        log.info("Mask created at {}, applying...".format(binary_mask_path))
        class_map = gdal.Open(binary_mask_path)
        class_array = class_map.GetVirtualMemArray()

        image_map = gdal.Open(image_path)
        image_array = image_map.GetVirtualMemArray()
        out_map = create_matching_dataset(
            image_map, out_map_path, bands=image_map.RasterCount
        )
        out_array = out_map.GetVirtualMemArray(eAccess=gdal.GA_Update)
        class_bounds = get_raster_bounds(class_map)
        image_bounds = get_raster_bounds(image_map)
        in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
            image_map, class_bounds
        )
        image_view = image_array[in_y_min:in_y_max, in_x_min:in_x_max]
        class_x_min, class_x_max, class_y_min, class_y_max = pixel_bounds_from_polygon(
            class_map, image_bounds
        )
        class_view = class_array[class_y_min:class_y_max, class_x_min:class_x_max]
        if invert:
            class_view = np.logical_not(class_view)
        filtered_array = apply_array_image_mask(image_view, class_view)

        np.copyto(out_array, filtered_array)
        out_array = None
        out_map = None
        class_array = None
        class_map = None

    log.info("Map filtered")
    return out_map_path


def open_dataset_from_safe(safe_file_path, band, resolution="10m"):
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
    image_glob = r"GRANULE/*/IMG_DATA/R{}/*_{}_{}.jp2".format(
        resolution, band, resolution
    )
    # edited by hb91
    # image_glob = r"GRANULE/*/IMG_DATA/*_{}.jp2".format(band)
    fp_glob = os.path.join(safe_file_path, image_glob)
    image_file_path = glob.glob(fp_glob)
    out = gdal.Open(image_file_path[0])
    return out


def preprocess_sen2_images(
    l2_dir,
    out_dir,
    l1_dir,
    cloud_threshold=60,
    buffer_size=0,
    epsg=None,
    bands=("B02", "B03", "B04", "B08"),
    out_resolution=10,
):
    """
    For every .SAFE folder in l2_dir and L1_dir, merges the rasters of bands 2,3,4 and 8 into a single geotiff file,
    creates a cloudmask from the combined fmask and sen2cor cloudmasks and reprojects to a given EPSG if provided.

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
    safe_file_path_list = [
        os.path.join(l2_dir, safe_file_path)
        for safe_file_path in os.listdir(l2_dir)
        if safe_file_path.endswith(".SAFE")
    ]
    for l2_safe_file in safe_file_path_list:
        with TemporaryDirectory(dir=os.path.expanduser('~')) as temp_dir:
            log.info("----------------------------------------------------")
            log.info(
                "Merging the selected 10m band files in directory {}".format(
                    l2_safe_file
                )
            )
            # log.info("Making temp dir {}".format(temp_dir))
            temp_file = (
                os.path.join(temp_dir, get_sen_2_granule_id(l2_safe_file)) + ".tif"
            )
            out_path = os.path.join(out_dir, os.path.basename(temp_file))
            log.info("Output file containing all bands: {}".format(out_path))

            if os.path.exists(out_path):
                log.info("Output file already exists. Skipping the band merging step.")
            else:
                stack_sentinel_2_bands(
                    l2_safe_file, temp_file, bands=bands, out_resolution=out_resolution
                )

                log.info("Creating cloud mask for {}".format(temp_file))
                l1_safe_file = get_l1_safe_file(l2_safe_file, l1_dir)
                if l1_safe_file:
                    mask_path = get_mask_path(temp_file)
                    create_mask_from_sen2cor_and_fmask(
                        l1_safe_file, l2_safe_file, mask_path, buffer_size=buffer_size
                    )
                    out_mask_path = os.path.join(out_dir, os.path.basename(mask_path))
                else:
                    log.error(
                        "No L1 data found for {}; cannot generate cloudmask.".format(
                            l2_safe_file
                        )
                    )

                if epsg:
                    log.info("Reprojecting images to EPSG code {}".format(epsg))
                    proj = osr.SpatialReference()
                    proj.ImportFromEPSG(epsg)
                    wkt = proj.ExportToWkt()
                    reproject_image(temp_file, out_path, wkt)
                    if l1_safe_file:
                        reproject_image(mask_path, out_mask_path, wkt)
                        resample_image_in_place(out_mask_path, out_resolution)
                else:
                    # log.info("Moving images to {}".format(out_dir))
                    shutil.move(temp_file, out_path)
                    if l1_safe_file:
                        shutil.move(mask_path, out_mask_path)
                        resample_image_in_place(out_mask_path, out_resolution)


def apply_scl_cloud_mask(
    l2_dir,
    out_dir,
    scl_classes,
    buffer_size=0,
    bands=["B02", "B03", "B04", "B08"],
    out_resolution=10,
    haze=None,
    epsg=None,
    skip_existing=False,
):
    """
    For every .SAFE folder in l2_dir, creates a cloud-masked raster band for each selected band
    based on the SCL layer. Applies a rough haze correction based on thresholding the blue band (optional).

    Parameters
    ----------
    l2_dir : str
        The directory containing a set of L2 .SAFE folders to preprocess
    out_dir : str
        The directory to store the preprocessed files
    scl_classes : list of int
        values of classes to be masked out
    buffer_size : int, optional
        The buffer to apply to the sen2cor mask - defaults to 0
    bands : list of str, optional
        List of names of bands to include in the final rasters. Defaults to ("B02", "B03", "B04", "B08")
    out_resolution : number, optional
        Resolution to resample every image to - units are defined by the image projection. Default is 10.
    haze : number, optional
        Threshold if a haze filter is to be applied. If specified, all pixel values where "B02" > haze will be masked out.
        Defaults to None. If set, recommended thresholds range from 325 to 600 but can vary by scene conditions.
    epsg : int
        EPSG code of the map projection / CRS if output rasters shall be reprojected (warped)
    skip_existing : boolean
        If True, skip cloud masking if a file already exists. If False, overwrite it.

    """
    safe_file_path_list = [
        os.path.join(l2_dir, safe_file_path)
        for safe_file_path in os.listdir(l2_dir)
        if safe_file_path.endswith(".SAFE")
    ]
    for l2_safe_file in safe_file_path_list:
        log.info("  L2A raster file: {}".format(l2_safe_file))
        f = get_sen_2_granule_id(l2_safe_file)
        pattern = (
            f.split("_")[0]
            + "_"
            + f.split("_")[1]
            + "_"
            + f.split("_")[2]
            + "_"
            + f.split("_")[3]
            + "_"
            + f.split("_")[4]
            + "_"
            + f.split("_")[5]
        )
        # log.info("  Granule ID  : {}".format(f))
        # log.info("  File pattern: {}".format(pattern))
        # Find existing matching files in the output directory
        df = get_raster_paths([out_dir], filepatterns=[pattern], dirpattern="")
        for i in range(len(df)):
            if df[pattern][i] != "" and skip_existing:
                log.info("  Skipping band merging for: {}".format(f))
                log.info("  Found stacked file: {}".format(df[pattern][i][0]))
            else:
                out_path = os.path.join(
                    out_dir, get_sen_2_granule_id(l2_safe_file) + ".tif"
                )
                with TemporaryDirectory(dir=os.path.expanduser('~')) as temp_dir:
                    stacked_file = os.path.join(
                        temp_dir, get_sen_2_granule_id(l2_safe_file) + "_stacked.tif"
                    )
                    masked_file = os.path.join(
                        temp_dir,
                        get_sen_2_granule_id(l2_safe_file) + "_stacked_masked.tif",
                    )
                    stack_sentinel_2_bands(
                        l2_safe_file,
                        stacked_file,
                        bands=bands,
                        out_resolution=out_resolution,
                    )
                    mask_path = get_mask_path(stacked_file)
                    create_mask_from_scl_layer(
                        l2_safe_file, mask_path, scl_classes, buffer_size=buffer_size
                    )
                    apply_mask_to_image(mask_path, stacked_file, masked_file)
                    if haze is not None:
                        log.info(
                            "Applying haze mask to pixels where B02 > {}. Assumes B02 is band 1 in the stacked image.".format(
                                haze
                            )
                        )
                        haze_masked_file = os.path.join(
                            temp_dir,
                            get_sen_2_granule_id(l2_safe_file)
                            + "_stacked_masked_haze.tif",
                        )
                        haze_mask_path = get_mask_path(haze_masked_file)
                        create_mask_from_band(
                            masked_file,
                            haze_mask_path,
                            band=1,
                            threshold=haze,
                            relation="smaller",
                            buffer_size=buffer_size,
                        )
                        apply_mask_to_image(
                            haze_mask_path, masked_file, haze_masked_file
                        )
                        shutil.move(haze_masked_file, masked_file)
                    resample_image_in_place(masked_file, out_resolution)
                    if epsg is not None:
                        log.info(
                            "Reprojecting stacked and masked image to EPSG code {}".format(
                                epsg
                            )
                        )
                        
                        proj = osr.SpatialReference()
                        proj.ImportFromEPSG(epsg)
                        wkt = proj.ExportToWkt()
                        log.info(f'epsg: {epsg}, wkt: {wkt}')

                        reproject_image(masked_file, out_path, wkt)
                    else:
                        shutil.move(masked_file, out_path)
    return


# Added I.R. 20220607 START
def apply_processing_baseline_0400_offset_correction_to_tiff_file_directory(
    in_tif_directory: str,
    out_tif_directory: str,
    bands_to_offset_labels=("B02", "B03", "B04", "B08"),
    bands_to_offset_index=[0, 1, 2, 3],
    BOA_ADD_OFFSET=-1000,
    backup_flag=False,
    log = logging.getLogger(__name__)
):
    """
    Offsets data within selected bands from a directory of stacked raster files
    Overwrites original file - option to save a backup copy.
    
    WARNING: apply_processing_baseline_0400_offset_correction_to_tiff_file_directory()
        will be depracated.
        Use apply_processing_baseline_offset_correction_to_tiff_file_directory() instead

    Parameters
    ----------
    in_tif_directory : str
        Path to the input (and output) directory of tif raster files
    out_tif_directory : str
        Path to the output directory of tif raster files
    bands_to_offset_labels : list[str]
        List of bands to offset
    bands_to_offset_index : list[int]
        List of indices of bands to offset within the tif image
    BOA_ADD_OFFSET : int
        Required offset per band (from xml information within L2A SAFE file directory)
    backup_flag : bool, optional
        If True leaves unoffset images with .backup extension in tif_directory
    log : logger
        Directs to the log file

    Returns
    -------
    out_tif_directory : str
        The path to the output directory

    """

    def get_processing_baseline(safe_path):
        return safe_path[28:32]

    def set_processing_baseline(safe_path, new_baseline):
        new_safe_path = safe_path[:28] + new_baseline + safe_path[32:]
        return new_safe_path

    # Check out_tif_directory exists and create it if not
    if not os.path.exists(out_tif_directory):
        os.mkdir(out_tif_directory)

    #TODO: 
    #  Force generated dtype to uint16 to save time and storage? 
    #    Compatible with classifier?
    #  Generate bands_to_offset_index from comparison of bands_to_offset labels
    #    in band.description
    #  Read individual BOA_ADD_OFFSET value for each band from xml information
    #    in SAFE file root
    #  Work out why offset files are larger (2GB from ~1GB)
    log.info(f"Radiometric offset correction if processing_baseline > 0400 in" + \
             " directory: {in_tif_directory}")
    log.info("NOTE: Processing baseline in the file names will be set to \"A0400\"" + \
             " if correction has been applied.")
    log.warning("WARNING: apply_processing_baseline_0400_offset_correction_to_tiff_file_directory() " + \
             "will be depracated. " + \
             "Use apply_processing_baseline_offset_correction_to_tiff_file_directory() instead.")

    image_files = [
        f
        for f in os.listdir(in_tif_directory)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]
    for f in image_files:
        log.info(f"File: {f}, Baseline: {get_processing_baseline(f)}")
        if get_processing_baseline(f) == "A400":
            log.info(f"Offset already applied - file marked as: {get_processing_baseline(f)}")
        if get_processing_baseline(f) == "0400":
            in_raster_path = os.path.join(in_tif_directory, f)
            log.info(f"Offsetting file: {f}")
            log.info(f'in_raster_path: {in_raster_path}')
            # Define temporary file destination for output
            out_temporary_raster_path = os.path.join(
                out_tif_directory,
                os.path.basename(f).split(".")[0] + "_offset_temp.tif",
            )
            log.info(f"out_temporary_raster_path: {out_temporary_raster_path}")
            # Open dataset
            in_raster_ds = gdal.Open(in_raster_path, gdal.GA_Update)
            raster_band_count = in_raster_ds.RasterCount
            in_raster_array = in_raster_ds.GetVirtualMemArray()
            out_temporary_raster_ds = (
                pyeo.raster_manipulation.create_matching_dataset(
                    in_raster_ds, out_temporary_raster_path, bands=raster_band_count
                )
            )
            out_temporary_raster_array = out_temporary_raster_ds.GetVirtualMemArray(
                eAccess=gdal.GA_Update
            )

            log.info(f"in_raster_array dtype: {in_raster_array.dtype}")
            dtype_max = 10000  # np.iinfo(in_raster_array.dtype).max # upper bound for range clipping - should be > any likely pixel value
            log.info(f"in_raster_array dtype_max used: {dtype_max}")

            # Simple offset of all image bands
            out_temporary_raster_array[...] = (
                np.clip(
                    in_raster_array[bands_to_offset_index, :, :],
                    (-1 * BOA_ADD_OFFSET),
                    dtype_max,
                )
                + BOA_ADD_OFFSET
            )

            # Untested: Improvement to offset just selected bands by label 
            #   - for band specific offsetting if required
            # out_raster_array[...] = in_raster_array[...]  # Copy over all data
            # for band_index in raster_band_count:
            #     band_in = in_raster_ds.GetRasterBand(band_index+1)
            #     band_out = out_raster_ds.GetRasterBand(band_index+1)
            #     band_out.SetDescription(band_in.GetDesciption())
            #     if (band_in.GetDesciption() in bands_to_offset_labels):
            #         out_raster_array[band_index, :, :] = np.clip(in_raster_array[band_index, :,:] , (-1 * BOA_ADD_OFFSET), dtype_max) + BOA_ADD_OFFSET

            # Deallocate to force write of generated file to disk by OS
            out_temporary_raster_array = None
            in_raster_array = None
            out_temporary_raster_ds = None
            in_raster_ds = None

            # Backup original .tif file to .backup file (subsequent algorithm 
            #   stages should filter for only .tif or .tiff)
            if backup_flag == True:
                shutil.move(in_raster_path, in_raster_path.split(".")[0] + ".backup")
            out_raster_path = os.path.join(out_tif_directory, f)
            # If in_tif_directory == out_tif_directory then overwrites original 
            #   .tif file with offset file so that next stage of ForestMind 
            #   algorithm can run
            log.info(f"Moving {out_temporary_raster_path} to {out_raster_path}")
            shutil.move(out_temporary_raster_path, out_raster_path)
            # Rename file with processing baseline code modified from 0400 to 
            #   A400 to avoid multiple runs of pipeline leading to multiple 
            #   offsets being applied
            out_raster_path_rename = os.path.join(
                out_tif_directory, set_processing_baseline(f, "A400")
            )
            log.info(f"Moving {out_raster_path} to {out_raster_path_rename}")
            shutil.move(out_raster_path, out_raster_path_rename)

    log.info("Offsetting Finished")
    return out_tif_directory
# Added I.R. 20220607 END


# Added I.R. 20230312 START
def apply_processing_baseline_offset_correction_to_tiff_file_directory(
    in_tif_directory,
    out_tif_directory,
    bands_to_offset_labels=("B02", "B03", "B04", "B08"),
    bands_to_offset_index=[0, 1, 2, 3],
    BOA_ADD_OFFSET=-1000,
    backup_flag=False,
    log = logging.getLogger(__name__)
):
    """
    Offsets data within selected bands from a directory of stacked raster files
    Overwrites original file - option to save a backup copy.

    Parameters
    ----------
    in_tif_directory : str
        Path to the input (and output) directory of tif raster files
    out_tif_directory : str
        Path to the output directory of tif raster files
    bands_to_offset_labels : list of string
        List of bands to offset
    bands_to_offset_index : list of int
        List of indices of bands to offset within the tif image
    BOA_ADD_OFFSET : int
        Required offset per band (from xml information within L2A SAFE file directory)
    backup_flag : True/False
        If True leaves unoffset images with .backup extension in tif_directory
    log : logger object
        Directs logging output

    Returns
    -------
    out_tif_directory : str
        The path to the output directory

    """

    def get_processing_baseline(safe_path):
        return safe_path[28:32]

    def set_processing_baseline(safe_path, new_baseline):
        new_safe_path = safe_path[:28] + new_baseline + safe_path[32:]
        return new_safe_path

    # Check out_tif_directory exists and create it if not
    if not os.path.exists(out_tif_directory):
        os.mkdir(out_tif_directory)

    log.info("Radiometric offset correction if processing_baseline > 0400 in" +
             f" directory: {in_tif_directory}")
    log.info("NOTE: Processing baseline in the file names will be set to e.g. 'A400'" +
             " if correction has been applied.")
    log.info("NOTE: Processing baseline 9999 for some L2A products originates from "+
             " ESA Cloud.Ferro and not from the Copernicus Data Space Ecosystem."+
             " These will be skipped.")
    log.info(" See https://documentation.dataspace.copernicus.eu/Data/Sentinel2.html")

    image_files = [
        f
        for f in os.listdir(in_tif_directory)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    # TODO: Force generated dtype to uint16 to save time and storage? Compatible with classifier?
    # Generate bands_to_offset_index from comparison of bands_to_offset labels in band.description
    # Read individual BOA_ADD_OFFSET value for each band from xml information in SAFE file root
    # Work out why offset files are larger (2GB from ~1GB)

    for f in image_files:
        log.info(f"File: {f}, Baseline: {get_processing_baseline(f)}")
        if get_processing_baseline(f)[0] == "A":
            log.info(f"Offset already applied - file marked as: {get_processing_baseline(f)}")
        if get_processing_baseline(f)[0] != "A" and \
            (int(get_processing_baseline(f)) != 9999 and \
            int(get_processing_baseline(f)) >= 400): # in ["0400", "0509"]:
            in_raster_path = os.path.join(in_tif_directory, f)
            log.info(f"Offsetting file: {f}")
            log.info(f'Full file path: {in_raster_path}')
            # Define temporary file destination for output
            out_temporary_raster_path = os.path.join(
                out_tif_directory,
                os.path.basename(f).split(".")[0] + "_offset_temp.tif",
            )
            log.info(f"out_temporary_raster_path: {out_temporary_raster_path}")
            # Open dataset
            in_raster_ds = gdal.Open(in_raster_path, gdal.GA_Update)
            raster_band_count = in_raster_ds.RasterCount
            in_raster_array = in_raster_ds.GetVirtualMemArray()
            out_temporary_raster_ds = (
                pyeo.raster_manipulation.create_matching_dataset(
                    in_raster_ds, out_temporary_raster_path, bands=raster_band_count
                )
            )
            out_temporary_raster_array = out_temporary_raster_ds.GetVirtualMemArray(
                eAccess=gdal.GA_Update
            )

            log.info(f"in_raster_array dtype: {in_raster_array.dtype}")
            dtype_max = 10000  # np.iinfo(in_raster_array.dtype).max # upper bound for range clipping - should be > any likely pixel value
            log.info(f"in_raster_array clipped to range of min: {-1 * BOA_ADD_OFFSET} and max: {dtype_max} then offset by: {BOA_ADD_OFFSET}")

            # Simple offset of all image bands
            out_temporary_raster_array[...] = (
                np.clip(
                    in_raster_array[:, :, :],
                    (-1 * BOA_ADD_OFFSET),
                    dtype_max,
                )
                + BOA_ADD_OFFSET
            )

            # Untested: Improvement to offset just selected bands by label 
            #   - for band specific offsetting if required
            # out_raster_array[...] = in_raster_array[...]  # Copy over all data
            # for band_index in raster_band_count:
            #     band_in = in_raster_ds.GetRasterBand(band_index+1)
            #     band_out = out_raster_ds.GetRasterBand(band_index+1)
            #     band_out.SetDescription(band_in.GetDesciption())
            #     if (band_in.GetDesciption() in bands_to_offset_labels):
            #         out_raster_array[band_index, :, :] = np.clip(in_raster_array[band_index, :,:] , (-1 * BOA_ADD_OFFSET), dtype_max) + BOA_ADD_OFFSET

            # Deallocate to force write of generated file to disk by OS
            out_temporary_raster_array = None
            in_raster_array = None
            out_temporary_raster_ds = None
            in_raster_ds = None

            # Backup original .tif file to .backup file (subsequent algorithm 
            #   stages should filter for only .tif or .tiff)
            if backup_flag == True:
                shutil.move(in_raster_path, in_raster_path.split(".")[0] + ".backup")
            out_raster_path = os.path.join(out_tif_directory, f)
            # If in_tif_directory == out_tif_directory then it overwrites the 
            #   original .tif file with offset file so that next stage of the
            #   algorithm can run
            log.info(f"Moving {out_temporary_raster_path} to {out_raster_path}")
            shutil.move(out_temporary_raster_path, out_raster_path)
            # Rename file with processing baseline code modified from 0XXX to 
            #   AXXX to avoid multiple runs of pipeline leading to multiple offsets being applied
            out_raster_path_rename = os.path.join(
                out_tif_directory,
                set_processing_baseline(f, "A" + get_processing_baseline(f)[1:]),
            )
            log.info(f"Moving {out_raster_path} to {out_raster_path_rename}")
            shutil.move(out_raster_path, out_raster_path_rename)

    log.info("Radiometric offsetting of processing baselines > 0400 finished.")
    return out_tif_directory


# Added I.R. 20230312 END


def preprocess_landsat_images(
    image_dir, out_image_path, new_projection=None, bands_to_stack=("B2", "B3", "B4")
):
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
    band_path_list = []  # This still feels like a Python antipattern, but hey.
    for band_id in bands_to_stack:
        band_glob = os.path.join(image_dir, "LC08_*_{}.TIF".format(band_id))
        band_path_list.append(glob.glob(band_glob)[0])

    n_bands = len(band_path_list)
    driver = gdal.GetDriverByName(str("GTiff"))
    first_ls_raster = gdal.Open(band_path_list[0])
    first_ls_array = first_ls_raster.GetVirtualMemArray()
    out_image = driver.Create(
        out_image_path,
        xsize=first_ls_array.shape[1],
        ysize=first_ls_array.shape[0],
        bands=n_bands,
        eType=first_ls_raster.GetRasterBand(1).DataType,
    )
    out_image.SetGeoTransform(first_ls_raster.GetGeoTransform())
    out_image.SetProjection(first_ls_raster.GetProjection())
    out_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
    first_ls_array = None
    first_ls_raster = None
    for ii, ls_raster_path in enumerate(band_path_list):
        # log.info("Stacking {} to raster layer {}".format(ls_raster_path, ii))
        ls_raster = gdal.Open(ls_raster_path)
        ls_array = ls_raster.GetVirtualMemArray()
        out_array[ii, ...] = ls_array[...]
        ls_array = None
        ls_raster = None
    out_array = None
    out_image = None
    if new_projection:
        with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
            # log.info("Making temp dir {}".format(td))
            # log.info("Reprojecting to {}")
            temp_path = os.path.join(td, "reproj_temp.tif")
            # log.info("Temporary image path at {}".format(temp_path))
            reproject_image(
                out_image_path, temp_path, new_projection, do_post_resample=False
            )
            os.remove(out_image_path)
            os.rename(temp_path, out_image_path)
            resample_image_in_place(out_image_path, 30)
    # log.info("Stacked image at {}".format(out_image_path))


def stack_sentinel_2_bands(
    safe_dir, out_image_path, bands=("B02", "B03", "B04", "B08"), out_resolution=10
):
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

    #for band_path in band_paths:
    #    print(f'Image Resolution: {band_path}')


    # Move every image NOT in the requested resolution to resample_dir and resample
    with TemporaryDirectory(dir=os.path.expanduser('~')) as resample_dir:
        # log.info("Making temp dir {}".format(resample_dir))
        new_band_paths = []
        for band_path in band_paths:
            if get_image_resolution(band_path) != out_resolution:
                resample_path = os.path.join(resample_dir, os.path.basename(band_path))
                shutil.copy(band_path, resample_path)
                resample_image_in_place(resample_path, out_resolution)
                new_band_paths.append(resample_path)
            else:
                new_band_paths.append(band_path)

        stack_images(new_band_paths, out_image_path, geometry_mode="intersect")

    # # I.R. 20220529 START
    # # if processing baseline = 0400 move every images offset_dir and offset by -1000 to compensate for introduction of negative radiometric offset
    # ## https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming
    # ## Value of offset is currently -1000 for all bands but should be read from <BOA_ADD_OFFSET_VALUES_LIST> for each bandv in xml file at base of SAFE directories
    # BOA_ADD_OFFSET = -1000
    # offset_band_paths = []
    #
    # processing_baseline = os.path.basename(safe_dir)[28:32]
    # # log.info(f'I.R.: safe_dir: {safe_dir}, basename: {os.path.basename(safe_dir)}, processing_baseline: {processing_baseline}')
    # print(f'I.R.: safe_dir: {safe_dir}, basename: {os.path.basename(safe_dir)}, processing_baseline: {processing_baseline}')
    # with TemporaryDirectory(dir=os.getcwd()) as offset_dir:
    #     offset_dir = os.getcwd()  # Temporary override of destination directory for intermediate files fro debugging purposes
    #     if processing_baseline == '0400':
    #         # log.info(f'I.R.: Offsetting to compensate for negative radiometric offset for processing baseline: {processing_baseline}')
    #         print(f'I.R.: Offsetting to compensate for negative radiometric offset for processing baseline: {processing_baseline}')
    #         for band_path in new_band_paths:
    #             offset_path = os.path.join(offset_dir, os.path.basename(band_path))
    #             # shutil.copy(band_path, offset_path)
    #             # TODO TEST & DEBUG: Insert GDAL operation to open band image at offset_path for update and add an offset of -1000 here
    #             image_band_ds = gdal.Open(offset_path, gdal.GA_Update)
    #             # image_array = image_band.GetVirtualMemArray(eAccess=gdal.GA_Update)
    #             # image_ds = gdal.Open(offset_path, gdal.GF_Write)
    #             # image_band = image_ds.GetRasterBand(1)
    #             image_band_array = image_band_ds.GetVirtualMemArray(eAccess=gdal.GA_ReadOnly) # (eAccess=gdal.GA_ReadOnly)  #(eAccess=gdal.GF_Write)
    #             # image_array = image_band.ReadAsArray()
    #             # log.info(f'I.R.: image_array: {type(image_array)}, {image_array.shape}, , {image_array.dtype}')
    #             print(f'I.R.: image_band_array: {type(image_band_array)}, shape: {image_band_array.shape}, dtype: {image_band_array.dtype}, min: {np.min(image_band_array)}, max: {np.max(image_band_array)}')
    #             print(image_band_array[0][0:10])
    #             image_band_array_offset = np.clip(image_band_array , (-1 * BOA_ADD_OFFSET), np.iinfo(image_band_array.dtype).max)
    #             image_band_array_offset = image_band_array_offset + BOA_ADD_OFFSET
    #             # image_array = image_array.astype(uint16)
    #             # image_array = image_array + 1
    #             print(f'I.R.: image_band_array_offset: {type(image_band_array_offset)}, shape: {image_band_array_offset.shape}, dtype: {image_band_array_offset.dtype}, min: {np.min(image_band_array_offset)}, max: {np.max(image_band_array_offset)}')
    #             print(image_band_array_offset[0][0:10])
    #
    #
    #             offset_ds = create_matching_dataset(image_band_ds, offset_path)
    #             offset_array = offset_ds.GetVirtualMemArray(eAccess=gdal.GA_Update)
    #             np.copyto(offset_array, image_band_array_offset.astype(np.uint16))
    #
    #             # np.copyto(image_band_array, image_band_array_offset.astype(np.uint16))
    #
    #             # offset_array = image_band_array
    #             # out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    #
    #             # image_band.WriteArray(image_array)
    #             # deallocate the dataset to force it to be flushed (updated) to disk
    #             # image_array = None
    #             # image_band = None
    #             offest_array = None
    #             offset_ds = None
    #             image_band_array = None
    #             image_band_ds = None
    #
    #             offset_band_paths.append(offset_path)
    #     else:
    #         offset_band_paths = new_band_paths
    #
    #     # log.info(f'I.R.: resample_dir: {resample_dir}')
    #     print(f'I.R.: resample_dir: {resample_dir}')
    #     # log.info(f'I.R.: offset_dir: {offset_dir}')
    #     print(f'I.R.: offset_dir: {offset_dir}')
    #     for i in offset_band_paths:
    #         print(f'I.R. offset_band_path: {i}')
    #
    #     # stack_images(new_band_paths, out_image_path, geometry_mode="intersect")
    #     stack_images(offset_band_paths, out_image_path, geometry_mode="intersect")
    #
    # # I.R. 20220529 END

    # Saving band labels in images
    new_raster = gdal.Open(out_image_path)
    for band_index, band_label in enumerate(bands):
        band = new_raster.GetRasterBand(band_index + 1)
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
        res_string = ""


    if get_safe_product_type(safe_dir) == "MSIL1C":
        filepattern = "_" + band + ".jp2"
        band_paths = get_filenames(safe_dir, filepattern, "IMG_DATA")
        # band_glob = "GRANULE/*/IMG_DATA/*_{}*.*".format(band)
        # band_glob = os.path.join(safe_dir, band_glob)
        # band_paths = glob.glob(band_glob)
        if not band_paths:
            raise FileNotFoundError(
                "Band {} not found for safe file {}".format(band, safe_dir)
            )
        band_path = band_paths[0]

    else:
        if res_string in [
            "10m",
            "20m",
            "60m",
        ]:  # If resolution is given, then find the band of that resolution
            filepattern = "_" + band + "_" + res_string + ".jp2"
            # band_glob = "GRANULE/*/IMG_DATA/R{}/*_{}_*.*".format(res_string, band)
            # band_glob = os.path.join(safe_dir, band_glob)
            try:
                band_paths = get_filenames(safe_dir, filepattern, res_string)
                band_path = band_paths[0]
                # band_path = glob.glob(band_glob)[0]
            except IndexError as e:
                log.warning(
                    "Band {} not found of specified resolution, searching in other available resolutions: {}".format(
                        band, e
                    )
                )

        if (
            res_string is None or "band_path" not in locals()
        ):  # Else use the highest resolution available for that band
            band_paths = []
            for res in ["10m", "20m", "60m"]:
                filepattern = "_" + band + "_" + res + ".jp2"

                # I.R. 20230527 Pattern for directory matching corrected
                #band_paths.append(get_filenames(safe_dir, filepattern, res_string))
                band_paths.append(get_filenames(safe_dir, filepattern, res))

                # band_glob = "GRANULE/*/IMG_DATA/R*/*_{}_*.*".format(band)
                # band_glob = os.path.join(safe_dir, band_glob)
                # band_paths = glob.glob(band_glob)

            # I.R. 20230527 Empty lists removed
            print(f'{band_paths=}')     
            band_paths_sorted = sorted(band_paths)
            print(f'{band_paths_sorted=}')
            band_paths_sorted_noempty = [x for x in band_paths_sorted if x != []]     
            print(f'{band_paths_sorted_noempty=}')

            # I.R. 20230527 Corrected to return a path string (as assumed above and by rest of code base). Previously returned a single element list containing the string.
            band_path = band_paths_sorted_noempty[0][0]  # Sorting alphabetically gives the highest resolution first
            print(f'{band_path=}')     
            if band_path == []:
                raise FileNotFoundError(
                    "Band {} not found for safe file {}".format(band, safe_dir)
                )
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
    image = gdal.Open(image_path)
    if image is None:
        raise FileNotFoundError("Image not found at {}".format(image_path))
    gt = image.GetGeoTransform()
    if gt[1] != gt[5] * -1:
        raise NonSquarePixelException(
            "Image at {} has non-square pixels - this is currently not implemented in pyeo"
        )
    return gt[1]


def stack_old_and_new_images(
    old_image_path, new_image_path, out_dir, create_combined_mask=True
):
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
    if tile_old == tile_new:
        log.info("Stacking {} and".format(old_image_path))
        log.info("         {}".format(new_image_path))
        old_timestamp = get_sen_2_image_timestamp(os.path.basename(old_image_path))
        new_timestamp = get_sen_2_image_timestamp(os.path.basename(new_image_path))
        out_path = os.path.join(
            out_dir, tile_new + "_" + old_timestamp + "_" + new_timestamp
        )
        log.info("Output stacked file: {}".format(out_path + ".tif"))
        stack_images([old_image_path, new_image_path], out_path + ".tif")
        if create_combined_mask:
            out_mask_path = out_path + ".msk"
            old_mask_path = get_mask_path(old_image_path)
            new_mask_path = get_mask_path(new_image_path)
            try:
                combine_masks(
                    [old_mask_path, new_mask_path],
                    out_mask_path,
                    combination_func="and",
                    geometry_func="intersect",
                )
            except FileNotFoundError as e:
                log.error(
                    "Mask not found for either {} or {}: {}".format(
                        old_image_path, new_image_path, e
                    )
                )
        return out_path + ".tif"
    else:
        log.error("Tiles  of the two images do not match. Aborted.")


def apply_sen2cor(
    image_path,
    sen2cor_path,
    delete_unprocessed_image=False,
    log=logging.getLogger(__name__),
):
    """
    Applies sen2cor to the SAFE file at image_path. Returns the path to the new product.

    Parameters
    ----------
    image_path : str
        Path to the L1C Sentinel 2 .SAFE file directory
    sen2cor_path : str
        Path to the l2a_process script (Linux) or l2a_process.exe (Windows)
    delete_unprocessed_image : bool, optional
        If True, delete the unprocessed image after processing is done. Defaults to False.
	log : logger object
		Will be used for output logging

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
    # gipp_path = os.path.join(os.path.dirname(__file__), "L2A_GIPP.xml")
    # out_dir = os.path.dirname(image_path)
    now_time = (
        datetime.datetime.now()
    )  # I can't think of a better way of getting the new outpath from sen2cor
    timestamp = now_time.strftime(r"%Y%m%dT%H%M%S")
    version = get_sen2cor_version(sen2cor_path)
    out_path = build_sen2cor_output_path(
        image_path, 
        timestamp, 
        version
        )

    # The application of sen2cor with the option --GIP_L2A caused an unspecified metadata error in the xml file.
    # Removing it resolves this problem.

    # I.R. 20220509
    log.info("Calling sen2cor:")
    log.info(sen2cor_path + " " + image_path + " --output_dir " + out_path)
    sen2cor_proc = subprocess.Popen(
        [sen2cor_path, image_path, "--output_dir", out_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    while True:
        nextline = sen2cor_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == "" and sen2cor_proc.poll() is not None:
            break    
        if "CRITICAL" in nextline:
            # log.error(nextline)
            raise subprocess.CalledProcessError(-1, "L2A_Process")

    log.info("sen2cor processing finished for {}".format(image_path))
    log.info("Checking for presence of band raster files in {}".format(out_path))
    if not check_for_invalid_l2_data(out_path):
        log.error("10m imagery not present in {}".format(out_path))
        raise BadS2Exception
    if delete_unprocessed_image:
        log.info("Removing {}".format(image_path))
        shutil.rmtree(image_path)
    return out_path


def build_sen2cor_output_path(image_path, timestamp, version):
    """
    Creates a sen2cor output path dependent on the version of sen2cor

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
        The path of the finished sen2cor L2A SAFE file directory

    """

    # Accounting for sen2cors ever-shifting filename format
    if version >= "2.08.00":
        out_path = image_path.replace("MSIL1C", "MSIL2A")
        # baseline = get_sen_2_baseline(image_path)
        # out_path = out_path.replace(baseline, "N9999")

        # get the first components of the SAFE file name up to the tile ID part
        out_path = "_".join(out_path.split("_"))[:-1] + "_" + timestamp + ".SAFE"
    else:
        out_path = image_path.replace("MSIL1C", "MSIL2A")

    if sys.platform.startswith("win"):
		# I.R. Modification to use home directory to store temporary sen2cor output 
		# - required to avoid errors due to path length exceeding maximum allowed under Windows
		# - note sen2cor fails if given a relative path
		# H.B. Modification to not use home directory to store temporary sen2cor output 
		# - required to avoid errors due to invalid cross-device link in path on the HPC
        user_home_path = os.path.expanduser('~')
        out_path = os.path.join(user_home_path, os.path.basename(out_path))
    #else:
    #    out_path = os.path.dirname(out_path)

    return out_path


def get_sen2cor_version(sen2cor_path):
    """
    Gets the version number of sen2cor from the help string.

    Parameters
    ----------
    sen2cor_path : str
        Path to the sen2cor executable

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
        version_regex = r"Sen2Cor (\d+.\d+.\d+)"
        match = re.search(version_regex, help_string)
        if match:
            return match.group(1)
        else:
            raise FileNotFoundError(
                "Version information not found; please check your sen2cor path."
            )


def atmospheric_correction(
    in_directory,
    out_directory,
    sen2cor_path,
    delete_unprocessed_image=False,
    log=logging.getLogger(__name__),
):
    """
    Applies Sen2cor atmospheric correction to each L1C image in in_directory

    Parameters
    ----------
    in_directory : str
        Path to the directory containing the L1C SAFE image directory
    out_directory : str
        Path to the directory that will contain the new L2A SAFE image directory
    sen2cor_path : str
        Path to the l2a_process script (Linux) or l2a_process.exe (Windows)
    delete_unprocessed_image : bool, optional
        If True, delete the unprocessed image after processing is done. 
        Defaults to False.
    log : optional
        if a logger object is provided, `atmospheric_correction` will pass 
            statements to that logger, otherwise the default namespace logger 
            is used.
    """

    images = [
        image for image in os.listdir(in_directory) \
            if image.startswith("MSIL1C", 4) and \
            os.path.isdir(image)
    ]
    # Opportunity for multithreading here
    for image in images:
        image_path = os.path.join(in_directory, image)
        # update the product discriminator part of the output file name
        # see https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        image_timestamp = datetime.datetime.now().strftime(r"%Y%m%dT%H%M%S")
        #log.info("   sen2cor processing time stamp = " + image_timestamp)
        #log.info("   out_directory = " + out_directory)
        out_name = build_sen2cor_output_path(image_path, 
                                             image_timestamp, 
                                             get_sen2cor_version(sen2cor_path)
                                             )
        #log.info("   out name = " + out_name)
        out_path = os.path.join(out_directory, os.path.basename(out_name))
        #log.info("   out path = " + out_path)

        # create a search string to check whether a file with that name pattern 
        # already exists. This will extract the following file name pattern:
        # e.g. full path: "/data/36NXG/composite/
        #                   S2B_MSIL2A_20220312T074719_N0400_R135_T36NXG_202203
        #                   12T103427.SAFE/S2B_MSIL2A_20220312T074719_N0400_
        #                   R135_T36NXG_20220312T103427.SAFE"
        # out_glob will be: "S2B_MSIL2A_20220312T074719_N0400_R135_T36NXG*"

        out_glob = "_".join(out_path.split("/")[-1].split("_")[0:6])+"*"
        out_glob = os.path.join(out_directory, out_glob)
        #log.info("   out glob created by .split = " + out_glob)
        matching_l2a = glob.glob(out_glob)

        if matching_l2a:
            log.info(f"Skipping atmospheric correction of     : {image_path}")
            if len(matching_l2a) == 1:
                log.info(f"  because an L2A product already exists: {matching_l2a[0]}")
            else:
                log.info("  because matching L2A product files already exist:")
                for item in matching_l2a:
                    log.info(f"  {item}")
            continue
        else:
            log.info("Atmospheric correction of {}".format(image_path))
            try:
                l2_path = apply_sen2cor(
                    image_path,
                    sen2cor_path,
                    delete_unprocessed_image=delete_unprocessed_image,
                    log=log,
                )
                l2_name = os.path.basename(l2_path)
                #log.info("Changing L2A path: {}".format(l2_path))
                #log.info("  to new L2A path: {}".format(os.path.join(
                #    out_directory, 
                #    l2_name))
                #    )
                if os.path.exists(l2_path):
                    os.rename(l2_path, os.path.join(out_directory, l2_name))
                else:
                    log.error(f"L2A path not found after atmospheric correction with Sen2Cor: {l2_path}")
            except (subprocess.CalledProcessError, BadS2Exception):
                log.error(f"Atmospheric correction failed for {image}. Skipping.")
    return


def create_mask_from_model(
    image_path, model_path, model_clear=0, num_chunks=10, buffer_size=0
):
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
    
    # TODO: Fix this properly. Deferred import to deal with circular reference
    from pyeo.classification import classify_image

    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        log = logging.getLogger(__name__)
        log.info(
            "Building cloud mask for {} with model {}".format(image_path, model_path)
        )
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


def create_mask_from_confidence_layer(
    l2_safe_path, out_path, cloud_conf_threshold=0, buffer_size=3
):
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
    log.info(
        "Creating mask for {} with {} confidence threshold".format(
            l2_safe_path, cloud_conf_threshold
        )
    )
    if cloud_conf_threshold:
        cloud_paths = get_filenames(l2_safe_path, "_CLDPRB_R20.jp2", "QI_DATA")
        if not cloud_paths:
            raise FileNotFoundError(
                "Cloud probability mask not found for safe file {}".format(l2_safe_path)
            )
        cloud_path = cloud_paths[0]
        # cloud_glob = "GRANULE/*/QI_DATA/*CLD*_20m.jp2"  # This should match both old and new mask formats
        # cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
        cloud_image = gdal.Open(cloud_path)
        cloud_confidence_array = cloud_image.GetVirtualMemArray()
        mask_array = cloud_confidence_array < cloud_conf_threshold
        cloud_confidence_array = None
    else:
        cloud_paths = get_filenames(l2_safe_path, "_SCL_20m.jp2", "R20m")
        if not cloud_paths:
            raise FileNotFoundError(
                "Scene classification layer (SCL) not found for safe file {}".format(
                    l2_safe_path
                )
            )
        cloud_path = cloud_paths[0]
        # cloud_glob = "GRANULE/*/IMG_DATA/R20m/*SCL*_20m.jp2"  # This should match both old and new mask formats
        # cloud_path = glob.glob(os.path.join(l2_safe_path, cloud_glob))[0]
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


def create_mask_from_scl_layer(l2_safe_path, out_path, scl_classes, buffer_size=0):
    """
    Creates a multiplicative binary mask where pixels of class scl_class are set to 0 and
    other pixels are 1.

    Parameters
    ----------
    l2_safe_path : str
        Path to the L1
    out_path : str
        Path to the new path
    scl_classes: list of int
        Class values of the SCL scene classification layer to be set to 0
    buffer_size : int, optional
        The size of the buffer to apply around the masked out pixels (dilation)

    Returns
    -------
    out_path : str
        The path to the mask

    Notes
    -----
    The SCL codes correspond to the following classification labels:

    - 0: NO_DATA
    - 1: SATURATED_OR_DEFECTIVE
    - 2: DARK_AREA_PIXELS
    - 3: CLOUD_SHADOWS
    - 4: VEGETATION
    - 5: NOT_VEGETATED
    - 6: WATER
    - 7: UNCLASSIFIED
    - 8: CLOUD_MEDIUM_PROBABILITY
    - 9: CLOUD_HIGH_PROBABILITY
    - 10: THIN_CIRRUS
    - 11: SNOW  

    """
    log = logging.getLogger(__name__)
    log.info(
        "Creating scene classification mask for {} with SCL classes {}".format(
            l2_safe_path, scl_classes
        )
    )
    scl_glob = "GRANULE/*/IMG_DATA/R20m/*SCL*_20m.jp2"  # This should match both old and new mask formats
    df = get_raster_paths([l2_safe_path], filepatterns=["SCL"], dirpattern="R20m")
    # log.info(df.columns)
    # log.info(len(df))
    scl_path = df["SCL"][0][0]
    # scl_path = glob.glob(os.path.join(l2_safe_path, scl_glob))[0]
    log.info("  Opening SCL image: {}".format(scl_path))
    scl_image = gdal.Open(scl_path)
    scl_array = scl_image.GetVirtualMemArray()
    mask_array = np.logical_not(np.isin(scl_array, (scl_classes)))
    mask_image = create_matching_dataset(scl_image, out_path)
    mask_image_array = mask_image.GetVirtualMemArray(eAccess=gdal.GF_Write)
    np.copyto(mask_image_array, mask_array)
    mask_image_array = None
    scl_image = None
    mask_image = None
    resample_image_in_place(out_path, 10)
    if buffer_size:
        buffer_mask_in_place(out_path, buffer_size)
    return out_path


def create_mask_from_class_map(
    class_map_path, out_path, classes_of_interest, buffer_size=0, out_resolution=None
):
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
    try:
        class_image = gdal.Open(class_map_path)
    except:
        log.warning(
            "Could not open file. Skipping classification mask creation: ".format(
                class_map_path
            )
        )
        return ""
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


def create_mask_from_band(
    in_raster_path,
    out_path,
    band,
    threshold,
    relation="smaller",
    buffer_size=0,
    out_resolution=None,
):
    """
    Creates a multiplicative mask from a classification mask: 1 for each pixel containing one of classes_of_interest,
    otherwise 0

    Parameters
    ----------
    in_raster_path : str
        Path to the raster file to build the mask from
    out_path : str
        Path to the new mask
    band : number
        Number of the band in the raster file to be used to create the mask, starting with band 1
    threshold : number
        Threshold to be applied when creating the mask
    relation : str
        Relationship to be applied to the threshold, can be that pixels are "smaller" or "greater"
        than the threshold to be set to 1.
    buffer_size : int
        If greater than 0, applies a buffer to the masked pixels of this size. Defaults to 0.
    out_resolution : int or None, optional
        If present, resamples the mask to this resolution. Applied before buffering. Defaults to 0.

    Returns
    -------
    out_path : str
        The path to the new mask.

    """
    raster_image = gdal.Open(in_raster_path)
    raster_array = raster_image.GetVirtualMemArray()
    band_array = raster_array[band - 1]
    if relation == "smaller":
        mask_array = np.less(band_array, np.full_like(band_array, threshold))
    if relation == "greater":
        mask_array = np.greater(band_array, np.full_like(band_array, threshold))
    out_mask = create_matching_dataset(
        raster_image, out_path, bands=1, datatype=gdal.GDT_Byte
    )
    out_array = out_mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    np.copyto(out_array, mask_array)
    band_array = None
    raster_image = None
    raster_array = None
    out_array = None
    out_mask = None
    if out_resolution:
        resample_image_in_place(out_path, out_resolution)
    if buffer_size:
        buffer_mask_in_place(out_path, buffer_size)
    return out_path


def add_masks(mask_paths, out_path, geometry_func="union"):
    """
    Creates a raster file by adding a list of mask files containing 0 and 1 values

    Parameters
    ----------
    mask_paths : list of str
        List of strings containing the full directory paths and file names of all masks to be added
    geometry_func : {'intersect' or 'union'}
        How to handle non-overlapping masks. Defaults to 'union'

    Returns
    -------
    out_path : str
        The path to the new raster file.

    """
    
    log = logging.getLogger(__name__)
    log.info("Adding masks:")
    for mask in mask_paths:
        log.info("   {}".format(mask))
    masks = [gdal.Open(mask_path, gdal.GA_ReadOnly) for mask_path in mask_paths]
    if None in masks:
        raise FileNotFoundError(
            "Bad mask path in one of the following: {}".format(mask_paths)
        )
    combined_polygon = align_bounds_to_whole_number(
        get_combined_polygon(masks, geometry_func)
    )
    gt = masks[0].GetGeoTransform()
    x_res = gt[1]
    y_res = gt[5] * -1  # Y res is -ve in geotransform
    bands = 1
    projection = masks[0].GetProjection()
    out_raster = create_new_image_from_polygon(
        combined_polygon,
        out_path,
        x_res,
        y_res,
        bands,
        projection,
        datatype=gdal.GDT_Byte,
    )
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write).squeeze()
    out_array[:, :] = 1
    for mask_index, in_mask in enumerate(masks):
        in_mask_array = in_mask.GetVirtualMemArray().squeeze()
        if geometry_func == "intersect":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
                out_raster, combined_polygon
            )
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
                in_mask, combined_polygon
            )
        elif geometry_func == "union":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
                out_raster, get_raster_bounds(in_mask)
            )
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
                in_mask, get_raster_bounds(in_mask)
            )
        else:
            raise Exception("Invalid geometry_func; can be 'intersect' or 'union'")
        out_view = out_array[out_y_min:out_y_max, out_x_min:out_x_max]
        in_mask_view = in_mask_array[in_y_min:in_y_max, in_x_min:in_x_max]
        if mask_index == 0:
            out_view[:, :] = in_mask_view
        else:
            out_view[:, :] = np.add(out_view, in_mask_view, dtype=np.uint8)
        in_mask_view = None
        out_view = None
        in_mask_array = None
    out_array = None
    out_raster = None
    for mask in masks:
        mask = None
    return out_path


def change_from_class_maps(
    old_class_path,
    new_class_path,
    change_raster,
    dNDVI_raster,
    NDVI_raster,
    change_from,
    change_to,
    report_path,
    skip_existing=False,
    old_image_dir=None,
    new_image_dir=None,
    viband1=None,
    viband2=None,
    dNDVI_threshold=None,
    log=logging.getLogger(__name__),
):
    """
    This function looks for changes from class 'change_from' in the composite to any of the 'change_to_classes'
    in the change images. Pixel values are the acquisition date of the detected change of interest or zero.
    Optionally, changes will be confirmed by thresholding a vegetation index calculated from two bands if the difference between
    the more recent date and the older date is below the confirmation threshold (e.g. NDVI < -0.2).
    It then updates the report file which has three layers:
      (1) pixels show the earliest change detection date (expressed as the number of days since 1/1/2000)
      (2) pixels show the number of change detections (summed up over time)
      (3) pixels show the number of consecutive change detections, omitting cloudy observations but resetting counter when no change is detected

    Parameters
    ----------
    old_class_path : str
        Paths to a classification map to be used as the baseline map for the change detection.

    new_class_path : str
        Paths to a classification map with a newer acquisition date.

    change_raster : str
        Path to the output raster file that will be created. Will be updated with the latest time stamp.

    dNDVI_raster : str
        Path to the output dNDVI raster file that will be created. Will be updated with the latest time stamp.

    NDVI_raster : str
        Path to the output NDVI raster file for the change image that will be created. Will be updated with the latest time stamp.

    change_from : list of int
        List of integers with the class codes to be used as 'from' in the change detection.

    change_to : list of int
        List of integers with the class codes to be used as 'to' in the change detection.

    report_path : str
        Path to an aggregated report map that will be continuously updated with change detections from newer acquisition dates.
        Will be created if it does not exist.

    skip_existing : boolean
        If True, skip the production of files that already exist.

    old_image_dir : str
        Path to the directory containing the spectral image file with a matching timestamp to the old_class_path for vegetation index calculation.

    new_image_dir : str
        Path to the directory containing the spectral image file with a matching timestamp to the new_class_path for vegetation index calculation.

    viband1 : int
        If given, this is the first band number (start from band 1) for calculating the vegetation index from a raster file with matching timestamp:
        vi = (viband1 - viband2) / (viband1 + viband2)

    viband2 : int
        If given, this is the second band number (start from band 1) for calculating the vegetation index from a raster file with matching timestamp
        vi = (viband1 - viband2) / (viband1 + viband2)

    dNDVI_threshold : float
        If given, this is the threhold for checking change detections based on the vegetation index:
        A change pixel is confirmed if vi < threshold and discarded otherwise.

    log : (Optional)
        If not provided, the default namespace logger will be used.
        Otherwise, the logger passed will be used to print statements to.

    Returns:
    ----------
    change_raster : str (path to a raster file of type Int32)
        The path to the new change layer containing the acquisition date of the new class image
        expressed as the difference to the 1/1/2000, where a change has been found, -1 for cloudy pixels
        or missing data in the more recent classification map (pixels == 0) or zero otherwise.
    """

    if not os.path.exists(report_path):
        ## Changed to always generate report from scratch - incremental update 
        ##   is unstable and inflexible
        ## Any previous reports automatically copied to prefix 'archived_' in
        ##   tile_based_change_detection_from_cover_maps.py
        ##TODO: Bands MUST be zeroed on creation as they hold state - CHECK GDAL guarantees this
        log.info("Creating report image file: {}".format(report_path))
        new_class_image = gdal.Open(new_class_path, gdal.GA_ReadOnly)
        report_image = create_matching_dataset(
            new_class_image,
            report_path,
            format="GTiff",
            bands=18,
            datatype=gdal.GDT_Int16,
        )
        new_class_image = None
        report_image = None
    if not os.path.exists(report_path):
        log.error("Report file already exists. Skipping change detection.")
        return -1

    dNDVI_scale_factor = 100  # Multiplier used to scale dNDVI to integer range

    # create masks from the classes of interest
    #TODO: create temp dir in tile_dir on Linux and in ~ on Windows
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        from_class_mask_path = create_mask_from_class_map(
            class_map_path=old_class_path,
            out_path=os.path.join(
                td, os.path.basename(old_class_path)[:-4] + "_temp.msk"
            ),
            classes_of_interest=change_from,
        )
        to_class_mask_path = create_mask_from_class_map(
            class_map_path=new_class_path,
            out_path=os.path.join(
                td, os.path.basename(new_class_path)[:-4] + "_temp.msk"
            ),
            classes_of_interest=change_to,
        )
        if from_class_mask_path == "" or to_class_mask_path == "":
            log.warning("Cannot create change raster from:")
            log.warning("        {}".format(old_class_path))
            log.warning("   and  {}".format(new_class_path))
            return ""
        # combine masks by finding pixels that are 1 in the old mask and 1 in the new mask
        added_mask_path = add_masks(
            [from_class_mask_path, to_class_mask_path],
            os.path.join(td, "combined.msk"),
            geometry_func="intersect",
        )
        log.info("added mask path {}".format(added_mask_path))
        new_class_image = gdal.Open(new_class_path, gdal.GA_ReadOnly)
        new_class_array = new_class_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()

        change_image = create_matching_dataset(
            in_dataset=new_class_image,
            out_path=change_raster,
            format="GTiff",
            bands=1,
            datatype=gdal.GDT_Int32,
        )
        change_array = change_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()

        dNDVI_image = create_matching_dataset(
            new_class_image,
            dNDVI_raster,
            format="GTiff",
            bands=1,
            datatype=gdal.GDT_Int32,
        )
        dNDVI_array = dNDVI_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()

        NDVI_image = create_matching_dataset(
            new_class_image,
            NDVI_raster,
            format="GTiff",
            bands=1,
            datatype=gdal.GDT_Int32,
        )
        NDVI_array = NDVI_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()

        added_mask = gdal.Open(added_mask_path, gdal.GA_ReadOnly)
        added_mask_array = added_mask.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()
        # Gets timestamp as integer in form yyyymmdd
        new_date = get_image_acquisition_time(new_class_path)
        reference_date = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
        date_difference = new_date - reference_date
        date = np.uint32(
            date_difference.total_seconds() / 60 / 60 / 24
        )  # convert to 24-hour days
        # Matt: added serial_date_to_string function
        log.info("date of change in days since 2000-01-01 = {}".format(date))
        log.info(f"date of change  : {serial_date_to_string(int(date))}")
        # replace all pixels != 2 with 0 and all pixels == 2 with the new acquisition date
        change_array[np.where(added_mask_array == 2)] = date
        change_array[np.where(added_mask_array != 2)] = 0
        # set clouds and missing values in latest class image to -1 in the change layer
        change_array[np.where(new_class_array == 0)] = -1
        if (
            viband1 is not None
            and viband2 is not None
            and dNDVI_threshold is not None
        ):
            log.info(
                "Confirming detected class transitions based on vegetation index differencing."
            )
            log.info(
                "  VI = (band{} - band{}) / (band{} + band{})".format(
                    viband1, viband2, viband1, viband2
                )
            )
            log.info(
                "  confirming changes if VI_new - VI_old < {}".format(
                    dNDVI_threshold
                )
            )
            # get composite file name and find bands in composite
            old_timestamp = pyeo.filesystem_utilities.get_image_acquisition_time(
                os.path.basename(old_class_path)
            )
            old_image_path = [
                f.name
                for f in os.scandir(old_image_dir)
                if f.is_file()
                and f.name.startswith("composite_")
                and old_timestamp.strftime("%Y%m%dT%H%M%S") in f.name
                and f.name.endswith(".tif")
            ][0]
            if len(old_image_path) > 0:
                log.info("Found old satellite image: {}".format(old_image_path))
                # open file and read bands
                old_image = gdal.Open(
                    os.path.join(old_image_dir, old_image_path), gdal.GA_ReadOnly
                )
                log.info("old image successfully read")
                old_image_array = old_image.GetVirtualMemArray(
                    eAccess=gdal.gdalconst.GF_Read
                ).squeeze()
                log.info(
                    "old image successfully read as a Virtual Memory Array and squeezed"
                )
                # calculate composite VI (N.B. -1 because array starts numbering with 0 and GDAL numbers bands from 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    vi_old = np.true_divide(
                        (
                            1.0 * old_image_array[viband1 - 1, :, :]
                            - 1.0 * old_image_array[viband2 - 1, :, :]
                        ),
                        (
                            1.0 * old_image_array[viband1 - 1, :, :]
                            + 1.0 * old_image_array[viband2 - 1, :, :]
                        ),
                    )
                    vi_old[vi_old == np.inf] = 0
                    vi_old = np.nan_to_num(vi_old)
                old_image_array = None
                old_image = None
                log.info("old image successfully closed, therefore, saved")
                # get change image file name and find bands in change image
                new_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        os.path.basename(new_class_path)
                    )
                )
                log.info(f"new timestamp {new_timestamp}")
                new_image_path = [
                    f.name
                    for f in os.scandir(new_image_dir)
                    if f.is_file()
                    and f.name.startswith("S2")
                    and new_timestamp.strftime("%Y%m%dT%H%M%S") in f.name
                    and f.name.endswith(".tif")
                ][0]
                log.info(f"new image path {new_image_path}")

                if len(new_image_path) > 0:
                    log.info("Found new satellite image: {}".format(new_image_path))
                    # open file and read bands
                    new_image = gdal.Open(
                        os.path.join(new_image_dir, new_image_path),
                        gdal.GA_ReadOnly,
                    )
                    new_image_array = new_image.GetVirtualMemArray(
                        eAccess=gdal.gdalconst.GF_Read
                    ).squeeze()
                    # calculate change image VI
                    with np.errstate(divide="ignore", invalid="ignore"):
                        vi_new = np.true_divide(
                            (
                                1.0 * new_image_array[viband1 - 1, :, :]
                                - 1.0 * new_image_array[viband2 - 1, :, :]
                            ),
                            (
                                1.0 * new_image_array[viband1 - 1, :, :]
                                + 1.0 * new_image_array[viband2 - 1, :, :]
                            ),
                        )
                        vi_new[vi_new == np.inf] = 0
                        vi_new = np.nan_to_num(
                            vi_new
                        )  # I.R. Replaces NaN with zero and infinity with large finite numbers

                    # calculate dVI = new minus old VI
                    dvi = vi_new - vi_old

                    # I.R. 20230501 Force cloud masked or out-of-orbit (no data) regions of ndvi to -1
                    vi_new[new_image_array[viband1 - 1, :, :] == 0] = -1

                    # I.R. 20230501 Force cloud masked or out-of-orbit (no data) regions of dNDVI to 1
                    dvi[new_image_array[viband1 - 1, :, :] == 0] = 1

                    # I.R. Sets all values where the VI difference exceeds the threshold to a distictive negative value
                    # to indicate a decision of 'no change' and as a record that this was due to the dNDVI test failing
                    # change_array[np.where(dvi >= dNDVI_threshold)] = -2

                    # I.R. 20230421+ START: Save NDVI and dNDVI of change image to disk for analysis
                    # dNDVI_scale_factor = (
                    #     100  # Multiplier used to scale dNDVI to integer range
                    # )
                    np.copyto(dNDVI_array, (dvi * dNDVI_scale_factor).astype(int))
                    NDVI_scale_factor = (
                        100  # Multiplier used to scale NDVI to integer range
                    )
                    np.copyto(NDVI_array, (vi_new * NDVI_scale_factor).astype(int))
                    # I.R. 20230421 END

                    new_image_array = None
                    new_image = None
                    dvi = None
                    vi_new = None
                    vi_old = None

                else:
                    log.error(
                        "Did not find a new satellite image with name pattern: {}".format(
                            new_timestamp.strftime("%Y%m%dT%H%M%S")
                        )
                    )
                    log.error(
                        "Skipping vegetation index calculation and confirmation of change detections."
                    )
            else:
                log.error(
                    "Did not find an old satellite image with name pattern: {}".format(
                        old_timestamp.strftime("%Y%m%dT%H%M%S")
                    )
                )
                log.error(
                    "Skipping vegetation index calculation and confirmation of change detections."
                )
        # save change layer
        new_class_array = None
        new_class_image = None
        added_mask_array = None
        added_mask = None
        change_array = None
        change_image = None
        dNDVI_array = None
        dNDVI_image = None
        NDVI_array = None
        NDVI_image = None

        # I.R. 20220611 START Section removed - no longer updating report file name incrementally
        # - now defined once in tile_base_change_detection_from_cover_maps.py
        # =============================================================================
        #         # within this processing loop, update the report layers indicating the length of temporal sequences of confirmed values
        #         # ensure that the date of the new change layer is AFTER the report file was last updated
        #         baseline_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(report_path))[0]
        #         report_last_updated_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(report_path))[1]
        #         new_changes_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(change_raster))[1]
        #         # pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(new_class_path))
        #         if report_last_updated_timestamp > new_changes_timestamp:
        #             log.warning("Date of the new change map is not recent enough to update the current report image product: ")
        #             log.warning("  report image: {}".format(report_path))
        #             log.warning("  last updated: {}".format(report_last_updated_timestamp))
        #             log.warning("  change image:  {}".format(change_raster))
        #             log.warning("  updated:      {}".format(new_changes_timestamp))
        #             log.warning("Skipping updating of report image product.")
        #             return change_raster
        #         else:
        #             log.info("Updating current report image product with the new change map: ")
        #             log.info("  report image: {}".format(report_path))
        #             log.info("  last updated: {}".format(report_last_updated_timestamp))
        #             log.info("  change image:  {}".format(change_raster))
        #             log.info("  updated:      {}".format(new_changes_timestamp))
        #         #TODO: update name of report_path with new_changes_timestamp
        #         tile_id = os.path.basename(report_path).split("_")[2]
        #
        #         old_report_path = str(report_path) # str() creates a copy of the string
        #         report_path = os.path.join(os.path.dirname(report_path),
        #                                        "report_{}_{}_{}.tif".format(
        #                                        baseline_timestamp.strftime("%Y%m%dT%H%M%S"),
        #                                        tile_id,
        #                                        new_changes_timestamp.strftime("%Y%m%dT%H%M%S"))
        #                                        )
        #         log.info("  updated report file:      {}".format(report_path))
        #         report_archive_path = os.path.join(os.path.dirname(old_report_path),
        #                                            "archived_"+os.path.basename(old_report_path))
        #         log.info("  archived report file:      {}".format(report_archive_path))
        #
        #         shutil.copy(old_report_path, report_archive_path)
        #         os.rename(old_report_path, report_path)
        #
        #
        # =============================================================================
        # I.R. 20220611 END

        #log.info("***   Loading data arrays required to build report layers   ***")

        new_class_image = gdal.Open(new_class_path, gdal.GA_ReadOnly)
        new_class_array = new_class_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()

        change_image = gdal.Open(change_raster, gdal.GA_ReadOnly)
        change_array = change_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()

        dNDVI_image = gdal.Open(dNDVI_raster, gdal.GA_ReadOnly)
        dNDVI_array = dNDVI_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()

        NDVI_image = gdal.Open(NDVI_raster, gdal.GA_ReadOnly)
        NDVI_array = NDVI_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()

        report_image = gdal.Open(report_path, gdal.GA_Update)
        out_report_array = report_image.GetVirtualMemArray(
            eAccess=gdal.GA_Update
        ).squeeze()
        reference_projection = report_image.GetProjection()
        projection = change_image.GetProjection()
        if projection != reference_projection:
            log.warning(
                "Skipping change layer with a different map projection: {} is not the same as {}".format(
                    change_raster, report_path
                )
            )
            change_image = None
            report_image = None
            return -1

        #log.info("***   Starting build of report layers:   ***")

        #TODO: If kept as a feature move these parameters into .ini file

        # Absolute number of valid detections for classifier opinion to be accepted.
        minimum_required_validated_detections_threshold = 2  
        # Absolute number of dNDVI change detections for classifier opinion to be accepted.
        minimum_required_dNDVI_detections_threshold = 5
        # Absolute number of classifier-only detections for opinion to be accepted
        minimum_required_classifier_detections_threshold = 5
        percentage_probability_threshold = 50
        minimum_required_FROM_detections_threshold = 2
        minimum_required_TO_detections_threshold = 2

        log.info(
            f"percentage_probability_threshold:   {percentage_probability_threshold}"
        )
        log.info(
            f"minimum_required_validated_detections_threshold: {minimum_required_validated_detections_threshold}"
        )
        log.info(
            f"minimum_required_dNDVI_detections_threshold: {minimum_required_dNDVI_detections_threshold}"
        )
        log.info(
            f"minimum_required_classifier_detections_threshold: {minimum_required_classifier_detections_threshold}"
        )
        log.info(
            f"minimum_required_FROM_detections_threshold: {minimum_required_FROM_detections_threshold}"
        )
        log.info(
            f"minimum_required_TO_detections_threshold: {minimum_required_TO_detections_threshold}"
        )

        #TODO: Move logging of layer meanings to a separate function and call 
        #      from detect_change.py and only log it once
        log.info("Creating Report Layers")

        # Layer 0
        log.info(
            "Layer 0: Total Image Count: Counts the number of images processed"+
            " per pixel - number of available images within overall cloud "+
            "percentage cover limit set in pyeo.ini file"
        )
        locs = (
            out_report_array[0, :, :] >= 0
        )   
        # i.e. locs covers all locations - an inefficient global counter 
        #    but useful for computed fields and when viewing in QGIS
        out_report_array[0, locs] = out_report_array[0, locs] + 1

        # Layer 1
        log.info(
            "Layer 1: Occluded Image Count: Counts number of cloud occluded "+
            "(or out-of-orbit) images that are thus unavailable for "+
            "classification and analysis"
        )
        locs = change_array == -1
        out_report_array[1, locs] = out_report_array[1, locs] + 1

        # Layer 2
        log.info(
            "Layer 2: Classifier Change Detection Count: Count if a from/to "+
            "change of classification was detected"
        )
        locs = change_array > 0
        out_report_array[2, locs] = out_report_array[2, locs] + 1

        # Layer 3
        log.info(
            "Layer 3: First-Change Trigger for Combined Classifier+dNDVI "+
            "Validated Change Detection: Records earliest date of a "+
            "classification change detection where values are greater than "+
            "zero (not missing data and not cloud)"
        )
        # where layer 2 is zero and the change array contains a value > 0, this date will be burned into the report layer 2
        locs = (
            (out_report_array[3, :, :] == 0)
            & (change_array > 0)
            & (dNDVI_array < int(dNDVI_threshold * dNDVI_scale_factor))
        )  # Base on dNDVI validated change detections
        out_report_array[3, locs] = change_array[locs]
        # where layer 3 is non-zero it is set to the earlier date
        locs = (
            (out_report_array[3, :, :] > 0)
            & (change_array > 0)
            & (dNDVI_array < int(dNDVI_threshold * dNDVI_scale_factor))
        )  # Base on dNDVI validated change detections
        out_report_array[3, locs] = np.minimum(
            out_report_array[3, locs], change_array[locs]
        )

        # Layer 4:
        log.info(
            "Layer 4: Combined Classifier+dNDVI Validated Change Detection "+
            "Count: Count if a change was detected after a first change has "+
            "already been detected"
        )
        locs = (
            (change_array > 0)
            & (dNDVI_array < int(dNDVI_threshold * dNDVI_scale_factor))
            & (out_report_array[3, :, :] > 0)
        )
        out_report_array[4, locs] = out_report_array[4, locs] + 1

        # Layer 5:
        log.info(
            "Layer 5: Combined Classifier+dNDVI Validated No Change Detection"+
            " Count: Count if a no change was detected after a first change "+
            "has already been detected"
        )
        locs = (change_array == 0) & (out_report_array[3, :, :] > 0)
        out_report_array[5, locs] = out_report_array[5, locs] + 1

        # Layer 6:
        log.info(
            "Layer 6: Cloud Occlusion Count: Count if a cloud occlusion "+
            "(or out-of-orbit) occured after a first change has already "+
            "been detected"
        )
        locs = (change_array == -1) & (out_report_array[3, :, :] > 0)
        out_report_array[6, locs] = out_report_array[6, locs] + 1

        # Compute following report measures only for pixels where first change 
        #   has been detected (to avoid division by zero)
        #TODO: INEFFICIENT! All computed layers should be generated in a 
        #   separate stage after iteration through the change layers has been 
        #   completed
        locs = out_report_array[3, :, :] > 0

        # Layer 7:
        log.info(
            "Layer 7: Valid Image Count: Total number of valid (no cloud) "+
            "images for this pixel since first change was detected"
        )
        # = total change + change_filtered + nochange
        out_report_array[7, locs] = (
            out_report_array[4, locs] + out_report_array[5, locs]
        )

        # Layer 8:
        log.info(
            "Layer 8: Change Detection Repeatability: Repeatability of "+
            "change detection after first change is detected - as a "+
            "percentage of available valid images"
        )
        # = ratio change/(change + nochange) for pixels where first change has been detected
        out_report_array[8, locs] = (
            100 * out_report_array[4, locs]
        ) / out_report_array[
            7, locs
        ]  # Base on dNDVI validated change detections

        # Layer 9:
        log.info(
            "Layer 9: Binary time-series decision: Based on "+
            "percentage_probability_threshold and "+
            "minimum_required_validated_detections_threshold"
        )
        out_report_array[9, :, :] = 0  ## Reset from previous calls
        locs_binarise = (
            out_report_array[8, :, :] >= percentage_probability_threshold
        ) & (
            out_report_array[4, :, :] >= minimum_required_validated_detections_threshold
        )  # Base on dNDVI validated change detections
        out_report_array[
            9, locs_binarise
        ] = 1  # Assumes empty array layer initialised to zero

        # Layer 10
        log.info(
            "Layer 10: Binary time-series decision by first-change date: "+
            "First change date masked by Binary Decision - Layer 9"
        )
        out_report_array[10, :, :] = (
            out_report_array[3, :, :] * out_report_array[9, :, :]
        )

        # Layer 11:
        log.info(
            "Layer 11: dNDVI Only Change Detection Count: Count if a change "+
            "was detected by the dNDVI test and that not cloud occluded "+
            "(or out-of-orbit)"
        )
        locs = (change_array >= 0) & (
            dNDVI_array < int(dNDVI_threshold * dNDVI_scale_factor)
        )
        out_report_array[11, locs] = out_report_array[11, locs] + 1

        # Layer 12:
        log.info(
            "Layer 12: Binary time-series decision: Based on dNDVI Only "+
            "and minimum_required_dNDVI_detections_threshold"
        )
        out_report_array[12, :, :] = 0  ## Reset from previous calls
        # locs_binarise = (out_report_array[8, :, :] >= percentage_probability_threshold) & (out_report_array[6, :, :] >= minimum_required_validated_detections_threshold)
        locs_binarise = (
            out_report_array[11, :, :] >= minimum_required_dNDVI_detections_threshold
        )
        out_report_array[
            12, locs_binarise
        ] = 1  # Assumes empty array layer initialised to zero

        # Layer 13:
        log.info("Layer 13: Binary time-series decision: Based on Classifier Only")
        out_report_array[13, :, :] = 0  ## Reset from previous calls
        locs_binarise = (
            out_report_array[2, :, :]
            >= minimum_required_classifier_detections_threshold
        )
        out_report_array[
            13, locs_binarise
        ] = 1  # Assumes empty array layer initialised to zero

        # Layer 14:
        log.info(
            "Layer 14: Combined Classifier+dNDVI Binary time-series "+
            "decision: Based on Classifier AND dNDVI opinion"
        )
        out_report_array[14, :, :] = 0  ## Reset from previous calls
        locs_binarise = (out_report_array[12, :, :] > 0) & (
            out_report_array[13, :, :] > 0
        )
        out_report_array[
            14, locs_binarise
        ] = 1  # Assumes empty array layer initialised to zero

        # Layer 15:
        log.info("Layer 15: FROM Classification Count")
        # Assumes layer was initialised to zero when report file was created 
        #   above e.g. with out_report_array[15, :, :] = 0
        locs_from = np.isin(
            new_class_array, change_from
        )  # change_from parameter holds a list of classes
        # locs_from = np.nonzero(np.isin(new_class_array, change_from))  
        #   change_from parameter holds a list of classes
        out_report_array[15, locs_from] = (
            out_report_array[15, locs_from] + 1
        )  # Assumes empty array layer initialised to zero

        # Layer 16:
        log.info("Layer 16: TO Classification Count")
        # Assumes layer was initialised to zero when report file was created 
        #   above e.g. with out_report_array[16, :, :] = 0
        locs_to = np.isin(
            new_class_array, change_to
        )  # change_from parameter holds a list of classes
        # locs_to = np.nonzero(np.isin(new_class_array, change_to))  
        #   change_to parameter holds a list of classes
        out_report_array[16, locs_to] = (
            out_report_array[16, locs_to] + 1
        )  # Assumes empty array layer initialised to zero

        # Layer 17:
        log.info(
            "Layer 17: Binary Decision Thresholds on FROM and TO Classification Counts"
        )
        out_report_array[17, :, :] = 0  ## Reset from previous calls
        locs_tofrom = (
            out_report_array[15, :, :] >= minimum_required_FROM_detections_threshold
        ) & (out_report_array[16, :, :] >= minimum_required_TO_detections_threshold)
        out_report_array[
            17, locs_tofrom
        ] = 1  # Assumes empty array layer initialised to zero

        # ORIGINAL CODE
        # increase counter if a change was detected
        # locs = ( change_array > 0 )
        # out_report_array[2, locs] = out_report_array[2, locs] + 1
        # # reset the counter if no change was detected
        # locs = ( change_array == 0 )
        # out_report_array[2, locs] = out_report_array[2, locs] - 1

        # I.R. 20220603/20230421 END

        change_array = None
        change_image = None
        dNDVI_array = None
        dNDVI_image = None
        out_report_array = None
        report_image = None
    return change_raster


def change_from_class_maps_depracated(
    old_class_path,
    new_class_path,
    change_raster,
    change_from,
    change_to,
    skip_existing=False,
):
    """
    This function looks for changes from class 'change_from' in the composite to any of the 'change_to_classes'
    in the change images. Pixel values are the acquisition date of the detected change of interest or zero.

    Parameters
    ----------
    old_class_path : str
        Paths to a classification map to be used as the baseline map for the change detection.

    new_class_path : str
        Paths to a classification map with a newer acquisition date.

    change_raster : str
        Path to the output raster file that will be created.

    change_from : list of int
        List of integers with the class codes to be used as 'from' in the change detection.

    change_to : list of int
        List of integers with the class codes to be used as 'to' in the change detection.

    skip_existing : boolean
        If True, skip the production of files that already exist.

    Returns:
    ----------
    change_raster : str (path to a raster file of type UInt32)
        The path to the new change layer containing the acquisition date of the new class image
        expressed as the difference to the 1/1/2000, where a change has been found, or zero otherwise.
    """

    log.error("The function raster_manipulation.change_from_class_maps() is "+
              "depracated. Use raster_manipulation.change_from_class_maps() "+
              "instead.")
    sys.exit(1)
    
    if skip_existing and os.path.exists(change_raster):
        log.info("File already exists: {}. Skipping.".format(change_raster))
        return
    # create masks from the classes of interest
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        from_class_mask_path = create_mask_from_class_map(
            class_map_path=old_class_path,
            out_path=os.path.join(
                td, os.path.basename(old_class_path)[:-4] + "_temp.msk"
            ),
            classes_of_interest=change_from,
        )
        to_class_mask_path = create_mask_from_class_map(
            class_map_path=new_class_path,
            out_path=os.path.join(
                td, os.path.basename(new_class_path)[:-4] + "_temp.msk"
            ),
            classes_of_interest=change_to,
        )
        if from_class_mask_path == "" or to_class_mask_path == "":
            log.warning("Cannot create change raster from:")
            log.warning("        {}".format(old_class_path))
            log.warning("   and  {}".format(new_class_path))
            return ""
        # combine masks by finding pixels that are 1 in the old mask and 1 in the new mask
        added_mask_path = add_masks(
            [from_class_mask_path, to_class_mask_path],
            os.path.join(td, "combined.msk"),
            geometry_func="intersect",
        )
        log.info("added mask path {}".format(added_mask_path))
        new_class_image = gdal.Open(new_class_path, gdal.GA_ReadOnly)
        change_image = create_matching_dataset(
            new_class_image,
            change_raster,
            format="GTiff",
            bands=1,
            datatype=gdal.GDT_UInt32,
        )
        new_class_image = None
        change_array = change_image.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Write
        ).squeeze()
        added_mask = gdal.Open(added_mask_path, gdal.GA_ReadOnly)
        added_mask_array = added_mask.GetVirtualMemArray(
            eAccess=gdal.gdalconst.GF_Read
        ).squeeze()
        # Gets timestamp as integer in form yyyymmdd
        new_date = get_image_acquisition_time(new_class_path)
        reference_date = datetime.datetime(2000, 1, 1, 0, 0, 0, 0)
        date_difference = new_date - reference_date
        date = np.uint32(
            date_difference.total_seconds() / 60 / 60 / 24
        )  # convert to 24-hour days
        log.info("date = {}".format(date))
        # replace all pixels != 2 with 0 and all pixels == 2 with the new acquisition date
        change_array[np.where(added_mask_array == 2)] = date
        change_array[np.where(added_mask_array != 2)] = 0
        added_mask_array = None
        added_mask = None
        change_array = None
        change_image = None
    return change_raster


def verify_change_detections(
    class_map_paths, out_path, classes_of_interest, buffer_size=0, out_resolution=None
):
    """
    Verifies repeated change detections from subsequent class maps.
    Reads in a list of classification masks where 1 shows pixels containing a change class and adds them up.
    Classification maps are sorted by timestamp in the file name.
    The result is a raster file that contains a number from 0 (no change detected) to n (number of mask files)
    where n is the greatest confidence in the change detection.

    Parameters
    ----------
    class_map_paths : list of str
        List of paths to the classification maps to build the mask from
    out_path : str
        Path to the new raster file containing the confidence layer
    classes_of_interest : list of int
        The list of classes to count as clear pixels
    buffer_size : int
        If greater than 0, applies a buffer to the masked pixels of this size. Defaults to 0.
    out_resolution : int or None, optional
        If present, resamples the mask to this resolution. Applied before buffering. Defaults to 0.

    Returns
    -------
    out_path : str
        The path to the new mask.
    """

    log.info(
        "Producing confidence layer from subsequent detections for classes: {}".format(
            classes_of_interest
        )
    )
    # create masks from the classes of interest
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        class_mask_paths = [
            create_mask_from_class_map(
                f,
                os.path.join(td, os.path.basename(f).split(sep=".")[0] + "_temp.msk"),
                classes_of_interest,
            )
            for f in class_map_paths
        ]
        # combine masks from n subsequent dates into a confirmed change detection image
        add_masks(class_mask_paths, out_path, geometry_func="union")
    return out_path


def raster2array(raster_file):
    """
    Loads the contents of a raster file into an array.

    Parameters
    ----------
    raster_file : str
        Path and file name of the raster file
    """
    rasterArray = gdal_array.LoadFile(raster_file)
    return rasterArray


def array2raster(raster_file, new_raster_file, array):
    """
    Saves the contents of an array to a new raster file and copies the projection from
    an existing raster file.

    Parameters
    ----------
    raster_file : str
        Path and file name of the raster file from which the metadata will be copied

    new_raster_file : str
        Path and file name of the newly created raster file

    array : array
        The array that will be written to new_raster_file
    """
    raster = gdal.Open(raster_file)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    bands = raster.RasterCount
    driver = gdal.GetDriverByName(str("GTiff"))
    outRaster = driver.Create(new_raster_file, cols, rows, bands, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for band in range(bands):
        outband = outRaster.GetRasterBand(band + 1)
        outband.WriteArray(array[band])
        outband.FlushCache()
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())


def apply_mask_to_image(mask_path, image_path, masked_image_path):
    """
    Applies a mask of 0 and 1 values to a raster image with one or more bands in Geotiff format
    by multiplying each pixel value with the mask value.

    After:
        https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

    Parameters
    ----------
    mask_path : str
        Paths and file name of the mask file

    image_path : str
        Path and file name of the raster image file

    masked_image_path : str
        Path and file name of the masked raster image file that will be created

    """
    log = logging.getLogger(__name__)
    # log.info("Applying mask {} to raster image {}.".format(mask_path, image_path))
    mask = gdal.Open(mask_path)
    if mask == None:
        raise FileNotFoundError("Mask not found: {}".format(mask_path))
    # make the name of the masked output image
    # log.info("   masked raster image will be created at {}.".format(masked_image_path))
    # gdal read raster as array
    image_as_array = raster2array(image_path)
    # reproject the mask into the same shape and projection as the raster file if necessary
    raster = gdal.Open(image_path)
    geotransform_of_image = raster.GetGeoTransform()
    bands = raster.RasterCount
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    pixelWidth = geotransform_of_image[1]
    pixelHeight = geotransform_of_image[5]
    mask = gdal.Open(mask_path, gdal.GA_Update)
    geotransform_of_mask = mask.GetGeoTransform()
    if geotransform_of_image != geotransform_of_mask:
        flag = True  # True means the two geotransforms are almost identical and only have small rounding errors
        for g in range(len(geotransform_of_mask)):
            if geotransform_of_image[g] - geotransform_of_mask[g] > 0.000001:
                flag = False  # raise exception
        if not flag:
            with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
                temp_path_1 = os.path.join(td, "reproj_mask_temp.tif")
                # log.info("Geotransforms do not match. Reprojecting {} to {}.".format(mask_path, temp_path_1))
                reproject_image(
                    mask_path,
                    temp_path_1,
                    raster.GetProjectionRef(),
                    do_post_resample=False,
                )
                # log.info("Output file exists? {}".format(str(os.path.exists(temp_path_1))))
                temp_path_2 = os.path.join(
                    td,
                    os.path.basename(mask_path).split(".")[0] + "_warped_clipped.tif",
                )
                # log.info("Clipping {} to {} using extent from {}".format(temp_path_1, temp_path_2, image_path))
                clip_raster_to_intersection(
                    temp_path_1, image_path, temp_path_2, is_landsat=False
                )
                mask_path = mask_path.split(".")[0] + "_warped_clipped_resampled.tif"
                ds = gdal.Open(temp_path_2)
                ds_out = gdal.Translate(
                    mask_path,
                    ds,
                    format="GTiff",
                    outputType=gdal.GDT_Float32,
                    width=cols,
                    height=rows,
                    resampleAlg="bilinear",
                )
                ds = None
                ds_out = None
                mask = None
                mask = gdal.Open(mask_path, gdal.GA_Update)
        else:
            # copy the geotransform of the raster to the mask for consistency
            mask.SetGeoTransform(geotransform_of_image)

    geotransform_of_mask = mask.GetGeoTransform()
    if geotransform_of_image != geotransform_of_mask:
        flag = True  # True means the two geotransforms are almost identical and only have small rounding errors
        for g in range(len(geotransform_of_mask)):
            if geotransform_of_image[g] - geotransform_of_mask[g] > 0.000001:
                flag = False  # raise exception
        if not flag:
            log.warning(
                "Could not bring the mask file into exactly the same projection as the image. Check co-registration visually."
            )
            log.warning(mask_path)
            log.warning(geotransform_of_mask)
            log.warning(geotransform_of_image)
        else:
            # copy the geotransform of the raster to the mask for consistency
            mask.SetGeoTransform(geotransform_of_image)
    mask_as_array = raster2array(mask_path)
    for band in range(bands):
        image_as_array[band, mask_as_array == 0] = 0
    array2raster(image_path, masked_image_path, image_as_array)


def apply_mask_to_dir(mask_path, image_dir, masked_image_dir):
    """
    Iterates over all raster images in image_dir and applies the mask to each image.

    Parameters
    ----------
    mask_path : str
        Paths and file name of the mask file

    image_dir : str
        Path and directory name which contains the raster image files in Geotiff format

    masked_image_dir : str
        Path and directory name in which the masked raster image files will be created
    """

    log = logging.getLogger(__name__)
    log.info("Applying mask to all tiff files in dir: {}".format(image_dir))
    image_files = [
        f for f in os.listdir(image_dir) if f.endswith(".tif") or f.endswith(".tiff")
    ]
    image_files = [
        f for f in image_files if "_masked" not in os.path.basename(f)
    ]  # remove already masked tiff files to avoid double-processing
    for image_file in image_files:
        masked_image_path = os.path.join(
            masked_image_dir, os.path.basename(image_file).split(".")[0] + "_masked.tif"
        )
        apply_mask_to_image(mask_path, image_dir + "/" + image_file, masked_image_path)


def combine_masks(
    mask_paths, out_path, combination_func="and", geometry_func="intersect"
):
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
    log.info(
        "Combining masks using combination function {} and geometry function {}.".format(
            combination_func, geometry_func
        )
    )
    for mask in mask_paths:
        log.info("   {}".format(mask))
    masks = [gdal.Open(mask_path) for mask_path in mask_paths]
    if None in masks:
        raise FileNotFoundError(
            "Bad mask path in one of the following: {}".format(mask_paths)
        )
    combined_polygon = align_bounds_to_whole_number(
        get_combined_polygon(masks, geometry_func)
    )
    gt = masks[0].GetGeoTransform()
    x_res = gt[1]
    y_res = gt[5] * -1  # Y res is -ve in geotransform
    bands = 1
    projection = masks[0].GetProjection()
    out_mask = create_new_image_from_polygon(
        combined_polygon,
        out_path,
        x_res,
        y_res,
        bands,
        projection,
        datatype=gdal.GDT_Byte,
        nodata=0,
    )

    # This bit here is similar to stack_raster, but different enough to not be worth spinning into a combination_func
    # I might reconsider this later, but I think it'll overcomplicate things.
    out_mask_array = out_mask.GetVirtualMemArray(eAccess=gdal.GF_Write)
    out_mask_array = (
        out_mask_array.squeeze()
    )  # This here to account for unaccountable extra dimension Windows patch adds
    out_mask_array[:, :] = 1
    for mask_index, in_mask in enumerate(masks):
        in_mask_array = in_mask.GetVirtualMemArray()
        in_mask_array = in_mask_array.squeeze()  # See previous comment
        if geometry_func == "intersect":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
                out_mask, combined_polygon
            )
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
                in_mask, combined_polygon
            )
        elif geometry_func == "union":
            out_x_min, out_x_max, out_y_min, out_y_max = pixel_bounds_from_polygon(
                out_mask, get_raster_bounds(in_mask)
            )
            in_x_min, in_x_max, in_y_min, in_y_max = pixel_bounds_from_polygon(
                in_mask, get_raster_bounds(in_mask)
            )
        else:
            raise Exception("Invalid geometry_func; can be 'intersect' or 'union'")
        out_mask_view = out_mask_array[out_y_min:out_y_max, out_x_min:out_x_max]
        in_mask_view = in_mask_array[in_y_min:in_y_max, in_x_min:in_x_max]
        if mask_index == 0:
            out_mask_view[:, :] = in_mask_view
        else:
            if combination_func == "or":
                out_mask_view[:, :] = np.bitwise_or(
                    out_mask_view, in_mask_view, dtype=np.uint8
                )
            elif combination_func == "and":
                out_mask_view[:, :] = np.bitwise_and(
                    out_mask_view, in_mask_view, dtype=np.uint8
                )
            elif combination_func == "nor":
                out_mask_view[:, :] = np.bitwise_not(
                    np.bitwise_or(out_mask_view, in_mask_view, dtype=np.uint8),
                    dtype=np.uint8,
                )
            else:
                raise Exception(
                    "Invalid combination_func; valid values are 'or', 'and', and 'nor'"
                )
        in_mask_view = None
        out_mask_view = None
        in_mask_array = None
        in_mask = None
    out_mask_array = None
    out_mask = None
    return out_path


def buffer_mask_in_place(mask_path, buffer_size, cache=None):
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
    # log.info("Buffering {} with buffer size {}".format(mask_path, buffer_size))
    mask = gdal.Open(mask_path, gdal.GA_Update)
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    if buffer_size > 10:
        bfs = int(buffer_size / 10)
        if bfs * 10 != buffer_size:
            log.warning("Approximating buffer size as 10*{} = {}".format(bfs, 10 * bfs))
        if cache is None:
            cache = np.empty(mask_array.squeeze().shape, dtype=bool)
        ndimage.binary_erosion(
            mask_array.squeeze(), structure=morph.disk(bfs), iterations=10, output=cache
        )
    else:
        cache = morph.binary_erosion(
            mask_array.squeeze(), footprint=morph.disk(buffer_size)
        )
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


def create_mask_from_sen2cor_and_fmask(
    l1_safe_file, l2_safe_file, out_mask_path, buffer_size=0
):
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
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        s2c_mask_path = os.path.join(td, "s2_mask.tif")
        fmask_mask_path = os.path.join(td, "fmask.tif")
        create_mask_from_confidence_layer(
            l2_safe_file, s2c_mask_path, buffer_size=buffer_size
        )
        create_mask_from_fmask(l1_safe_file, fmask_mask_path)
        combine_masks(
            [s2c_mask_path, fmask_mask_path],
            out_mask_path,
            combination_func="and",
            geometry_func="union",
        )


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
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        # log.info("Making temp dir {}".format(td))
        temp_fmask_path = os.path.join(td, "fmask.tif")
        apply_fmask(in_l1_dir, temp_fmask_path)
        fmask_image = gdal.Open(temp_fmask_path)
        fmask_array = fmask_image.GetVirtualMemArray()
        out_image = create_matching_dataset(
            fmask_image, out_path, datatype=gdal.GDT_Byte
        )
        out_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
        log.info("fmask created, converting to binary cloud/shadow mask")
        out_array[:, :] = np.isin(fmask_array, (2, 3, 4), invert=True)
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
    if "torque" in os.getenv(
        "PATH"
    ):  # Are we on a HPC? If so, give explicit path to fmask
        fmask_command = "/home/h/hb91/python-fmask/bin/fmask_sentinel2Stacked.py"
    if sys.platform.startswith("win"):
        fmask_command = subprocess.check_output(
            ["where", fmask_command], text=True
        ).strip()
    log = logging.getLogger(__name__)
    fmask_args = [fmask_command, "-o", out_file, "--safedir", in_safe_dir]
    log.info("Creating fmask from {}, output at {}".format(in_safe_dir, out_file))
    fmask_proc = subprocess.Popen(
        fmask_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    while True:
        nextline = fmask_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == "" and fmask_proc.poll() is not None:
            break


def scale_to_uint8(x, percentiles=[0, 100]):
    """
    Scales an array to the range from 0-255 and converts the data type to uint8.
    NaN values will be ignored.

    Args:
      x = input array
      percentiles = list of length 2 of percentiles for trimming the histogram (0-100)

    Returns:
      Scaled array of uint8 data type
    """
    x = np.float32(x)
    amin = np.nanpercentile(x, percentiles[0])
    amax = np.nanpercentile(x, percentiles[1])
    anewmin = 0.0
    anewmax = 255.0
    x[x < amin] = amin
    x[x > amax] = amax
    if amin == amax:
        if amin < 0:
            amin = 0
        if amin > 255:
            amin = 255
        xscaled = np.full_like(x, fill_value=np.uint8(amin), dtype=np.uint8)
    else:
        xscaled = (x - amin) / (amax - amin) * (anewmax - anewmin) + anewmin
    return xscaled.astype(np.uint8)


def create_quicklook(
    in_raster_path,
    out_raster_path,
    width,
    height,
    format="PNG",
    bands=[1, 2, 3],
    nodata=None,
    scale_factors=None,
):
    """
    Creates a quicklook image of reduced size from an input GDAL object and saves it to out_raster_path.

    Parameters
    ----------
    in_raster_path : string
        The string containing the full directory path to the input file for the quicklook
    out_raster_path : string
        The string containing the full directory path to the output file for the quicklook
    width : number
        Width of the output raster in pixels
    height : number
        Height of the output raster in pixels
    format : string, optional
        GDAL format for the quicklook raster file, default PNG
    bands : list of numbers
        List of the band numbers to be displayed as RGB. Will be ignored if only one band is in the image raster.
    nodata : number (optional)
        Missing data value.

    Returns
    -------
    out_raster_path : string
        The output path of the generated quicklook raster file
    """
    # Useful options:
    # widthPct --- width of the output raster in percentage (100 = original width)
    # heightPct --- height of the output raster in percentage (100 = original height)
    # xRes --- output horizontal resolution
    # yRes --- output vertical resolution
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        try:
            image = gdal.Open(in_raster_path, gdal.GA_ReadOnly)
            tmpfile_path = os.path.join(
                td, os.path.basename(in_raster_path)[:-4] + "_copy.tif"
            )
            driver = gdal.GetDriverByName("GTiff")
            driver.CreateCopy(tmpfile_path, image, 0)
            image = None
            image = gdal.Open(tmpfile_path, gdal.GA_Update)
        except RuntimeError as e:
            log.error(
                "Error opening raster file: {}    /   {}".format(in_raster_path, e)
            )
            return
        #TODO: check data type of the in_raster - currently crashes when looking 
        #       at images from the probabilities folder (wrong data type)
        if image.RasterCount < 3:
            # log.info("Raster count is {}. Using band 1.".format(image.RasterCount))
            bands = [image.RasterCount]
            alg = "nearest"
            palette = "rgba"
            band = image.GetRasterBand(1)
            data = band.ReadAsArray()
            scale_factors = None
            output_type = gdal.GDT_Byte
        else:
            alg = None
            palette = None
            output_type = gdal.GDT_Byte
            if scale_factors is None:
                scale_factors = [[0, 2000, 0, 255]]  # this is specific to Sentinel-2
            # log.info("Scaling values from {}...{} to {}...{}".format(scale_factors[0][0], scale_factors[0][1], scale_factors[0][2], scale_factors[0][3]))

        # All the options that gdal.Translate() takes are listed here: gdal.org/python/osgeo.gdal-module.html#TranslateOptions
        kwargs = {
            "format": format,
            "outputType": output_type,
            "bandList": bands,
            "noData": nodata,
            "width": width,
            "height": height,
            "resampleAlg": alg,
            "scaleParams": scale_factors,
            "rgbExpand": palette,
        }

        if image.RasterCount < 3:
            try:
                # histo = np.array(band.GetHistogram())
                # log.info("Histogram: {}".format(np.where(histo > 0)[0]))
                # log.info("           {}".format(histo[np.where(histo > 0)]))
                # log.info("Band data min, max: {}, {}".format(data.min(), data.max()))
                colors = gdal.ColorTable()
                #TODO: load a colour table (QGIS style file) from file if 
                #      specified as an option by the function call
                """
                Comment: A *.qml file contains:
                <colorPalette>
                <paletteEntry label="0" color="#000000" alpha="255" value="0"/>
                <paletteEntry label="1" color="#287d28" alpha="255" value="1"/>
                <paletteEntry label="3" color="#c28540" alpha="255" value="3"/>
                <paletteEntry label="4" color="#e1de0b" alpha="255" value="4"/>
                <paletteEntry label="5" color="#bbdc00" alpha="255" value="5"/>
                <paletteEntry label="11" color="#69de33" alpha="255" value="11"/>
                <paletteEntry label="12" color="#3cd24b" alpha="255" value="12"/>
                </colorPalette>
                """

                if data.max() < 13:
                    log.info("Using custom colour table for up to 12 classes (0..11)")
                    colors.SetColorEntry(0, (0, 0, 0, 0))  # no data
                    colors.SetColorEntry(1, (0, 100, 0, 255))  # Primary Forest
                    colors.SetColorEntry(2, (154, 205, 50, 255))  # plantation Forest
                    colors.SetColorEntry(3, (139, 69, 19, 255))  # Bare Soil
                    colors.SetColorEntry(4, (189, 183, 107, 255))  # Crops
                    colors.SetColorEntry(5, (240, 230, 140, 255))  # Grassland
                    colors.SetColorEntry(6, (0, 0, 205, 255))  # Open Water
                    colors.SetColorEntry(7, (128, 0, 0, 255))  # Burn Scar
                    colors.SetColorEntry(8, (255, 255, 255, 255))  # cloud
                    colors.SetColorEntry(9, (60, 60, 60, 255))  # cloud shadow
                    colors.SetColorEntry(10, (128, 128, 128, 255))  # Haze
                    colors.SetColorEntry(11, (46, 139, 87, 255))  # Open Woodland
                    colors.SetColorEntry(12, (92, 145, 92, 255))  # Toby's Woodland
                else:
                    # log.info("Using viridis colour table for {} classes".format(data.max()))
                    viridis = cm.get_cmap("viridis", min(data.max(), 255))
                    for index, color in enumerate(viridis.colors):
                        colors.SetColorEntry(
                            index,
                            (
                                int(color[0] * 255),
                                int(color[1] * 255),
                                int(color[2] * 255),
                                int(color[3] * 255),
                            ),
                        )
                band.SetRasterColorTable(colors)
                band.WriteArray(data)
                out_image = gdal.Translate(
                    out_raster_path, image, options=gdal.TranslateOptions(**kwargs)
                )
                driver = gdal.GetDriverByName("PNG")
                driver.CreateCopy(out_raster_path, out_image, 0)
                data = None
                out_image = None
                image = None
                band = None
            except Exception as e:
                log.error("An error occurred: {}".format(e))
                log.error("  Skipping quicklook for image: {}".format(out_raster_path))
                image = None
                return
        else:
            out_image = gdal.Translate(
                out_raster_path, image, options=gdal.TranslateOptions(**kwargs)
            )
            out_image = None
            image = None
        return out_raster_path


def __combine_date_maps(date_image_paths, output_product):
    """
    UNTESTED DEVELOPMENT VERSION:
                #TODO: In combine_date_maps, also extract the length of 
                #      consecutive detections over time.
                #TODO: Add a third layer with the length of confirmation sequence
                #      over time.
                #TODO: -1 in the change layers indicates cloudy pixels (Done?)
    Combines all change date layers into one output raster with three layers:
      (1) pixels show the earliest change detection date (expressed as the number of days since 1/1/2000)
      (2) pixels show the number of change detection dates (summed up over all change images in the folder)
      (3) ???

    Parameters
    ----------
    date_image_paths : list of strings
        Containing the full directory paths to the input files with the detection dates as pixel values in UInt32 format
    output_product : string
        The string containing the full directory path to the output file for the 2-layer raster file

    Returns
    -------
    output_product : string
        The string containing the full directory path to the output file for the 2-layer raster file
    """

    log = logging.getLogger(__name__)
    # check which files in the list of input files are not found
    notfound = []
    for path in date_image_paths:
        if not os.path.exists(path):
            log.warning(
                "Change detection image does not exist and will be removed from the report creation: {}".format(
                    path
                )
            )
            notfound = notfound + [path]
    for path in notfound:
        date_image_paths.remove(path)
    # check which files can be opened
    corrupted = []
    for path in date_image_paths:
        try:
            open_file = gdal.Open(path)
            open_file = None
        except:
            log.warning("Cannot open file: {}".format(path))
            corrupted = corrupted + [path]
    for path in corrupted:
        date_image_paths.remove(path)

    if len(date_image_paths) == 0:
        log.warning("No valid input files remain for report image creation.")
        return

    date_images = [gdal.Open(path) for path in date_image_paths]

    # ensure the images have the same map projection
    reference_projection = date_images[0].GetProjection()
    different_projection = []
    for index, date_image in enumerate(date_images):
        projection = date_image.GetProjection()
        if projection != reference_projection:
            log.warning(
                "Skipping image with a different map projection: {} is not the same as {}".format(
                    date_image, date_images[0]
                )
            )
            different_projection = different_projection + date_image
    for image in different_projection:
        date_images.remove(image)
    if len(date_images) == 0:
        log.warning("No valid input files remain for report image creation.")
        date_images = None
        return

    out_raster = create_matching_dataset(
        date_images[0],
        output_product,
        format="GTiff",
        bands=2,
        datatype=gdal.GDT_UInt32,
    )
    # Squeeze() to account for unaccountable extra dimension Windows patch adds
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write).squeeze()
    out_raster_array[:, :, :] = 0  # [bands, y, x]
    reference_projection = date_images[0].GetProjection()
    time_steps = len(date_images)
    for index, date_image in enumerate(date_images):
        #TODO: within this processing loop, update the report layers 
        #      indicating the length of temporal sequences of confirmed values
        projection = date_image.GetProjection()
        if projection != reference_projection:
            log.warning(
                "Skipping image with a different map projection: {} is not the same as {}".format(
                    date_image, date_images[0]
                )
            )
            time_steps = time_steps - 1
            continue
        date_array = date_image.GetVirtualMemArray().squeeze()
        locs = (out_raster_array[0, :, :] > 0) & (date_array > 0)
        out_raster_array[0, locs] = np.minimum(
            out_raster_array[0, locs], date_array[locs]
        )
        locs = (out_raster_array[0, :, :] == 0) & (date_array > 0)
        out_raster_array[0, locs] = date_array[locs]
        date_mask = np.where(date_array > 0, 1, 0)
        # log.info("Types: {} and {}".format(type(out_raster_array[1, 0, 0]), type(date_mask[0,0]))
        out_raster_array[1, :, :] = np.add(out_raster_array[1, :, :], date_mask)
        date_array = None
        date_mask = None

    """
    REMOVE THIS
    # Access data cube as an array
    data_cube = np.zero(shape = (time_steps, y, x)) #[time, y, x]
    for index, date_image in enumerate(date_images):
        projection = date_image.GetProjection()
        if projection != reference_projection:
            continue
        data_cube[index, :, :] = date_image.GetVirtualMemArray().squeeze()

    segment_length = 4
    log.info("Finding temporal segments with at least {} subsequent changes, not counting cloud covered times".format(segment_length))
    for y in range(np.shape(data_cube)[1]):
        for x in range(np.shape(data_cube)[2]):
            time_slice = data_cube[:,y,x][data_cube[:,y,x] > -1]
            result = np.where((v == 1) & (np.roll(v,-1) == 2))[0] # work to be done here
            if len(result) > 0:
                print(i, result[0])
    date_mask = np.where(data_cube > 0, 1, 0) #not right yet
    out_raster_array[2, :, :] = np.add(out_raster_array[1, :, :], date_mask)
    data_cube = None
    """

    out_raster_array = None
    out_raster = None
    date_images = None
    return output_product


def combine_date_maps(date_image_paths, output_product):
    """
    Combines all change date layers into one output raster with two layers:
      (1) pixels show the earliest change detection date (expressed as the number of days since 1/1/2000)
      (2) pixels show the number of change detection dates (summed up over all change images in the folder)

    Parameters
    ----------
    date_image_paths : list of strings
        Containing the full directory paths to the input files with the detection dates as pixel values in UInt32 format
    output_product : string
        The string containing the full directory path to the output file for the 2-layer raster file

    Returns
    -------
    output_product : string
        The string containing the full directory path to the output file for the 2-layer raster file
    """

    log = logging.getLogger(__name__)
    # check which files in the list of input files are not found
    notfound = []
    for path in date_image_paths:
        if not os.path.exists(path):
            log.warning(
                "Change detection image does not exist and will be removed from the report creation: {}".format(
                    path
                )
            )
            notfound = notfound + [path]
    for path in notfound:
        date_image_paths.remove(path)
    # check which files can be opened
    corrupted = []
    for path in date_image_paths:
        try:
            open_file = gdal.Open(path)
            open_file = None
        except:
            log.warning("Cannot open file: {}".format(path))
            corrupted = corrupted + [path]
    for path in corrupted:
        date_image_paths.remove(path)

    if len(date_image_paths) == 0:
        log.warning("No valid input files remain for report image creation.")
        return

    date_images = [gdal.Open(path) for path in date_image_paths]

    # ensure the images have the same map projection
    reference_projection = date_images[0].GetProjection()
    different_projection = []
    for index, date_image in enumerate(date_images):
        projection = date_image.GetProjection()
        if projection != reference_projection:
            log.warning(
                "Skipping image with a different map projection: {} is not the same as {}".format(
                    date_image, date_images[0]
                )
            )
            different_projection = different_projection + date_image
    for image in different_projection:
        date_images.remove(image)
    if len(date_images) == 0:
        log.warning("No valid input files remain for report image creation.")
        date_images = None
        return

    out_raster = create_matching_dataset(
        date_images[0],
        output_product,
        format="GTiff",
        bands=2,
        datatype=gdal.GDT_UInt32,
    )
    # Squeeze() to account for unaccountable extra dimension Windows patch adds
    out_raster_array = out_raster.GetVirtualMemArray(eAccess=gdal.GF_Write).squeeze()
    out_raster_array[:, :, :] = 0  # [bands, y, x]
    reference_projection = date_images[0].GetProjection()
    for index, date_image in enumerate(date_images):
        projection = date_image.GetProjection()
        if projection != reference_projection:
            log.warning(
                "Skipping image with a different map projection: {} is not the same as {}".format(
                    date_image, date_images[0]
                )
            )
            continue
        date_array = date_image.GetVirtualMemArray().squeeze()
        locs = (out_raster_array[0, :, :] > 0) & (date_array > 0)
        out_raster_array[0, locs] = np.minimum(
            out_raster_array[0, locs], date_array[locs]
        )
        locs = (out_raster_array[0, :, :] == 0) & (date_array > 0)
        out_raster_array[0, locs] = date_array[locs]
        date_mask = np.where(date_array > 0, 1, 0)
        # log.info("Types: {} and {}".format(type(out_raster_array[1, 0, 0]), type(date_mask[0,0]))
        out_raster_array[1, :, :] = np.add(out_raster_array[1, :, :], date_mask)
        date_array = None
        date_mask = None
    out_raster_array = None
    out_raster = None
    date_images = None
    return output_product


def sieve_image(image_path, out_path, neighbours=8, sieve=10, skip_existing=False):
    """
    Sieves a class image using gdal. Output is saved out_path.

    Parameters
    ----------
    image_path : str
        The path to the class image file with a single band.
    out_path : str
        The path to the output file that will store the sieved class image.
    neighbours : int
        Number of neighbouring pixels at sieve stage. Can be 4 or 8.
    sieve : int
        Number of pixels in a class polygon. Only polygons below this threshold will be removed. See GDAL Sieve documentation.
    skip_existing : boolean, optional
        If True, skips the classification if the output file already exists.
    """

    if neighbours != 4 and neighbours != 8:
        log.warning("Invalid neighbour connectedness for sieve. Changing value to 4.")
        neighbours = 4
    if os.path.exists(out_path) and skip_existing:
        log.info("File exists. Skipping sieve stage. {}".format(out_path))
        return
    try:
        image = gdal.Open(image_path)
        out_image = create_matching_dataset(image, out_path, format="GTiff")
        in_band = image.GetRasterBand(1)
        out_band = out_image.GetRasterBand(1)
        gdal.SieveFilter(
            srcBand=in_band,
            maskBand=None,
            dstBand=out_band,
            threshold=sieve,
            connectedness=neighbours,
            callback=gdal.TermProgress_nocb,
        )
        in_band = None
        out_band = None
        image = None
    except:
        log.warning(
            "Could not open file for sieve filtering. Skipping. {}".format(image_path)
        )
    return


def sieve_directory(
    in_dir, out_dir=None, neighbours=8, sieve=10, out_type="GTiff", skip_existing=False
):
    """
    Sieves all class images ending in .tif in in_dir using gdal.
    Outputs are saved out_dir.

    Parameters
    ----------
    in_dir : str
        The path to the directory containing the class image files.
    out_dir : str
        The directory that will store the sieved class image files
    neighbours : int
        Number of neighbouring pixels at sieve stage. Can be 4 or 8.
    sieve : int
        Number of pixels in a class polygon. Only polygons below this threshold will be removed. See GDAL Sieve documentation.
    out_type : str, optional
        The raster format of the class image. Defaults to "GTiff" (geotif). See gdal docs for valid datatypes.
    skip_existing : boolean, optional
        If True, skips the classification if the output file already exists.

    Returns:
    --------
    out_image_paths : list of str
        A list with all paths to the sieved class image files, including those that already existed.
    """

    log = logging.getLogger(__name__)
    log.info("Sieving class files in      {}".format(in_dir))
    log.info("Sieved class files saved in {}".format(out_dir))
    log.info("Neighbours = {}   Sieve = {}".format(neighbours, sieve))
    log.info("Skip existing files? {}".format(skip_existing))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_image_paths = []
    for image_path in glob.glob(in_dir + r"/*.tif"):
        image_name = os.path.basename(image_path)[:-4] + "_sieved.tif"
        out_path = os.path.join(out_dir, image_name)
        out_image_paths = out_image_paths + [out_path]
        sieve_image(
            image_path=image_path,
            out_path=out_path,
            neighbours=neighbours,
            sieve=sieve,
            skip_existing=skip_existing,
        )
    return out_image_paths


def compress_tiff(in_path: str, 
                  out_path: str,
                  log: logging.Logger
                  ):
    """
    LZW-compresses a Geotiff file using gdal if not already done.

    Parameters
    ----------
    in_path : str
        The path to the input GeoTiff file.
    out_path : str
        The path to the output GeoTiff file.
    log : logging.Logger
        Logging output
    """
    
    dataset = gdal.OpenEx(in_path)
    md = dataset.GetMetadata('IMAGE_STRUCTURE')
    compression = md.get('COMPRESSION', None)
    if compression == 'LZW':
        log.info(f"GeoTiff file is already LZW compressed: {in_path}")
        return
    log.info(f"Compressing GeoTiff file: {in_path}")
    with TemporaryDirectory(dir=os.path.expanduser('~')) as td:
        try:
            tmp_path = os.path.join(td, "tmp_compressed.tif")
            translateoptions = gdal.TranslateOptions(
                gdal.ParseCommandLine("-of Gtiff -co COMPRESS=LZW")
            )
            gdal.Translate(tmp_path, in_path, options=translateoptions)
            shutil.move(tmp_path, out_path)
        except RuntimeError as e:
            log.error(f"Error opening GeoTiff file: {in_path}")
            log.error(f"  {e}")
    return


def write_n_band_tiff(tiff_paths: list, output_path: str) -> None:
    """
    This function takes a list of tiff paths and merges the tiffs into a single n-band image.

    Parameters
    ----------
    tiff_paths : list
        List of tiff input images to merge
    output_path : str
        The path to the output GeoTiff file.
    """

    reference = gdal.Open(tiff_paths[0])
    num_rows = reference.RasterYSize
    num_cols = reference.RasterXSize
    num_bands = len(tiff_paths)

    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(output_path, num_cols, num_rows, num_bands, gdal.GDT_Byte)

    for band_num, input_path in enumerate(tiff_paths):
        input_image = gdal.Open(input_path)

        band = input_image.GetRasterBand(1)
        band_data = band.ReadAsArray()

        output_band = output.GetRasterBand(band_num + 1)
        output_band.WriteArray(band_data)

        output_band.SetColorInterpretation(band.GetColorInterpretation())

        output_band.SetNoDataValue(band.GetNoDataValue())

        input_image = None

    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())

    output = None

    log.info(f"Successfully merged {num_bands} TIFF images into {output_path}.")

    return
