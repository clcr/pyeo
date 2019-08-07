"""
pyeo.raster_manipulation
------------------------
Functions for working with raster data.
"""

import glob
import logging
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory

import gdal
import numpy as np
from osgeo import gdal_array, osr, ogr
from skimage import morphology as morph

from pyeo.coordinate_manipulation import get_combined_polygon, pixel_bounds_from_polygon, write_geometry, \
    get_aoi_intersection, get_raster_bounds, align_bounds_to_whole_number, get_poly_bounding_rect
from pyeo.array_utilities import project_array
from pyeo.filesystem_utilities import sort_by_timestamp, get_sen_2_tiles, get_l1_safe_file, get_sen_2_image_timestamp, \
    get_sen_2_image_tile, get_sen_2_granule_id, check_for_invalid_l2_data, get_mask_path
from pyeo.exceptions import CreateNewStacksException, StackImagesException, BadS2Exception


def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype = None):
    """Creates an empty gdal dataset with the same dimensions, projection and geotransform. Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified"""
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
    """Saves a given array as a geospatial image in the format 'format'
    Array must be gdal format: [bands, y, x]. Returns the gdal object"""
    driver = gdal.GetDriverByName(format)
    type_code = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    out_dataset = driver.Create(
        path,
        xsize=array.shape[2],
        ysize=array.shape[1],
        bands=array.shape[0],
        eType=type_code
    )
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)
    out_array = out_dataset.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_array[...] = array
    out_array = None
    out_dataset = None
    return path


def create_new_stacks(image_dir, stack_dir):
    """
    Creates new stacks with with adjacent image acquisition dates. Threshold; how small a part
    of the latest_image will be before it's considered to be fully processed.
    New_image_name must exist inside image_dir.

    Step 1: Sort directory as follows:
            Relative Orbit number (RO4O), then Tile Number (T15PXT), then
            Datatake sensing start date (YYYYMMDD) and time(THHMMSS).
            newest first.
    Step 2: For each tile number:
            new_data_polygon = bounds(new_image_name)
    Step 3: For each tiff image coverring that tile, work backwards in time:
            a. Check if it intersects new_data_polygon
            b. If it does
               - add to a to_be_stacked list,
               - subtract it's bounding box from new_data_polygon.
            c. If new_data_polygon drops having a total area less than threshold, stop.
    Step 4: Stack new rasters for each tile in new_data list.

    """
    log = logging.getLogger(__name__)
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
            latest_image_path = safe_files[0]
            for image in safe_files[1:]:
                new_images.append(stack_old_and_new_images(image, latest_image_path, stack_dir))
                latest_image_path = image
    return new_images


def stack_image_with_composite(image_path, composite_path, out_dir, create_combined_mask=True, skip_if_exists=True,
                               invert_stack = False):
    """Stacks an image with a cloud-free composite"""
    log = logging.getLogger(__name__)
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
    """Stacks multiple images in image_paths together, using the information of the top image.
    geometry_mode can be "union" or "intersect" """
    log = logging.getLogger(__name__)
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


def trim_image(in_raster_path, out_raster_path, polygon, format="GTiff"):
    """Trims image to polygon"""
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

    TODO: consider using GDAL:

    gdal_merge.py [-o out_filename] [-of out_format] [-co NAME=VALUE]*
              [-ps pixelsize_x pixelsize_y] [-tap] [-separate] [-q] [-v] [-pct]
              [-ul_lr ulx uly lrx lry] [-init "value [value...]"]
              [-n nodata_value] [-a_nodata output_nodata_value]
              [-ot datatype] [-createonly] input_files
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
    """Works down in_raster_path_list, updating pixels in composite_out_path if not masked. Masks are assumed to
    be a binary .msk file with the same path as their corresponding image. All images must have the same
    number of layers and resolution, but do not have to be perfectly on top of each other. If it does not exist,
    composite_out_path will be created. Takes projection, resolution, ect from first band of first raster in list.
    Will reproject images and masks if they do not match initial raster."""

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
        dates_array = dates_image.GetVirtualMemArray(eAccess=gdal.gdalconst.GF_Write)

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
    """Reprojects every file ending with extension to new_projection and saves in out_dir"""
    log = logging.getLogger(__name__)
    image_paths = [os.path.join(in_dir, image_path) for image_path in os.listdir(in_dir) if image_path.endswith(extension)]
    for image_path in image_paths:
        reproj_path = os.path.join(out_dir, os.path.basename(image_path))
        log.info("Reprojecting {} to {}, storing in {}".format(image_path, reproj_path, new_projection))
        reproject_image(image_path, reproj_path, new_projection)


def reproject_image(in_raster, out_raster_path, new_projection,  driver = "GTiff",  memory = 2e3, do_post_resample=True):
    """Creates a new, reprojected image from in_raster. Wraps gdal.ReprojectImage function. Will round projection
    back to whatever 2gb memory limit by default (because it works in most places)"""
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
    """Composites every image in image_dir, assumes all have associated masks.  Will
     place a file named composite_[last image date].tif inside composite_out_dir"""
    log = logging.getLogger(__name__)
    log.info("Compositing {}".format(image_dir))
    sorted_image_paths = [os.path.join(image_dir, image_name) for image_name
                          in sort_by_timestamp(os.listdir(image_dir), recent_first=False)  # Let's think about this
                          if image_name.endswith(".tif")]
    last_timestamp = get_sen_2_image_timestamp(os.path.basename(sorted_image_paths[-1]))
    composite_out_path = os.path.join(composite_out_dir, "composite_{}.tif".format(last_timestamp))
    composite_images_with_mask(sorted_image_paths, composite_out_path, format, generate_date_image=generate_date_images)


def flatten_probability_image(prob_image, out_path):
    """Produces a single-band raster containing the highest certainties in a input probablility raster"""
    prob_raster = gdal.Open(prob_image)
    out_raster = create_matching_dataset(prob_raster, out_path, bands=1)
    prob_array = prob_raster.GetVirtualMemArray()
    out_array = out_raster.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_array[:, :] = prob_array.max(axis=0)
    out_array = None
    prob_array = None
    out_raster = None
    prob_raster = None


def get_masked_array(raster, mask_path, fill_value = -9999):
    """Returns a numpy.mask masked array for the raster.
    Masked pixels are FALSE in the mask image (multiplicateive map),
    but TRUE in the masked_array (nodata pixels)"""
    mask = gdal.Open(mask_path)
    mask_array = mask.GetVirtualMemArray()
    raster_array = raster.GetVirtualMemArray()
    # If the shapes do not match, assume single-band mask for multi-band raster
    if len(mask_array.shape) == 2 and len(raster_array.shape) == 3:
        mask_array = project_array(mask_array, raster_array.shape[0], 0)
    return np.ma.array(raster_array, mask=np.logical_not(mask_array))


def stack_and_trim_images(old_image_path, new_image_path, aoi_path, out_image):
    """Stacks an old and new S2 image and trims to within an aoi"""
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


def clip_raster(raster_path, aoi_path, out_path, srs_id=4326):
    """Clips a raster at raster_path to a shapefile given by aoi_path. Assumes a shapefile only has one polygon.
    Will np.floor() when converting from geo to pixel units and np.absolute() y resolution form geotransform."""
    # https://gis.stackexchange.com/questions/257257/how-to-use-gdal-warp-cutline-option
    with TemporaryDirectory() as td:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(srs_id)
        intersection_path = os.path.join(td, 'intersection')
        raster = gdal.Open(raster_path)
        in_gt = raster.GetGeoTransform()
        aoi = ogr.Open(aoi_path)
        intersection = get_aoi_intersection(raster, aoi)
        min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
        width_pix = int(np.floor(max_x_geo - min_x_geo)/in_gt[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo)/np.absolute(in_gt[5]))
        new_geotransform = (min_x_geo, in_gt[1], 0, min_y_geo, 0, in_gt[5])
        write_geometry(intersection, intersection_path)
        clip_spec = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=intersection_path,
            cropToCutline=True,
            width=width_pix,
            height=height_pix,
            srcSRS=srs,
            dstSRS=srs
        )
        out = gdal.Warp(out_path, raster, options=clip_spec)
        out.SetGeoTransform(new_geotransform)
        out = None


def create_new_image_from_polygon(polygon, out_path, x_res, y_res, bands,
                           projection, format="GTiff", datatype = gdal.GDT_Int32, nodata = -9999):
    """Returns an empty image of the extent of input polygon"""
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
    """Resamples an image in-place using gdalwarp to new_res in metres"""
    # I don't like using a second object here, but hey.
    with TemporaryDirectory() as td:
        args = gdal.WarpOptions(
            xRes=new_res,
            yRes=new_res
        )
        temp_image = os.path.join(td, "temp_image.tif")
        gdal.Warp(temp_image, image_path, options=args)
        shutil.move(temp_image, image_path)


def raster_to_array(rst_pth):
    """Reads in a raster file and returns a N-dimensional array.

    :param str rst_pth: Path to input raster.
    :return: N-dimensional array.
    """
    log = logging.getLogger(__name__)
    in_ds = gdal.Open(rst_pth)
    out_array = in_ds.ReadAsArray()

    return out_array


def raster_sum(inRstList, outFn, outFmt='GTiff'):
    """Creates a raster stack from a list of rasters. Adapted from Chris Gerard's
    book 'Geoprocessing with Python'. The out put data type is the same as the input data type.

    :param str inRstList: List of rasters to stack.
    :param str outFmt: String specifying the input data format e.g. 'GTiff' or 'VRT'.
    :param str outFn: Filename output as str including directory else image will be
    written to current working directory.

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
    """Filters class_map_path for pixels in filter_map_path containing only classes_of_interest.
    Assumes that filter_map_path and class_map_path are same resolution and projection."""
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
    """Opens a dataset given a safe file. Give band as a string."""
    image_glob = r"GRANULE/*/IMG_DATA/R{}/*_{}_{}.jp2".format(resolution, band, resolution)
    # edited by hb91
    #image_glob = r"GRANULE/*/IMG_DATA/*_{}.jp2".format(band)
    fp_glob = os.path.join(safe_file_path, image_glob)
    image_file_path = glob.glob(fp_glob)
    out = gdal.Open(image_file_path[0])
    return out


def preprocess_sen2_images(l2_dir, out_dir, l1_dir, cloud_threshold=60, buffer_size=0, epsg=None):
    """For every .SAFE folder in in_dir, stacks band 2,3,4 and 8  bands into a single geotif, creates a cloudmask from
    the combined fmask and sen2cor cloudmasks and reprojects to a given EPSG if provided"""
    log = logging.getLogger(__name__)
    safe_file_path_list = [os.path.join(l2_dir, safe_file_path) for safe_file_path in os.listdir(l2_dir)]
    for l2_safe_file in safe_file_path_list:
        with TemporaryDirectory() as temp_dir:
            log.info("----------------------------------------------------")
            log.info("Merging 10m bands in SAFE dir: {}".format(l2_safe_file))
            temp_path = os.path.join(temp_dir, get_sen_2_granule_id(l2_safe_file)) + ".tif"
            log.info("Output file: {}".format(temp_path))
            stack_sentinel_2_bands(l2_safe_file, temp_path, band='10m')

            #pdb.set_trace()

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
            else:
                log.info("Moving images to {}".format(out_dir))
                shutil.move(temp_path, out_path)
                shutil.move(mask_path, out_mask_path)


def stack_sentinel_2_bands(safe_dir, out_image_path, band = "10m"):
    """Stacks the contents of a .SAFE granule directory into a single geotiff"""
    log = logging.getLogger(__name__)
    granule_path = r"GRANULE/*/IMG_DATA/R{}/*_B0[8,4,3,2]_{}.jp2".format(band, band)
    image_glob = os.path.join(safe_dir, granule_path)
    file_list = glob.glob(image_glob)
    file_list.sort()   # Sorting alphabetically gives the right order for bands
    if not file_list:
        log.error("No 10m imagery present in {}".format(safe_dir))
        raise BadS2Exception
    stack_images(file_list, out_image_path, geometry_mode="intersect")
    return out_image_path


def stack_old_and_new_images(old_image_path, new_image_path, out_dir, create_combined_mask=True):
    """
    Stacks two images with the same tile
    Names the result with the two timestamps.
    First, decompose the granule ID into its components:
    e.g. S2A, MSIL2A, 20180301, T162211, N0206, R040, T15PXT, 20180301, T194348
    are the mission ID(S2A/S2B), product level(L2A), datatake sensing start date (YYYYMMDD) and time(THHMMSS),
    the Processing Baseline number (N0206), Relative Orbit number (RO4O), Tile Number field (T15PXT),
    followed by processing run date and then time
    """
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
    """Applies sen2cor to the SAFE file at image_path. Returns the path to the new product."""
    # Here be OS magic. Since sen2cor runs in its own process, Python has to spin around and wait
    # for it; since it's doing that, it may as well be logging the output from sen2cor. This
    # approach can be multithreaded in future to process multiple image (1 per core) but that
    # will take some work to make sure they all finish before the program moves on.
    log = logging.getLogger(__name__)
    # added sen2cor_path by hb91
    log.info("calling subprocess: {}".format([sen2cor_path, image_path]))
    sen2cor_proc = subprocess.Popen([sen2cor_path, image_path],
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
    if not check_for_invalid_l2_data(image_path.replace("MSIL1C", "MSIL2A")):
        log.error("10m imagery not present in {}".format(image_path.replace("MSIL1C", "MSIL2A")))
        raise BadS2Exception
    if delete_unprocessed_image:
            log.info("removing {}".format(image_path))
            shutil.rmtree(image_path)
    return image_path.replace("MSIL1C", "MSIL2A")


def atmospheric_correction(in_directory, out_directory, sen2cor_path, delete_unprocessed_image=False):
    """Applies Sen2cor cloud correction to level 1C images"""
    log = logging.getLogger(__name__)
    images = [image for image in os.listdir(in_directory)
              if image.startswith('MSIL1C', 4)]
    # Opportunity for multithreading here
    for image in images:
        log.info("Atmospheric correction of {}".format(image))
        image_path = os.path.join(in_directory, image)
        #image_timestamp = get_sen_2_image_timestamp(image)
        if glob.glob(os.path.join(out_directory, image.replace("MSIL1C", "MSIL2A"))):
            log.warning("{} exists. Skipping.".format(image.replace("MSIL1C", "MSIL2A")))
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
    """Returns a multiplicative mask (0 for cloud, shadow or haze, 1 for clear) built from the model at model_path."""
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
    """Creates a multiplicative binary mask where cloudy pixels are 0 and non-cloudy pixels are 1. If
    cloud_conf_threshold = 0, use scl mask else use confidence image """
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
    """Creates a mask from a classification mask: 1 for each pixel containing one of classes_of_interest, otherwise 0"""
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
        buffer_mask_in_place(out_path)
    return out_path


def combine_masks(mask_paths, out_path, combination_func = 'and', geometry_func ="intersect"):
    """ORs or ANDs several masks. Gets metadata from top mask. Assumes that masks are a
    Python true or false. Also assumes that all masks are the same projection for now."""
    # TODO Implement intersection and union
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
    out_mask_array[:, :] = 1
    for i, in_mask in enumerate(masks):
        in_mask_array = in_mask.GetVirtualMemArray()
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
        if i is 0:
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
    """Expands a mask in-place, overwriting the previous mask"""
    log = logging.getLogger(__name__)
    log.info("Buffering {} with buffer size {}".format(mask_path, buffer_size))
    mask = gdal.Open(mask_path, gdal.GA_Update)
    mask_array = mask.GetVirtualMemArray(eAccess=gdal.GA_Update)
    cache = morph.binary_erosion(mask_array, selem=morph.disk(buffer_size))
    np.copyto(mask_array, cache)
    mask_array = None
    mask = None


def apply_array_image_mask(array, mask, fill_value=0):
    """Applies a mask of (y,x) to an image array of (bands, y, x). Replaces any masked pixels with fill_value
    Mask is an a 2 dimensional array of 1 ( unmasked) and 0 (masked)"""
    stacked_mask = np.broadcast_to(mask, array.shape)
    return np.where(stacked_mask == 1, array, fill_value)


def create_mask_from_sen2cor_and_fmask(l1_safe_file, l2_safe_file, out_mask_path, buffer_size=0):
    with TemporaryDirectory() as td:
        s2c_mask_path = os.path.join(td, "s2_mask.tif")
        fmask_mask_path = os.path.join(td, "fmask.tif")
        create_mask_from_confidence_layer(l2_safe_file, s2c_mask_path, buffer_size=buffer_size)
        create_mask_from_fmask(l1_safe_file, fmask_mask_path)
        combine_masks([s2c_mask_path, fmask_mask_path], out_mask_path, combination_func="and", geometry_func="union")


def create_mask_from_fmask(in_l1_dir, out_path):
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
    """Calls fmask to create a new mask for L1 data"""
    # For reasons known only to the spirits, calling subprocess.run from within this function on a HPC cause the PATH
    # to be prepended with a Windows "eoenv\Library\bin;" that breaks the environment. What follows is a large kludge.
    if "torque" in os.getenv("PATH"):  # Are we on a HPC? If so, give explicit path to fmask
        fmask_command = "/data/clcr/shared/miniconda3/envs/eoenv/bin/fmask_sentinel2Stacked.py"
    log = logging.getLogger(__name__)
    args = [
        fmask_command,
        "-o", out_file,
        "--safedir", in_safe_dir
    ]
    log.info("Creating fmask from {}, output at {}".format(in_safe_dir, out_file))
    # pdb.set_trace()
    fmask_proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    while True:
        nextline = fmask_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and fmask_proc.poll() is not None:
            break
