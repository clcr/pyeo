import glob
import logging
import os
import shutil
from tempfile import TemporaryDirectory

import gdal
import numpy as np
from osgeo import gdal_array, osr, ogr

from pyeo.masks import get_mask_path, combine_masks, create_mask_from_class_map, apply_array_image_mask
from pyeo.coordinate_manipulation import get_combined_polygon, pixel_bounds_from_polygon, write_geometry, \
    get_aoi_intersection, get_raster_bounds, align_bounds_to_whole_number, get_poly_bounding_rect
from pyeo.array_utilities import project_array
from pyeo.sen2_funcs import get_sen_2_tiles, stack_old_and_new_images, get_sen_2_image_timestamp, get_sen_2_image_tile
from pyeo.filesystem_utilities import sort_by_timestamp
from pyeo.exceptions import CreateNewStacksException, StackImagesException


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


def reproject_image(in_raster, out_raster_path, new_projection, driver = "GTiff",  memory = 2e3):
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
    # TODO: put into proper place in core
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