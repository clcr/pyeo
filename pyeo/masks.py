"""
Contains functions for creating, manipulating and applying masks.
Unless otherwise stated, each mask is a binary multiplicative mask; pixels with the value 0 are masked and pixels
with the value 1 are not masked.
"""

import glob
import logging
import os
from tempfile import TemporaryDirectory

import gdal
import numpy as np
from skimage import morphology as morph

from pyeo.coordinate_manipulation import align_bounds_to_whole_number, get_combined_polygon, pixel_bounds_from_polygon, \
    get_raster_bounds
from pyeo.classification import classify_image
from pyeo.sen2_funcs import apply_fmask
from pyeo.raster_manipulation import create_matching_dataset, create_new_image_from_polygon, resample_image_in_place


def create_mask_from_model(image_path, model_path, model_clear=0, num_chunks=10, buffer_size=0):
    """Returns a multiplicative mask (0 for cloud, shadow or haze, 1 for clear) built from the model at model_path."""
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


def create_mask_from_fmask(in_l1_dir, out_path):
    log = logging.getLogger(__name__)
    log.info("Creating fmask for {}".format(in_l1_dir))
    with TemporaryDirectory() as td:
        #pdb.set_trace()
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


def create_mask_from_sen2cor_and_fmask(l1_safe_file, l2_safe_file, out_mask_path, buffer_size=0):
    with TemporaryDirectory() as td:
        s2c_mask_path = os.path.join(td, "s2_mask.tif")
        fmask_mask_path = os.path.join(td, "fmask.tif")
        create_mask_from_confidence_layer(l2_safe_file, s2c_mask_path, buffer_size=buffer_size)
        create_mask_from_fmask(l1_safe_file, fmask_mask_path)
        combine_masks([s2c_mask_path, fmask_mask_path], out_mask_path, combination_func="and", geometry_func="union")


def get_mask_path(image_path):
    """A gdal mask is an image with the same name as the image it's masking, but with a .msk extension"""
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    mask_name = image_name.rsplit('.')[0] + ".msk"
    mask_path = os.path.join(image_dir, mask_name)
    return mask_path


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