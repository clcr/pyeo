"""
Functions for managing and manipulating sentinel-2 .SAFE files
"""

import datetime as dt
import glob
import logging
import os
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import gdal
from osgeo import osr

from pyeo.filesystem_utilities import check_for_invalid_l2_data
from pyeo.masks import create_mask_from_sen2cor_and_fmask, get_mask_path, combine_masks
from pyeo.raster_manipulation import stack_images, reproject_image
from pyeo.exceptions import CreateNewStacksException, BadS2Exception


def get_sen_2_tiles(image_dir):
    """
    gets the list of tiles present in the directory
    """
    image_files = glob.glob(os.path.join(image_dir, "*.tif"))
    if len(image_files) == 0:
        raise CreateNewStacksException("Image_dir is empty")
    else:
        tiles = []
        for image_file in image_files:
            tile = get_sen_2_image_tile(image_file)
            tiles.append(tile)
    return tiles


def get_image_acquisition_time(image_name):
    """Gets the datetime object from a .safe filename of a planet image. No test. Returns None if no timestamp present"""
    try:
        return dt.datetime.strptime(get_sen_2_image_timestamp(image_name), '%Y%m%dT%H%M%S')
    except AttributeError:
        return None


def get_related_images(target_image_name, project_dir):
    """Gets the paths of all images related to that one in a project, by timestamp"""
    timestamp = get_sen_2_image_timestamp(target_image_name)
    image_glob = r"*{}*".format(timestamp)
    return glob.glob(image_glob, project_dir)


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


def get_l1_safe_file(image_name, l1_dir):
    """Returns the path to the L1 .SAFE directory of image. Gets from granule and timestamp. image_name can be a path or
    a filename"""
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL1C_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l1_dir, safe_glob))[0]
    return out


def get_l2_safe_file(image_name, l2_dir):
    """Returns the path to the L2 .SAFE directory of image. Gets from granule and timestamp. image_name can be a path or
    a filename"""
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l2_dir, safe_glob))[0]
    return out


def get_sen_2_image_timestamp(image_name):
    """Returns the timestamps part of a Sentinel 2 image"""
    timestamp_re = r"\d{8}T\d{6}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def get_sen_2_image_orbit(image_name):
    """Returns the relative orbit number of a Sentinel 2 image"""
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[4]


def get_sen_2_image_tile(image_name):
    """Returns the tile number of a Sentinel 2 image or path"""
    name = os.path.basename(image_name)
    tile = re.findall(r"T\d{2}[A-Z]{3}", name)[0]  # Matches tile ID, but not timestamp
    return tile


def get_sen_2_granule_id(safe_dir):
    """Returns the unique ID of a Sentinel 2 granule from a SAFE directory path"""
    tmp = os.path.basename(safe_dir) # removes path to SAFE directory
    id = tmp.split(".")[0] # removes ".SAFE" from the ID name
    return id


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