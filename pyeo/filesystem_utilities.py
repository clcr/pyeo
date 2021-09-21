"""
pyeo.filesystem_utilities
=========================
Contains functions for sorting, creating and comparing images as part of the filesystem. Includes any function
that works on a generic pyeo timestamp and sentinel 2 functions.

Key functions
-------------

:py:func:`init_log` Sets up logging to both console and file

:py:func:`create_file_structure` Creates a recommended file structure for automated work

:py:func:`sort_by_timestamp` Sorts a set of files by timestamp

Function reference
-------------
"""

import datetime
import datetime as dt
import glob
import logging
import os
import re
import shutil

from pyeo.exceptions import CreateNewStacksException


import pyeo.windows_compatability

# Set up logging on import
log = logging.getLogger("pyeo")
formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")


def init_log(log_path):
    """
    Sets up the log format and log handlers; one for stdout and to write to a file, 'log_path'.
    Returns the log for the calling script.
    Parameters
    ----------
    log_path : str
        The path to the file output of this log.

    Returns
    -------
    log : logging.Logger
        The logging object.

    """
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    log = logging.getLogger("pyeo")
    log.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info("****PROCESSING START****")
    return log


def create_file_structure(root):
    """
    Creates the folder structure used in rolling_s2_composite and some other functions: ::
        root
        -images
        --L1C
        --L2A
        --bandmerged
        --stacked
        -composite
        --L1C
        --L2A
        --bandmerged
        -output
        --classified
        --probabilities

    Parameters
    ----------
    root : str
        The root folder for the file strucutre
    """
    os.chdir(root)
    dirs = [
        "images/",
        "images/L1C/",
        "images/L2A/",
        "images/bandmerged/",
        "images/stacked/",
        "images/planet/",
        "composite/",
        "composite/L1C",
        "composite/L2A",
        "composite/bandmerged",
        "output/",
        "output/classified",
        "output/probabilities",
        "output/report_image",
        "output/display_images",
        "log/"
    ]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass


def validate_config_file(config_path):
    #TODO: fill
    pass


# What was I thinking with these two functions?
def check_for_invalid_l2_data(l2_SAFE_file, resolution="10m"):
    """
    Checks the existence of the specified resolution of imagery. Returns a True-value with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.

    Parameters
    ----------
    l2_SAFE_file : str
        Path to the L2A file to check
    resolution : {"10m", "20m", "60m"}
        The resolution of imagery to check. Defaults to 10m.

    Returns
    -------
    result : int
       1 if imagery is valid, 0 if not and 2 if an invalid .SAFE file

    """
    if not l2_SAFE_file.endswith(".SAFE") or "L2A" not in l2_SAFE_file:
        log.info("{} does not exist.".format(l2_SAFE_file))
        return 2
    log.info("Checking {} for incomplete {} imagery".format(l2_SAFE_file, resolution))
    granule_path = r"GRANULE/*/IMG_DATA/R{}/*_B0[8,4,3,2]*.jp2".format(resolution)
    image_glob = os.path.join(l2_SAFE_file, granule_path)
    if len(glob.glob(image_glob)) == 4:
        return 1
    else:
        return 0


def check_for_invalid_l1_data(l1_SAFE_file):
    """
    Checks the existance of the specified resolution of imagery. Returns True with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.

    Parameters
    ----------
    l1_SAFE_file : str
        Path to the L1 file to check

    Returns
    -------
    result : int
        1 if imagery is valid, 0 if not and 2 if not a safe-file
    """
    if not l1_SAFE_file.endswith(".SAFE") or "L1C" not in l1_SAFE_file:
        log.info("{} does not exist.".format(l1_SAFE_file))
        return 2
    log.info("Checking {} for incomplete imagery".format(l1_SAFE_file))
    granule_path = r"GRANULE/*/IMG_DATA/*_B0[8,4,3,2]*.jp2"
    image_glob = os.path.join(l1_SAFE_file, granule_path)
    if len(glob.glob(image_glob)) == 4:
        return 1
    else:
        return 0


def clean_l2_data(l2_SAFE_file, resolution="10m", warning=True):
    """
    Removes a safe file if it doesn't have bands 2, 3, 4 or 8 in the specified resolution folder.
    If warning=True, prompts before removal.

    Parameters
    ----------
    l2_SAFE_file : str
        Path to the L2 .SAFE file

    """
    is_valid = check_for_invalid_l2_data(l2_SAFE_file, resolution)
    if not is_valid:
        if warning:
            if not input("About to delete {}: Y/N?".format(l2_SAFE_file)).upper().startswith("Y"):
                return
        log.warning("Missing band data. Removing {}".format(l2_SAFE_file))
        shutil.rmtree(l2_SAFE_file)


def clean_l2_dir(l2_dir, resolution="10m", warning=True):
    """
    Calls clean_l2_data on every SAFE file in l2_dir

    Parameters
    ----------
    l2_dir : str
        The L2 directory
    resolution : {"10m", "20m", "60m"}, optional
        Resolution to check. Defaults to 10m
    warning : bool, optional
        If True, prompts user before deleting files.


    Returns
    -------

    """
    log.info("Scanning {} for missing band data in .SAFE files".format(l2_dir))
    for safe_file_path in [os.path.join(l2_dir, safe_file_name) for safe_file_name in os.listdir(l2_dir)]:
        clean_l2_data(safe_file_path, resolution, warning)


def clean_aoi(aoi_dir, images_to_keep=4, warning=True):
    """
    Removes all but the last images_to_keep newest images in the L1, L2, merged, stacked and
    composite directories. Will not affect the output folder. Use with caution.

    Parameters
    ----------
    aoi_dir : str
        A directory made by :py:func:`create_file_structure`
    images_to_keep : int, optional
        The number of images to keep

    """
    l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L1C")), recent_first=True)
    l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L2A")), recent_first=True)
    comp_l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2A")), recent_first=True)
    comp_l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2A")), recent_first=True)
    merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/bandmerged")) if image.endswith(".tif")],
        recent_first=True)
    stacked_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/stacked")) if image.endswith(".tif")],
        recent_first=True)
    comp_merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "composite/bandmerged")) if image.endswith(".tif")],
        recent_first=True)
    for image_list in (l1_list, l2_list, comp_l1_list, comp_l2_list):
        for safe_file in image_list[images_to_keep:]:
            os.rmdir(safe_file)
    for image_list in (merged_list, stacked_list, comp_merged_list):
        for image in image_list[images_to_keep:]:
            os.remove(image)
            os.remove(image.rsplit('.')(0)+".msk")


def sort_by_timestamp(strings, recent_first=True):
    """
    Takes a list of strings that contain sen2 timestamps and returns them sorted, from most recent on back by default.
    Removes any string that does not contain a timestamp

    Parameters
    ----------
    strings : list of str
        list of timestamped strings to sort
    recent_first : bool, optional
        If True, `sorted_strings[0]` is the most recent image. If False, `sorted_strings[0]` is the least recent image.

    Returns
    -------
    sorted_strings : list of str
        The strings sorted by timestamp.

    Notes
    -----
    Does not guarantee preservation of ordering of strings with the same timestamp.

    """
    strings = list(filter(get_image_acquisition_time, strings))
    strings.sort(key=lambda x: get_image_acquisition_time(x), reverse=recent_first)
    return strings


def get_change_detection_dates(image_name):
    """
    Extracts the before_date and after_date dates from a change detection image name.

    Parameters
    ----------
    image_name : str
        A Pyeo-produced image name (example: `class_composite_T36MZE_20190509T073621_20190519T073621.tif`)

    Returns
    -------
    before_date, after_date : DateTime
        The dates associated with the image

    """
    date_regex = r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d"
    timestamps = re.findall(date_regex, image_name)
    date_times = [datetime.datetime.strptime(timestamp, r"%Y%m%dT%H%M%S") for timestamp in timestamps]
    date_times.sort()
    return date_times


def get_preceding_image_path(target_image_name, search_dir):
    """
    Finds the image that directly precedes the target image in time.

    Parameters
    ----------
    target_image_name : str
        A Pyeo or Sentinel generated image name
    search_dir : str
        The directory to search for the preceding image.

    Returns
    -------
    preceding_path : str
        The path to the preceding image

    Raises
    ------
    FileNotFoundError
        If there is no image older than the target image

    """
    target_time = get_image_acquisition_time(target_image_name)
    image_paths = sort_by_timestamp(os.listdir(search_dir), recent_first=True)  # Sort image list newest first
    image_paths = filter(is_tif, image_paths)
    for image_path in image_paths:   # Walk through newest to oldest
        accq_time = get_image_acquisition_time(image_path)   # Get this image time
        if accq_time < target_time:   # If this image is older than the target image, return it.
            return os.path.join(search_dir, image_path)
    raise FileNotFoundError("No image older than {}".format(target_image_name))


def is_tif(image_string):
    """
    :meta private:
    Returns True if image ends with .tif
    """
    if image_string.endswith(".tif"):
        return True
    else:
        return False


def get_pyeo_timestamp(image_name):
    """
    Returns a list of all timestamps in a Pyeo image (yyyymmddhhmmss)

    Parameters
    ----------
    image_name : str
        The Pyeo image name

    Returns
    -------
    timestamp_list : list of str
        A list of the timestamps in the name

    """
    timestamp_re = r"\d{14}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def get_sen_2_tiles(image_dir):
    """
    Returns a list of Sentinel-2 tile IDs present in a folder

    Parameters
    ----------
    image_dir : str
        Path to the folder containing images

    Returns
    -------
    tiles : list of str
        A list of the tile identifiers present in the folder

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
    """
    Gets the datetime object from a .safe filename of a Sentinel image. No test. Returns None if no timestamp present
    Parameters
    ----------
    image_name : str
        The .SAFE filename

    Returns
    -------
    acquisition_time : DateTime
        A DateTime object providing the acquisition time
    """
    try:
        return dt.datetime.strptime(get_sen_2_image_timestamp(image_name), '%Y%m%dT%H%M%S')
    except AttributeError:
        return None


def get_related_images(target_image_name, project_dir):
    """
    Finds every image related to the target image (L1, L2, merged, stacked and classified). Based on timestamp, and
    assumes a project structured by :py:func:`create_file_structure`
    Parameters
    ----------
    target_image_name : str
        The target image
    project_dir : str
        The root of the project directory

    Returns
    -------
    related_images : list of str
        A list of all the images related to target_image

    """
    timestamp = get_sen_2_image_timestamp(target_image_name)
    image_glob = r"*{}*".format(timestamp)
    return glob.glob(image_glob, project_dir)


def get_safe_product_type(image_name):
    """
    Returns the product string (MSIL1C or MSIL2A) from a .safe file identifier
    Parameters
    ----------
    image_name : str
        The name of the image

    Returns
    -------
    product_string : {"MSIL1C" or "MSIL2A"}
        The product string

    """
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[1]


def get_l1_safe_file(image_name, l1_dir):
    """
    Returns the path to the L1 .SAFE file of a L2 image. Gets from granule and timestamp. image_name can be a path or
    a filename. Returns None if not found.

    Parameters
    ----------
    image_name : str
        The name of the L2 image.
    l1_dir : str
        The path to the folder containing L1 images.

    Returns
    -------
    l1_image : str or None
        The path to the L1 image, or None if not found.

    """
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL1C_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l1_dir, safe_glob))
    if len(out) == 0:
        return None
    return out[0]


def get_l2_safe_file(image_name, l2_dir):
    """
    Returns the path to the L2 .SAFE file of a L1 image. Gets from granule and timestamp. image_name can be a path or
    a filename. Returns None if not found.

    Parameters
    ----------
    image_name : str
        The name of the L1 image.
    l2_dir : str
        The path to the folder containing L2 images.

    Returns
    -------
    l2_image : str or None
        The path to the L2 image, or None if not found.

    """
    timestamp = get_sen_2_image_timestamp(os.path.basename(image_name))
    granule = get_sen_2_image_tile(os.path.basename(image_name))
    safe_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.SAFE".format(timestamp, granule)
    out = glob.glob(os.path.join(l2_dir, safe_glob))
    if len(out) == 0:
        return None
    return out[0]


def get_sen_2_image_timestamp(image_name):
    """
    Returns the timestamps part of a Sentinel image

    Parameters
    ----------
    image_name : str
        The Sentinel image name or path

    Returns
    -------
    timestamp : str
        The timestamp (yyyymmddThhmmss)

    """
    timestamp_re = r"\d{8}T\d{6}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def get_sen_2_image_orbit(image_name):
    """
    Returns the relative orbit identifer of a Sentinel 2 image

    Parameters
    ----------
    image_name : str
        The Sentinel image name or path

    Returns
    -------
    orbit_number : str
        The orbit. (eg: 'R012')

    """
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[4]


def get_sen_2_baseline(image_name):
    """
    "Returns the baseline orbit identifier of a Sentinel 2 image
    Parameters
    ----------
    image_name : str
        The Sentinel image name or path

    Returns
    -------
    baseline : str
        The baseline orbit identifier (eg: 'N0214')

    """
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[3]


def get_sen_2_image_tile(image_name):
    """
    Returns the tile number of a Sentinel 2 image or path
    Parameters
    ----------
    image_name : str
        The Sentinel image name or path

    Returns
    -------
    tile_id : str
        The tile ID (eg: 'T13QFB')
    """
    name = os.path.basename(image_name)
    tile = re.findall(r"T\d{2}[A-Z]{3}", name)[0]  # Matches tile ID, but not timestamp
    return tile


def get_sen_2_granule_id(safe_dir):
    """
    Returns the unique ID of a Sentinel 2 granule from a SAFE directory path

    Parameters
    ----------
    safe_dir : str
        Path to a .SAFE file

    Returns
    -------
    id : str
        The full ID of that S2 product (eg: 'S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359')

    """
    tmp = os.path.basename(safe_dir) # removes path to SAFE directory
    id = tmp.split(".")[0] # removes ".SAFE" from the ID name
    return id


def get_mask_path(image_path):
    """
    Returns the path to the mask file of the image. Does not verify that the mask exists.
    Parameters
    ----------
    image_path : str
        Path to the image

    Returns
    -------
    mask_path : str
        Path to the corresponding mask.

    """
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    mask_name = image_name.rsplit('.')[0] + ".msk"
    mask_path = os.path.join(image_dir, mask_name)
    return mask_path
