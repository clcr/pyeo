"""
pyeo.filesystem_utilities
-------------------------
Contains functions for sorting, creating and comparing images as part of the filesystem. Includes any function
that works on a generic pyeo timestamp and sentinel 2 functions.
"""

import datetime
import datetime as dt
import glob
import logging
import os
import re
import shutil

from pyeo.exceptions import CreateNewStacksException

# Set up logging on import
log = logging.getLogger("pyeo")
formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")


def init_log(log_path):
    """Sets up the log format and log handlers; one for stdout and to write to a file, 'log_path'.
     Returns the log for the calling script"""
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    log = logging.getLogger("pyeo")
    log.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info("****PROCESSING START****")
    return log


def create_file_structure(root):
    """Creates the file structure if it doesn't exist already"""
    os.chdir(root)
    dirs = [
        "images/",
        "images/L1/",
        "images/L2/",
        "images/merged/",
        "images/stacked/",
        "images/planet/",
        "composite/",
        "composite/L1",
        "composite/L2",
        "composite/merged",
        "output/",
        "output/categories",
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
    """Checks the existance of the specified resolution of imagery. Returns a True-value with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.
    Retuns 1 if imagery is valid, 0 if not and 2 if not a safe-file"""
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
    """Checks the existance of the specified resolution of imagery. Returns True with a warning if passed
    an invalid SAFE directory; this will prevent disconnected files from being deleted.
    Retuns 1 if imagery is valid, 0 if not and 2 if not a safe-file"""
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
    """Removes any directories that don't have band 2, 3, 4 or 8 in the specified resolution folder
    If warning=True, prompts first."""
    is_valid = check_for_invalid_l2_data(l2_SAFE_file, resolution)
    if not is_valid:
        if warning:
            if not input("About to delete {}: Y/N?".format(l2_SAFE_file)).upper().startswith("Y"):
                return
        log.warning("Removing {}".format(l2_SAFE_file))
        shutil.rmtree(l2_SAFE_file)


def clean_l2_dir(l2_dir, resolution="10m", warning=True):
    """Calls clean_l2_data on every SAFE file in l2_dir"""
    log.info("Scanning {} for incomplete SAFE files".format(l2_dir))
    for safe_file_path in [os.path.join(l2_dir, safe_file_name) for safe_file_name in os.listdir(l2_dir)]:
        clean_l2_data(safe_file_path, resolution, warning)


def clean_aoi(aoi_dir, images_to_keep = 4, warning=True):
    """Removes all but the last images_to_keep newest images in the L1, L2, merged, stacked and
    composite directories. Will not affect the output folder."""
    l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L1")), recent_first=True)
    l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "images/L2")), recent_first=True)
    comp_l1_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2")), recent_first=True)
    comp_l2_list = sort_by_timestamp(os.listdir(os.path.join(aoi_dir, "composite/L2")), recent_first=True)
    merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/merged")) if image.endswith(".tif")],
        recent_first=True)
    stacked_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "images/stacked")) if image.endswith(".tif")],
        recent_first=True)
    comp_merged_list = sort_by_timestamp(
        [image for image in os.listdir(os.path.join(aoi_dir, "composite/merged")) if image.endswith(".tif")],
        recent_first=True)
    for image_list in (l1_list, l2_list, comp_l1_list, comp_l2_list):
        for safe_file in image_list[images_to_keep:]:
            os.rmdir(safe_file)
    for image_list in (merged_list, stacked_list, comp_merged_list):
        for image in image_list[images_to_keep:]:
            os.remove(image)
            os.remove(image.rsplit('.')(0)+".msk")


def sort_by_timestamp(strings, recent_first=True):
    """Takes a list of strings that contain sen2 timestamps and returns them sorted, most recent first. Does not
    guarantee ordering of strings with the same timestamp. Removes any string that does not contain a timestamp"""
    strings = list(filter(get_image_acquisition_time, strings))
    strings.sort(key=lambda x: get_image_acquisition_time(x), reverse=recent_first)
    return strings


def get_change_detection_dates(image_name):
    """Takes the source filepath and extracts the before_date and after_date dates from in, in that order."""
    date_regex = r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d"
    timestamps = re.findall(date_regex, image_name)
    date_times = [datetime.datetime.strptime(timestamp, r"%Y%m%dT%H%M%S") for timestamp in timestamps]
    date_times.sort()
    return date_times


def get_preceding_image_path(target_image_name, search_dir):
    """Gets the path to the image in search_dir preceding the image called image_name"""
    target_time = get_image_acquisition_time(target_image_name)
    image_paths = sort_by_timestamp(os.listdir(search_dir), recent_first=True)  # Sort image list newest first
    image_paths = filter(is_tif, image_paths)
    for image_path in image_paths:   # Walk through newest to oldest
        accq_time = get_image_acquisition_time(image_path)   # Get this image time
        if accq_time < target_time:   # If this image is older than the target image, return it.
            return os.path.join(search_dir, image_path)
    raise FileNotFoundError("No image older than {}".format(target_image_name))


def is_tif(image_string):
    """Returns True if image ends with .tif"""
    if image_string.endswith(".tif"):
        return True
    else:
        return False


def get_pyeo_timestamp(image_name):
    """Returns a list of all timestamps in a Pyeo image."""
    timestamp_re = r"\d{14}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


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


def get_sen_2_baseline(image_name):
    """Returns the baseline string of a s2 image"""
    tmp1 = image_name.split("/")[-1]  # remove path
    tmp2 = tmp1.split(".")[0] # remove file extension
    comps = tmp2.split("_") # decompose
    return comps[3]


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


def get_mask_path(image_path):
    """A gdal mask is an image with the same name as the image it's masking, but with a .msk extension"""
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    mask_name = image_name.rsplit('.')[0] + ".msk"
    mask_path = os.path.join(image_dir, mask_name)
    return mask_path