"""
Contains functions for sorting, creating and comparing images as part of the filesystem. Includes any function
that works on a generic pyeo timestamp and sentinel 2 functions.

Key functions
-------------

:py:func:`init_log` Sets up logging to both console and file

:py:func:`create_file_structure` Creates a recommended file structure for automated work

:py:func:`sort_by_timestamp` Sorts a set of files by timestamp

Function reference
------------------
"""

import configparser
import datetime
import datetime as dt
import glob
import json
import logging
import os
import sys
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from pyeo.exceptions import CreateNewStacksException

# Set up logging on import
log = logging.getLogger("pyeo")
formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")

# def gdal_switch(installation: str,
#                 config_dict: dict) -> None:
#     """
#     This function performs a Platform (OS) Independent switch of the `GDAL_DATA` and `PROJ_LIB` installation paths to the required version. This is necessary because of using the same conda environment to perform functions that use different GDAL installations.

#     Parameters
#     ---------
#     installation : str
#         a string of either 'gdal_api' or 'geopandas', indicating which GDAL and PROJ_LIB to switch to
#     config_dict : dict
#         a config_dict containing `conda_directory` and `conda_env_name`
    
#     Returns
#     ---------
#     None

#     """
    

#     conda_env_name = config_dict["conda_env_name"]
#     conda_directory = config_dict["conda_directory"]
#     # platform if branches
#     if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
#         # try:
#         if installation == "geopandas":
#             # geopandas installation of GDAL 
#             gdal_path = f"{conda_directory}/envs/{conda_env_name}/lib/python3.10/site-packages/fiona/gdal_data"
#             proj_path = f"{conda_directory}/envs/{conda_env_name}/lib/python3.10/site-packages/fiona/proj_data"

#             if not os.path.exists(gdal_path):
#                 log.info("gdal branch reached")
#                 log.error(f"{gdal_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether geopandas was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)

#             if not os.path.exists(proj_path):
#                 log.error(f"{proj_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether geopandas was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)  
#             else:
#                 # set geopandas GDAL and PROJ_LIB installations
#                 os.environ["GDAL_DATA"] = gdal_path
#                 os.environ["PROJ_LIB"] = proj_path

#         if installation == "gdal_api":
#             # GDAL and PROJ_LIB standard installation
#             gdal_path = f"{conda_directory}/envs/{conda_env_name}/share/gdal"
#             proj_path = f"{conda_directory}/envs/{conda_env_name}/share/proj"

#             if not os.path.exists(gdal_path):
#                 log.error(f"{gdal_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)

#             if not os.path.exists(proj_path):
#                 log.error(f"{proj_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)   
#             else:
#                 os.environ["GDAL_DATA"] = gdal_path
#                 os.environ["PROJ_LIB"] = proj_path
#     # except Exception as error:
#     #     log.error(f"received this error : {error}'")
#     #     log.error(f"exiting...")
#     #     sys.exit(1)

#     elif sys.platform.startswith("win"):
#         # try:
#         if installation == "geopandas":
#             # geopandas installation of GDAL
#             gdal_path = f"{conda_directory}\\envs\\{conda_env_name}\\Lib\\site-packages\\fiona\\gdal_data"
#             proj_path = f"{conda_directory}\\envs\\{conda_env_name}\\Lib\\site-packages\\fiona\\proj_data"

#             if not os.path.exists(gdal_path):
#                 log.error(f"{gdal_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)

#             if not os.path.exists(proj_path):
#                 log.error(f"{proj_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)
#             else:
#                 os.environ["GDAL_DATA"] = gdal_path
#                 os.environ["PROJ_LIB"] = proj_path

#         if installation == "gdal_api":
#             # GDAL and PROJ_LIB standard installation
#             gdal_path = f"{conda_directory}\\envs\\{conda_env_name}\\Library\\share\\gdal"
#             proj_path = f"{conda_directory}\\envs\\{conda_env_name}\\Library\\share\\proj"

#             if not os.path.exists(gdal_path):
#                 log.error(f"{gdal_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)

#             if not os.path.exists(proj_path):
#                 log.error(f"{proj_path} does not exist")
#                 log.error(f"check conda directory and conda env name were typed correctly in the .ini")
#                 log.error(f"check whether GDAL was installed")
#                 log.error("now exiting the pipeline...")
#                 sys.exit(1)
#             else:
#                 os.environ["GDAL_DATA"] = gdal_path
#                 os.environ["PROJ_LIB"] = proj_path

#         # except Exception as error:
#         #     log.error(f"received this error : {error}'")
#         #     log.error(f"exiting...")
#         #     sys.exit(1)
#     else:
#         log.error(f"OS is not one of 'win', 'linux' or 'darwin'")
#         log.error(f"OS this script is running on is : {sys.platform}")
#         log.error(f"exiting pipeline")
#         sys.exit(1)

#     return

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


def init_log_acd(log_path, logger_name):
    """
    This function differs slightly to `init_log` in that it accomodates a logger_name. This enables \n
    unique logger objects to be created so multiple loggers can be run at a time.

    Sets up the log format and log handlers; one for stdout and to write to a file, 'log_path'.
    Returns the log for the calling script.

    Parameters
    ----------
    log_path : str
        The path to the file output of this log.

    logger_name : str
        A unique logger name.

    Returns
    -------
    log : logging.Logger
        The logging object.

    """

    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("---------------------------------------------------------------")
    logger.info("                    ****PROCESSING START****")
    logger.info("---------------------------------------------------------------")


    return logger


def conda_check(config_dict: dict, log):
    """

    This function takes the path to the config (pyeo.ini) and checks whether the conda environment exists.

    Parameters
    ----------

    config_dict : dict
        config_dict containing `conda_directory` and `conda_env_name`, which will be checked to see if they exist.

    Returns
    --------

    True/False (bool)

    """

    conda_directory = config_dict["conda_directory"]
    conda_env_name = config_dict["conda_env_name"]
    conda_env_path = f"{conda_directory}{os.sep}envs{os.sep}{conda_env_name}"
    log.info(conda_env_path)
    if os.path.exists(conda_env_path):
        return True
    else:
        return False


def config_path_to_config_dict(config_path: str):
    """

    This function takes the path to the config (pyeo.ini) and simplifies the keys

    Parameters
    ----------

    config_path : str
        path to pyeo.ini

    Returns
    --------

    config_dict : dict
        a config dictionary

    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    config_dict = {}

    config_dict["qsub_processor_options"] = config["run_mode"]["qsub_processor_options"]

    config_dict["do_parallel"] = config.getboolean("run_mode", "do_parallel")
    config_dict["wall_time_hours"] = int(config["run_mode"]["wall_time_hours"])
    config_dict["watch_time_hours"] = int(config["run_mode"]["watch_time_hours"])
    config_dict["watch_period_seconds"] = int(
        config["run_mode"]["watch_period_seconds"]
    )
    
    config_dict["do_tile_intersection"] = config.getboolean("raster_processing_parameters", "do_tile_intersection")

    config_dict["do_raster"] = config.getboolean(
        "raster_processing_parameters", "do_raster"
    )
    config_dict["do_dev"] = config.getboolean("raster_processing_parameters", "do_dev")
    config_dict["do_all"] = config.getboolean("raster_processing_parameters", "do_all")

    # config_dict["do_download_from_scihub"] = config.getboolean("raster_processing_parameters", "do_download_from_scihub")

    # config_dict["do_download_from_dataspace"] = config.getboolean("raster_processing_parameters", "do_download_from_dataspace")
    
    config_dict["do_classify"] = config.getboolean(
        "raster_processing_parameters", "do_classify"
    )
    config_dict["do_change"] = config.getboolean(
        "raster_processing_parameters", "do_change"
    )
    config_dict["do_download"] = config.getboolean(
        "raster_processing_parameters", "do_download"
    )
    config_dict["do_update"] = config.getboolean(
        "raster_processing_parameters", "do_update"
    )
    config_dict["do_quicklooks"] = config.getboolean(
        "raster_processing_parameters", "do_quicklooks"
    )
    config_dict["do_delete"] = config.getboolean(
        "raster_processing_parameters", "do_delete"
    )

    config_dict["do_zip"] = config.getboolean("raster_processing_parameters", "do_zip")
    config_dict["build_composite"] = config.getboolean(
        "raster_processing_parameters", "do_build_composite"
    )
    config_dict["build_prob_image"] = config.getboolean(
        "raster_processing_parameters", "do_build_prob_image"
    )
    config_dict["do_skip_existing"] = config.getboolean(
        "raster_processing_parameters", "do_skip_existing"
    )

    config_dict["start_date"] = config["forest_sentinel"]["start_date"]
    config_dict["end_date"] = config["forest_sentinel"]["end_date"]
    config_dict["composite_start"] = config["forest_sentinel"]["composite_start"]
    config_dict["composite_end"] = config["forest_sentinel"]["composite_end"]
    config_dict["epsg"] = int(config["forest_sentinel"]["epsg"])
    config_dict["cloud_cover"] = int(config["forest_sentinel"]["cloud_cover"])
    config_dict["cloud_certainty_threshold"] = int(
        config["forest_sentinel"]["cloud_certainty_threshold"]
    )
    config_dict["model_path"] = config["forest_sentinel"]["model"]
    config_dict["download_source"] = config["raster_processing_parameters"][
        "download_source"
    ]

    # print(config["raster_processing_parameters"]["band_names"])
    config_dict["bands"] = json.loads(
        config["raster_processing_parameters"]["band_names"]
    )

    config_dict["resolution_string"] = config["raster_processing_parameters"][
        "resolution_string"
    ]
    config_dict["output_resolution"] = int(
        config["raster_processing_parameters"]["output_resolution"]
    )
    config_dict["buffer_size_cloud_masking"] = int(
        config["raster_processing_parameters"]["buffer_size_cloud_masking"]
    )
    config_dict["buffer_size_cloud_masking_composite"] = int(
        config["raster_processing_parameters"]["buffer_size_cloud_masking_composite"]
    )
    config_dict["download_limit"] = int(
        config["raster_processing_parameters"]["download_limit"]
    )
    config_dict["faulty_granule_threshold"] = int(
        config["raster_processing_parameters"]["faulty_granule_threshold"]
    )
    config_dict["sieve"] = int(config["raster_processing_parameters"]["sieve"])
    config_dict["chunks"] = int(config["raster_processing_parameters"]["chunks"])
    config_dict["class_labels"] = json.loads(
        config["raster_processing_parameters"]["class_labels"]
    )
    config_dict["from_classes"] = json.loads(
        config["raster_processing_parameters"]["change_from_classes"]
    )
    config_dict["to_classes"] = json.loads(
        config["raster_processing_parameters"]["change_to_classes"]
    )
    config_dict["environment_manager"] = config["environment"]["environment_manager"]
    if config_dict["environment_manager"] == "conda":
        config_dict["conda_directory"] = config["environment"]["conda_directory"]
        config_dict["conda_env_name"] = config["environment"]["conda_env_name"]
    config_dict["pyeo_dir"] = config["environment"]["pyeo_dir"]
    config_dict["tile_dir"] = config["environment"]["tile_dir"]
    config_dict["integrated_dir"] = config["environment"]["integrated_dir"]
    config_dict["roi_dir"] = config["environment"]["roi_dir"]
    config_dict["roi_filename"] = config["environment"]["roi_filename"]
    config_dict["geometry_dir"] = config["environment"]["geometry_dir"]
    config_dict["s2_tiles_filename"] = config["environment"]["s2_tiles_filename"]
    config_dict["log_dir"] = config["environment"]["log_dir"]
    config_dict["log_filename"] = config["environment"]["log_filename"]
    config_dict["sen2cor_path"] = config["environment"]["sen2cor_path"]

    config_dict["level_1_filename"] = config["vector_processing_parameters"][
        "level_1_filename"
    ]
    config_dict["level_1_boundaries_path"] = os.path.join(
        config_dict["geometry_dir"], config_dict["level_1_filename"]
    )
    config_dict["do_delete_existing_vector"] = config.getboolean(
        "vector_processing_parameters", "do_delete_existing_vector"
    )

    config_dict["do_vectorise"] = config.getboolean(
        "vector_processing_parameters", "do_vectorise"
    )
    config_dict["do_integrate"] = config.getboolean(
        "vector_processing_parameters", "do_integrate"
    )
    config_dict["do_filter"] = config.getboolean(
        "vector_processing_parameters", "do_filter"
    )

    config_dict["counties_of_interest"] = json.loads(
        config["vector_processing_parameters"]["counties_of_interest"]
    )
    config_dict["minimum_area_to_report_m2"] = int(
        config["vector_processing_parameters"]["minimum_area_to_report_m2"]
    )
    config_dict["do_distribution"] = config.getboolean(
        "vector_processing_parameters", "do_distribution"
    )

    config_dict["credentials_path"] = config["environment"]["credentials_path"]

    return config_dict


def create_file_structure(root: str):
    """
    Creates the folder structure used in rolling_s2_composite and some other functions: ::
        root
        -images
        --L1C
        --L2A
        --bandmerged
        --stacked
        --stacked_mosaic
        --stacked_masked
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
        "images/stacked_mosaic/",
        "images/stacked_masked/",
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
        "output/quicklooks/",
        "log/",
    ]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass


def create_folder_structure_for_tiles(root):
    """
    Creates the folder structure used in tile_based_change_detection.py: ::
        root
        -images
        --L1C
        --L2A
        --cloud_masked
        -composite
        --L1C
        --L2A
        --cloud_masked
        -output
        --classified
        --probabilities

    Parameters
    ----------
    root : str
        The root folder for the file strucutre
    """

    if not os.path.exists(root):
        os.makedirs(root)
    os.chdir(root)
    dirs = [
        "composite/",
        "composite/L1C/",
        "composite/L2A/",
        "composite/cloud_masked/",
        "images/",
        "images/L1C/",
        "images/L2A/",
        "images/cloud_masked/",
        "output/",
        "output/classified/",
        "output/probabilities/",
        "output/quicklooks/",
        "log/",
    ]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass


def validate_config_file(config_path):
    # TODO: fill
    pass


def get_filenames(path, filepattern, dirpattern):
    """
    Finds all file names in a directory for which the file name matches a certain string pattern,
    and the directory name matches a different string pattern.

    Args:
      path = string indicating the path to a directory in which the search will be done
      filepattern = string of the file name pattern to search for
      dirpattern = string of the directory name pattern to search for

    Returns:
      a list of all found files with the full path directory
    """

    log = logging.getLogger("pyeo")

    filelist = []
    for root, dirs, files in os.walk(path, topdown=True):
        # log.info("root, dirs, files: {}".format(root,dirs,files))
        dirs[:] = [d for d in dirs]
        for f in files:
            if filepattern in f and dirpattern in root:
                thisfile = os.path.join(root, f)
                # log.info("Found file: {}".format(thisfile))
                filelist.append(thisfile)
    return sorted(filelist)


def get_raster_paths(paths, filepatterns, dirpattern):
    """
    Iterates over get_filenames for different paths and different file patterns and
    returns a dataframe of all directory paths that match the conditions together with
    the root path in which they were found.

    Args:
      paths = list of strings indicating the path to a root directory in which the search will be done
      filepatterns = list of strings of the file name patterns to search for
      dirpattern = string of the directory name pattern to search for
    Returns:
      a dataframe of all found file paths, one line per path
    """
    cols = ["safe_path"]
    for filepattern in filepatterns:
        cols.append(filepattern)
    results = []
    # iterate over all SAFE directories
    for path in paths:
        # log.info("  path = {}".format(path))
        row = [path]
        # iterate over all band file name patterns
        for filepattern in filepatterns:
            # log.info("  filepattern = {}".format(filepattern))
            f = get_filenames(path, filepattern, dirpattern)
            if len(f) == 1:
                row.append(f)
            if len(f) > 1:
                log.warning("More than one file path returned in raster path search:")
                log.warning("  root        = {}".format(path))
                log.warning("  filepattern = {}".format(filepattern))
                log.warning("  dirpattern  = {}".format(dirpattern))
                log.info("The search returned:")
                for i in f:
                    log.info("  {}".format(i))
                row.append("")
            if len(f) == 0:
                log.info(
                    "File pattern {} and dir pattern {} not found in {}".format(
                        filepattern, dirpattern, path
                    )
                )
                row.append("")
        # log.info("  results = {}".format(results))
        # log.info("  paths   = {}".format(paths))
        # log.info("  row     = {}".format(row))
        results.extend(row)
        # log.info("  results = {}".format(results))
    # todo: the following line expects exactly one search result per path. Challenge that assumption
    if len(results) == len(paths) * (len(filepatterns) + 1):
        arr = np.array(results, dtype=object).reshape(len(paths), len(filepatterns) + 1)
        results = pd.DataFrame(arr, columns=cols)
    else:
        log.warning("  Unexpected shape of file name pattern search results:")
        log.warning("    Results      = {}".format(results))
        log.warning("    Column names = {}".format(cols))
        log.warning("    Returning an empty string.")
        results = ""
    # log.info("  DF = {}".format(results))
    return results


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

    log = logging.getLogger("pyeo")

    if not os.path.exists(l2_SAFE_file):
        log.info("{} does not exist.".format(l2_SAFE_file))
        return 2

    if not l2_SAFE_file.endswith(".SAFE") or "L2A" not in l2_SAFE_file:
        log.info("{} does not exist.".format(l2_SAFE_file))
        return 2
    log.info("Checking {} for incomplete {} imagery".format(l2_SAFE_file, resolution))

    bands = ["B08", "B04", "B03", "B02"]
    nb = 0
    for band in bands:
        f = get_filenames(l2_SAFE_file, band, "")
        f = [
            filename
            for filename in f
            if (".jp2" in filename) and (resolution in filename)
        ]
        if len(f) > 0:
            for i in range(len(f)):
                log.info("   {}".format(f[i]))
            nb = nb + 1
        else:
            log.warning("Band file not found for band: {}".format(band))
    if nb == len(bands):
        log.info("All necessary bands have been found")
        return 1
    else:
        log.warning("Not all necessary bands have been found in the SAFE directory")
        log.warning("n bands = {}".format(nb))
        return 0

    """
    # NOT USED
    
    def find_file(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

    def find_dir(name, path):
        for root, dirs, files in os.walk(path):
            if name in dirs:
                return os.path.join(root, name)
    """

    """
    # check whether the band rasters are in the IMG_DATA/R10 or similar subdirectory
    granule_path = r"GRANULE/*/IMG_DATA/R{}/*_B0[8,4,3,2]*.jp2".format(resolution)
    image_glob = os.path.join(l2_SAFE_file, granule_path)
    if len(glob.glob(image_glob)) == 4:
        log.info("All necessary bands are complete")
        return 1
    else:
        #TODO: check whether the moving of band raster files into the "Rxx" subdirectory works OK
        # check whether the band rasters are in the IMG_DATA subdirectory
        granule_path = r"GRANULE/*/IMG_DATA/*_B0[8,4,3,2]*.jp2"
        image_glob = os.path.join(l2_SAFE_file, granule_path)
        if len(glob.glob(image_glob)) == 4:
            log.info("All necessary bands are complete")
            d=find_dir(r"GRANULE/*/IMG_DATA/R{}".format(resolution), l2_SAFE_file)
            os.mkdir(d)
            bands=["B08","B04","B03","B02"]
            for band in bands:
                path=glob.glob(os.path.join(l2_SAFE_file, r"GRANULE/*/IMG_DATA"))
                f=find_file(band, l2_SAFE_file)
                os.rename(f, os.path.join(os.path.dirname(f),r"R{}".format(resolution),os.path.basename(f)))
            return 1
        else:
            log.warning("Not all necessary bands have been found in the SAFE directory")
            return 0
    """


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
    if not os.path.exists(l1_SAFE_file):
        log.info("{} does not exist.".format(l1_SAFE_file))
        return 2
    if not l1_SAFE_file.endswith(".SAFE") or "L1C" not in l1_SAFE_file:
        log.info("{} does not exist.".format(l1_SAFE_file))
        return 2
    log.info("Checking {} for incomplete imagery".format(l1_SAFE_file))
    granule_path = r"GRANULE/*/IMG_DATA/*_B0[8,4,3,2]*.jp2"
    image_glob = os.path.join(l1_SAFE_file, granule_path)
    if len(glob.glob(image_glob)) == 4:
        log.info("All necessary bands are complete")
        return 1
    else:
        log.warning("Not all necessary bands have been found in the SAFE directory")
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
            if (
                not input("About to delete {}: Y/N?".format(l2_SAFE_file))
                .upper()
                .startswith("Y")
            ):
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
    for safe_file_path in [
        os.path.join(l2_dir, safe_file_name) for safe_file_name in os.listdir(l2_dir)
    ]:
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
    l1_list = sort_by_timestamp(
        os.listdir(os.path.join(aoi_dir, "images/L1C")), recent_first=True
    )
    l2_list = sort_by_timestamp(
        os.listdir(os.path.join(aoi_dir, "images/L2A")), recent_first=True
    )
    comp_l1_list = sort_by_timestamp(
        os.listdir(os.path.join(aoi_dir, "composite/L2A")), recent_first=True
    )
    comp_l2_list = sort_by_timestamp(
        os.listdir(os.path.join(aoi_dir, "composite/L2A")), recent_first=True
    )
    merged_list = sort_by_timestamp(
        [
            image
            for image in os.listdir(os.path.join(aoi_dir, "images/bandmerged"))
            if image.endswith(".tif")
        ],
        recent_first=True,
    )
    stacked_list = sort_by_timestamp(
        [
            image
            for image in os.listdir(os.path.join(aoi_dir, "images/stacked"))
            if image.endswith(".tif")
        ],
        recent_first=True,
    )
    comp_merged_list = sort_by_timestamp(
        [
            image
            for image in os.listdir(os.path.join(aoi_dir, "composite/bandmerged"))
            if image.endswith(".tif")
        ],
        recent_first=True,
    )
    for image_list in (l1_list, l2_list, comp_l1_list, comp_l2_list):
        for safe_file in image_list[images_to_keep:]:
            os.rmdir(safe_file)
    for image_list in (merged_list, stacked_list, comp_merged_list):
        for image in image_list[images_to_keep:]:
            os.remove(image)
            os.remove(image.rsplit(".")(0) + ".msk")


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


def get_change_detection_date_strings(image_name):
    """
    Extracts the before_date and after_date dates from a change detection image name.

    Parameters
    ----------
    image_name : str
        A pyeo-produced image name (example: `class_composite_T36MZE_20190509T073621_20190519T073621.tif`)

    Returns
    -------
    before_date, after_date : str
        The dates associated with the image in string format

    """
    date_regex = r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d"
    timestamps = re.findall(date_regex, image_name)
    return timestamps


def get_change_detection_dates(image_name):
    """
    Extracts the before_date and after_date dates from a change detection image name.

    Parameters
    ----------
    image_name : str
        A pyeo-produced image name (example: `class_composite_T36MZE_20190509T073621_20190519T073621.tif`)

    Returns
    -------
    before_date, after_date : DateTime
        The dates associated with the image

    """
    date_regex = r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d"
    timestamps = re.findall(date_regex, image_name)
    date_times = [
        datetime.datetime.strptime(timestamp, r"%Y%m%dT%H%M%S")
        for timestamp in timestamps
    ]
    date_times.sort()
    return date_times


def get_preceding_image_path(target_image_name, search_dir):
    """
    Finds the image that directly precedes the target image in time.

    Parameters
    ----------
    target_image_name : str
        A pyeo or Sentinel generated image name
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
    image_paths = sort_by_timestamp(
        os.listdir(search_dir), recent_first=True
    )  # Sort image list newest first
    image_paths = filter(is_tif, image_paths)
    for image_path in image_paths:  # Walk through newest to oldest
        accq_time = get_image_acquisition_time(image_path)  # Get this image time
        if (
            accq_time < target_time
        ):  # If this image is older than the target image, return it.
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
    Returns a list of all timestamps in a pyeo image (yyyymmddhhmmss)

    Parameters
    ----------
    image_name : str
        The pyeo image name

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
        return dt.datetime.strptime(
            get_sen_2_image_timestamp(image_name), "%Y%m%dT%H%M%S"
        )
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
    tmp2 = tmp1.split(".")[0]  # remove file extension
    comps = tmp2.split("_")  # decompose
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
    tmp2 = tmp1.split(".")[0]  # remove file extension
    comps = tmp2.split("_")  # decompose
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
    tmp2 = tmp1.split(".")[0]  # remove file extension
    comps = tmp2.split("_")  # decompose
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
    tmp = os.path.basename(safe_dir)  # removes path to SAFE directory
    id = tmp.split(".")[0]  # removes ".SAFE" from the ID name
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
    mask_name = image_name.rsplit(".")[0] + ".msk"
    mask_path = os.path.join(image_dir, mask_name)
    return mask_path


def serial_date_to_string(srl_no: int) -> str:
    """
    Converts a serial date (days since X) to a date as a string.

    Parameters
    ----------
    srl_no : int
        Serial number representing days since X.

    Returns
    -------
    str
        Date in the format "YYYY-MM-DD".

    Notes
    -----
    This function assumes the base date as January 1, 2000.

    References
    ----------
    - Original implementation by AER:
      https://stackoverflow.com/a/39988256/6809533
    """

    import datetime

    new_date = datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(srl_no)
    return new_date.strftime("%Y-%m-%d")



def zip_contents(directory: str, notstartswith=None) -> None:
    """
    Zip the contents of the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory whose contents to zip.
    notstartswith : list or None, optional
        List of prefixes to exclude from zipping. Default is None.

    Returns
    -------
    None

    Notes
    -----
    - This function skips files that have the ".zip" extension.
    - If `notstartswith` is provided, files starting with any of the specified prefixes are skipped.

    """
    paths = [f for f in os.listdir(directory) if not f.endswith(".zip")]
    for f in paths:
        do_it = True
        if notstartswith is not None:
            for i in notstartswith:
                if f.startswith(i):
                    do_it = False
                    log.info("Skipping file that starts with '{}':   {}".format(i, f))
        if do_it:
            file_to_zip = os.path.join(directory, f)
            zipped_file = file_to_zip.split(".")[0]
            log.info("Zipping   {}".format(file_to_zip))
            if os.path.isdir(file_to_zip):
                shutil.make_archive(zipped_file, "zip", file_to_zip)
            else:
                with zipfile.ZipFile(
                    zipped_file + ".zip", "w", compression=zipfile.ZIP_DEFLATED
                ) as zf:
                    zf.write(file_to_zip, os.path.basename(file_to_zip))
            if os.path.exists(zipped_file + ".zip"):
                if os.path.isdir(file_to_zip):
                    shutil.rmtree(file_to_zip)
                else:
                    os.remove(file_to_zip)
            else:
                log.error("Zipping failed: {}".format(zipped_file + ".zip"))
    return


def unzip_contents(zippath: str, ifstartswith=None, ending=None) -> None:
    """
    Unzip the contents of the specified zipped folder.

    Parameters
    ----------
    zippath : str
        Path to the zipped folder to unzip.
    ifstartswith : str or None, optional
        Prefix to check before extracting. Default is None.
    ending : str or None, optional
        Suffix to add to the extracted folder name. Default is None.

    Returns
    -------
    None

    Notes
    -----
    - This function assumes the zip file extension to be ".zip".
    - If `ifstartswith` is provided, extraction occurs only if the folder name starts with the specified prefix.
    - If `ending` is provided, it is added to the extracted folder name.
    """
    
    dirpath = zippath[:-4]  # cut away the  .zip ending
    if ifstartswith is not None and ending is not None:
        if dirpath.startswith(ifstartswith):
            dirpath = dirpath + ending
    log.info("Unzipping {}".format(zippath))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if os.path.exists(dirpath):
        if os.path.exists(zippath):
            shutil.unpack_archive(filename=zippath, extract_dir=dirpath, format="zip")
            os.remove(zippath)
    else:
        log.error("Unzipping failed")
    return
