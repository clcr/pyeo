"""
send_report
-------------------------------------
An app for sending out summary information on detected changes between
classified image maps to users in various ways.
It is intended to support WhatsApp and Email at this stage.
It uses some of the ini file parameters but not the do_x flags.
"""

import argparse
import configparser
import datetime
import geopandas as gpd
import pandas as pd
#import json
import numpy as np
import os
from osgeo import gdal
import shutil
import sys
import warnings
#import zipfile
from pyeo import (
    filesystem_utilities, 
    queries_and_downloads, 
    raster_manipulation
    )
from pyeo.filesystem_utilities import (
    zip_contents,
    unzip_contents
    )
from pyeo.acd_national import (
    acd_initialisation,
    acd_config_to_log,
    acd_roi_tile_intersection
    )

gdal.UseExceptions()

def send_report(config_path, tile_id="None"):
    """
    The main function that sends out change reports created by detect_change.py
        between the median composite and the newly downloaded images with the 
        parameters specified in the ini file found in config_path.

    Args:

        config_path : string with the full path to the ini file or config file containing the
                        processing parameters

        tile_id : string with the Sentinel-2 tile ID. If not "None", this ID is used
                        instead of the region of interest file to define the area to be processed

    """


    # read the ini file contents into a dictionary
    configparser.ConfigParser(allow_no_value=True)
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)
    config_dict, log = acd_initialisation(config_path)
    acd_config_to_log(config_dict, log)

    try:
        # ensure that pyeo is looking in the correct directory
        os.chdir(config_dict["pyeo_dir"]) 
        start_date = config_dict["start_date"]
        end_date = config_dict["end_date"]
        if end_date == "TODAY":
            end_date = datetime.date.today().strftime("%Y%m%d")
        #composite_start_date = config_dict["composite_start"]
        #composite_end_date = config_dict["composite_end"]
        #cloud_cover = config_dict["cloud_cover"]
        #cloud_certainty_threshold = config_dict["cloud_certainty_threshold"]
        #model_path = config_dict["model_path"]
        tile_dir = config_dict["tile_dir"]
        #sen2cor_path = config_dict["sen2cor_path"]
        #epsg = config_dict["epsg"]
        #bands = config_dict["bands"]
        #resolution = config_dict["resolution_string"]
        #out_resolution = config_dict["output_resolution"]
        #buffer_size = config_dict["buffer_size_cloud_masking"]
        #buffer_size_composite = config_dict["buffer_size_cloud_masking_composite"]
        #max_image_number = config_dict["download_limit"]
        #faulty_granule_threshold = config_dict["faulty_granule_threshold"]
        #download_limit = config_dict["download_limit"]
        skip_existing = config_dict["do_skip_existing"]
        #sieve = config_dict["sieve"]
        #from_classes = config_dict["from_classes"]
        #to_classes = config_dict["to_classes"]
        #download_source = config_dict["download_source"]
        #if download_source == "scihub":
        #    log.info("scihub API is the download source")
        #if download_source == "dataspace":
        #    log.info("dataspace API is the download source")
        #log.info("Faulty Granule Threshold is set to   : {}".format(
        #        config_dict['faulty_granule_threshold'])
        #        )
        #log.info("    Files below this threshold will not be downloaded")
        log.info("Successfully read the processing arguments")
    except:
        log.error("Could not read the processing arguments from the ini file.")
        sys.exit(1)

    if tile_id == "None":
        # if no tile ID is given by the call to the function, use the geometry file
        #   to get the tile ID list
        tile_based_processing_override = False
        tilelist_filepath = acd_roi_tile_intersection(config_dict, log)
        log.info("Sentinel-2 tile ID list: " + tilelist_filepath)
        tiles_to_process = list(pd.read_csv(tilelist_filepath)["tile"])
    else:
        # if a tile ID is specified, use that and do not use the tile intersection
        #   method to get the tile ID list
        tile_based_processing_override = True
        tiles_to_process = [tile_id]

    if tile_based_processing_override:
        log.info("Tile based processing selected. Overriding the geometry file intersection method")
        log.info("  to get the list of tile IDs.")

    log.info(str(len(tiles_to_process)) + " Sentinel-2 tile reports to process.")

    # iterate over the tiles
    for tile_to_process in tiles_to_process:
        log.info("Sending out the latest reports for Sentinel-2 tile: " + tile_to_process)
        log.info("    See tile log file for details.")
        individual_tile_directory_path = os.path.join(tile_dir, tile_to_process)
        log.info(individual_tile_directory_path)

        try:
            filesystem_utilities.create_folder_structure_for_tiles(individual_tile_directory_path)
            #composite_dir = os.path.join(individual_tile_directory_path, r"composite")
            #composite_l1_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L1C")
            #composite_l2_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L2A")
            #composite_l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"cloud_masked")
            #change_image_dir = os.path.join(individual_tile_directory_path, r"images")
            #l1_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L1C")
            #l2_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L2A")
            #l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"images", r"cloud_masked")
            #categorised_image_dir = os.path.join(individual_tile_directory_path, r"output", r"classified")
            probability_image_dir = os.path.join(individual_tile_directory_path, r"output", r"probabilities")
            report_image_dir = os.path.join(individual_tile_directory_path, r"output", r"report_images")
            #quicklook_dir = os.path.join(individual_tile_directory_path, r"output", r"quicklooks")
            log.info("Successfully identified the subdirectory paths for this tile")
        except:
            log.error("ERROR: Tile subdirectory paths could not be created")
            sys.exit(1)

        # initialise tile log file
        tile_log_file = os.path.join(
            individual_tile_directory_path, 
            "log", 
            tile_to_process + ".log"
            )
        log.info("---------------------------------------------------------------")
        log.info(f"--- Redirecting log output to tile log: {tile_log_file}")
        log.info("---------------------------------------------------------------")
        tile_log = filesystem_utilities.init_log_acd(
            log_path=tile_log_file,
            logger_name="pyeo_"+tile_to_process
        )
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("---   PROCESSING START: {}   ---".format(tile_dir))
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Sending out change reports from the latest available report image."
        )

        #TODO: Search the local directories, probabilities and report_images
        #    for any report images for that tile
        
        #TODO: Modify search_term to match report image naming convention
        # e.g. report_20221104T110129_30UYD_20230915T105701.tif
        # or   archived_report_20221104T110129_30UYD_20230915T105701.zip
        search_term = (
            "report"
            + "_*_"
            + tile_to_process
            + "_*"
        )
        tile_log.info(
            "Searching for change report images in this tile directory " +
            f"containing: {search_term}."
            )
        file_list = (
            [
                os.path.join(probability_image_dir, f)
                for f in os.listdir(probability_image_dir)
            ]
            + [
                os.path.join(report_image_dir, f)
                for f in os.listdir(report_image_dir)
            ]
        )

        report_images = []
        for f in file_list:
            if search_term in f:
                tile_log.info(f"  Report image found: {f}")
                report_images.append(f)

        tile_log.info(
            f"{len(report_images)} matching report images found."
            )

        # Out of all files found, choose the one with the most recent date stamp
        
        
        # Read the list of registered users from a file specified in the ini file


        # Choose medium of sending out the reports - check acd_national for functions
        
        
        
        
        #************************************



        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Reports from the report image have been sent to users in the contact list.")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(" ")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("---             TILE PROCESSING END                           ---")
        tile_log.info("---------------------------------------------------------------")

        # process the next tile if more than one tile are specified at this point (for loop)

    # after the for loop ends, log that all tiles are complete:
    log.info(" ")
    log.info("---------------------------------------------------------------")
    log.info("---                  ALL PROCESSING END                      ---")
    log.info("---------------------------------------------------------------")
    return


if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Downloads and preprocesses Sentinel 2 images into median composites."
    )
    parser.add_argument(
        dest="config_path",
        action="store",
        default=r"pyeo_linux.ini",
        help="A path to a .ini file containing the specification for the job. See "
        "pyeo/pyeo_linux.ini for an example.",
    )
    parser.add_argument(
        "--tile",
        dest="tile_id",
        type=str,
        default="None",
        help="Overrides the geojson location with a" "Sentinel-2 tile ID location",
    )

    args = parser.parse_args()

    send_report(**vars(args))