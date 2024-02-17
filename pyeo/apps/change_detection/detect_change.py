"""
detect_change
-------------------------------------
An app for the detection of changes between new images and the 
initial cloud-free median composite from Sentinel-2, i.e. the baseline map.
It uses some of the ini file parameters but not the do_x flags.
"""

import argparse
import configparser
import cProfile
import datetime
#import geopandas as gpd
import glob
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
    classification,
    queries_and_downloads, 
    raster_manipulation
    )
from pyeo.filesystem_utilities import (
    zip_contents,
    unzip_contents,
    config_to_log
    )
from pyeo.acd_national import acd_roi_tile_intersection
from pyeo.vectorisation import (
                zonal_statistics, 
                merge_and_calculate_spatial, 
                vectorise_from_band, 
                clean_zero_nodata_vectorised_band
                )

gdal.UseExceptions()

def detect_change(config_path, tile_id="None"):
    """
    The main function that detects changes between the median composite 
        and the newly downloaded images with the parameters specified 
        in the ini file found in config_path.

    Args:

        config_path : string with the full path to the ini file or config file containing the
                        processing parameters

        tile_id : string with the Sentinel-2 tile ID. If not "None", this ID is used
                        instead of the region of interest file to define the area to be processed

    """

    #TODO: To speed up processing, limit the classification and change detection
    #  to forest areas only using https://glad.umd.edu/Potapov/TCC_2010/
    '''
    downloadfile = os.path.join(SAVE_DIR, file_name)
    # treecover files have filename pattern: 
    #   'treecover2010_70N_160E.tif'
    # coordinates are in steps of 10 degrees lat, lon
    #   from 0...80 lat and 0...170 lon
    #TODO: Determine the coordinates from the tile extent
    coords = '70N_160E'
    response = requests.get(
        'https://glad.umd.edu/Potapov/TCC_2010/treecover2010_'+coords+'.tif',
        stream=True
        )
    with open(downloadfile,'wb') as d:
        d.write(response.content)
    '''
    
    # read the ini file contents into a dictionary
    configparser.ConfigParser(allow_no_value=True)
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)

    ##########################################################
    # Initialisation
    ##########################################################
    
    # changes directory to pyeo_dir, enabling the use of relative paths from 
    #    the config file
    os.chdir(config_dict["pyeo_dir"])

    # check that log directory exists and create if not
    if not os.path.exists(config_dict["log_dir"]):
        os.makedirs(config_dict["log_dir"])

    # initialise log file
    log = filesystem_utilities.init_log_acd(
        log_path=os.path.join(config_dict["log_dir"], config_dict["log_filename"]),
        logger_name="pyeo_log",
    )

    # check conda directory exists
    if config_dict["environment_manager"] == "conda":
        conda_env_found = filesystem_utilities.conda_check(config_dict=config_dict, log=log)
        if not conda_env_found:
            log.error("Conda Environment Directory does not exist.")
            log.error("Ensure this exists before running pyeo.")
            log.error("Now exiting the pipeline.")
            sys.exit(1)

    log.info("---------------------------------------------------------------")
    log.info(f"Config file: {config_path}")
    log.info("---------------------------------------------------------------")

    config_to_log(config_dict, log)

    try:
        os.chdir(config_dict["pyeo_dir"]) # ensures pyeo is looking in the correct directory
        start_date = config_dict["start_date"]
        end_date = config_dict["end_date"]
        if end_date == "TODAY":
            end_date = datetime.date.today().strftime("%Y%m%d")
        #composite_start_date = config_dict["composite_start"]
        #composite_end_date = config_dict["composite_end"]
        cloud_cover = config_dict["cloud_cover"]
        #cloud_certainty_threshold = config_dict["cloud_certainty_threshold"]
        model_path = config_dict["model_path"]
        tile_dir = config_dict["tile_dir"]
        sen2cor_path = config_dict["sen2cor_path"]
        level_1_boundaries_path = config_dict["level_1_boundaries_path"]
        epsg = config_dict["epsg"]
        bands = config_dict["bands"]
        #resolution = config_dict["resolution_string"]
        out_resolution = config_dict["output_resolution"]
        buffer_size = config_dict["buffer_size_cloud_masking"]
        #buffer_size_composite = config_dict["buffer_size_cloud_masking_composite"]
        max_image_number = config_dict["download_limit"]
        faulty_granule_threshold = config_dict["faulty_granule_threshold"]
        #download_limit = config_dict["download_limit"]
        skip_existing = config_dict["do_skip_existing"]
        sieve = config_dict["sieve"]
        from_classes = config_dict["from_classes"]
        to_classes = config_dict["to_classes"]
        class_labels = config_dict["class_labels"]
        download_source = config_dict["download_source"]
        if download_source == "scihub":
            log.info("scihub API is the download source")
        if download_source == "dataspace":
            log.info("dataspace API is the download source")
        log.info("Faulty Granule Threshold is set to   : {}".format(
                config_dict['faulty_granule_threshold'])
                )
        log.info("    Files below this threshold will not be downloaded")
        log.info("Successfully read the processing arguments")
    except:
        log.error("Could not read the processing arguments from the ini file.")
        sys.exit(1)

    # check and read in credentials for downloading Sentinel-2 data
    credentials_path = config_dict["credentials_path"]
    if os.path.exists(credentials_path):
        try:
            credentials_conf = configparser.ConfigParser(allow_no_value=True)
            credentials_conf.read(credentials_path)
            credentials_dict = {}
            if download_source == "dataspace":
                log.info("Running download handler for " + download_source)
                credentials_dict["sent_2"] = {}
                credentials_dict["sent_2"]["user"] = credentials_conf["dataspace"]["user"]
                credentials_dict["sent_2"]["pass"] = credentials_conf["dataspace"]["pass"]
                sen_user = credentials_dict["sent_2"]["user"]
                sen_pass = credentials_dict["sent_2"]["pass"]
                #log.info(credentials_dict["sent_2"]["user"])
            else:
                if download_source == "scihub":
                    log.info("Running download handler for " + download_source)
                    credentials_dict["sent_2"] = {}
                    credentials_dict["sent_2"]["user"] = credentials_conf["sent_2"]["user"]
                    credentials_dict["sent_2"]["pass"] = credentials_conf["sent_2"]["pass"]
                    sen_user = credentials_dict["sent_2"]["user"]
                    sen_pass = credentials_dict["sent_2"]["pass"]
                else:
                    log.error("Invalid option selected for download_source in config file: " \
                                + download_source)
                    log.error("  download_source must be either \'dataspace\' or \'scihub\'.")
                    sys.exit(1)
        except:
            log.error("Could not open " + credentials_path)
            log.error("Create the file with your login credentials.")
            sys.exit(1)
        else:
            log.info("Credentials read from " + credentials_path)
    else:
        log.error(credentials_path + " does not exist.")
        log.error("Did you write the correct filepath in the config file?")
        sys.exit(1)

    os.chdir(config_dict["pyeo_dir"]) # ensures pyeo is looking in the correct directory
    if tile_id == "None":
        # if no tile ID is given by the call to the function, use the geometry file
        #   to get the tile ID list
        #tile_based_processing_override = False
        tilelist_filepath = acd_roi_tile_intersection(config_dict, log)
        tiles_to_process = list(pd.read_csv(tilelist_filepath)["tile"])
        # move filelist file from roi dir to main directory
        tilelist_filepath = shutil.move(
            tilelist_filepath, 
            os.path.join(
                config_dict["tile_dir"], 
                tilelist_filepath.split(os.path.sep)[-1])
        )
        log.info("Region of interest processing based on ROI file.")        
    else:
        # if a tile ID is specified, use that and do not use the tile intersection
        #   method to get the tile ID list
        #tile_based_processing_override = True
        tiles_to_process = [tile_id]
        log.info(f"Tile based processing: {tile_id}. Ignoring ROI file.")
        try:
            tilelist_filepath = os.path.join(
                    config_dict["tile_dir"],
                    "tile_list.txt"
                    )
            pd.DataFrame({"tile": tiles_to_process}).to_csv(
                tilelist_filepath, 
                header=True, 
                index=False)
        except:
            log.error(f"Could not write to {tilelist_filepath}")
    log.info(f"Saved Sentinel-2 tile ID list: {tilelist_filepath}")
    log.info(str(len(tiles_to_process)) + " Sentinel-2 tiles to process.")

    # iterate over the tiles
    for tile_to_process in tiles_to_process:
        log.info("Processing Sentinel-2 tile: " + tile_to_process)
        log.info("    See tile log file for details.")
        individual_tile_directory_path = os.path.join(tile_dir, tile_to_process)
        log.info(individual_tile_directory_path)

        try:
            filesystem_utilities.create_folder_structure_for_tiles(individual_tile_directory_path)
            composite_dir = os.path.join(individual_tile_directory_path, r"composite")
            #composite_l1_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L1C")
            #composite_l2_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L2A")
            #composite_l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"cloud_masked")
            #change_image_dir = os.path.join(individual_tile_directory_path, r"images")
            l1_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L1C")
            l2_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L2A")
            l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"images", r"cloud_masked")
            sieved_image_dir = os.path.join(individual_tile_directory_path, r"output", r"sieved")
            categorised_image_dir = os.path.join(individual_tile_directory_path, r"output", r"classified")
            probability_image_dir = os.path.join(individual_tile_directory_path, r"output", r"probabilities")
            reports_dir = os.path.join(individual_tile_directory_path, r"output", r"reports")
            quicklook_dir = os.path.join(individual_tile_directory_path, r"output", r"quicklooks")
            if start_date == "LATEST":
                report_file_name = [
                    f
                    for f in os.listdir(probability_image_dir)
                    if os.path.isfile(f) and f.startswith("report_") and f.endswith(".tif")
                ][0]
                report_file_path = os.path.join(probability_image_dir, report_file_name)
                after_timestamp = filesystem_utilities.get_change_detection_dates(
                    os.path.basename(report_file_path)
                )[-1]
                after_timestamp.strftime(
                    "%Y%m%d"
                )  
                # Returns the yyyymmdd string of the acquisition date from 
                #   which the latest classified image was derived
    
            if end_date == "TODAY":
                end_date = datetime.date.today().strftime("%Y%m%d")

            log.info("Successfully created the subdirectory paths for this tile")
        except:
            log.error("ERROR: Tile subdirectory paths could not be created")
            sys.exit(-1)

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
        tile_log.info("Folder structure build successfully finished for tile "+tile_to_process)
        tile_log.info("  Directory path created: "+individual_tile_directory_path)
        tile_log.info("")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(f"---  TILE PROCESSING START: {tile_to_process}  ---")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Detecting change between classified change detection images")
        tile_log.info("and the baseline median image composite.")
        tile_log.info("List of image bands: {}".format(bands))
        tile_log.info("List of class labels: {}".format(class_labels))
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Searching for change detection images.")
        if skip_existing:
            tile_log.info("Skipping existing change detection images if found.")

        if download_source == "dataspace":
            # convert date string to YYYY-MM-DD
            date_object = datetime.datetime.strptime(start_date, "%Y%m%d")
            dataspace_change_start = date_object.strftime("%Y-%m-%d")
            date_object = datetime.datetime.strptime(end_date, "%Y%m%d")
            dataspace_change_end = date_object.strftime("%Y-%m-%d")

            '''
            if not tile_based_processing_override:
                tiles_geom_path = os.path.join(config_dict["pyeo_dir"], \
                                    os.path.join(config_dict["geometry_dir"], \
                                    config_dict["s2_tiles_filename"]))
                tile_log.info("Path to the S2 tile information file: {}".format( \
                            os.path.abspath(tiles_geom_path)))
    
                #try opening the Sentinel-2 tile file as a geometry file (e.g. shape file)
                try:
                    tiles_geom = gpd.read_file(os.path.abspath(tiles_geom_path))
                except FileNotFoundError:
                    tile_log.error("Path to the S2 tile geometry does not exist: {}".format( \
                                os.path.abspath(tiles_geom_path)))

                tile_geom = tiles_geom[tiles_geom["Name"] == tile_to_process]
                tile_geom = tile_geom.to_crs(epsg=4326)
                geometry = tile_geom["geometry"].iloc[0]
                geometry = geometry.representative_point().wkt

                # attempt a geometry based query
                try:
                    dataspace_products_all = queries_and_downloads.query_dataspace_by_polygon(
                        max_cloud_cover=cloud_cover,
                        start_date=dataspace_change_start,
                        end_date=dataspace_change_end,
                        area_of_interest=geometry,
                        max_records=100,
                        log=tile_log
                    )
                except Exception as error:
                    tile_log.error("Query_dataspace_by_polygon received this error: {}".format(error))
            else:
            '''
            if 1<2:
                # attempt a tile ID based query
                try:
                    dataspace_products_all = queries_and_downloads.query_dataspace_by_tile_id(
                        max_cloud_cover=cloud_cover,
                        start_date=dataspace_change_start,
                        end_date=dataspace_change_end,
                        tile_id=tile_to_process,
                        max_records=100,
                        log=tile_log
                    )
                except Exception as error:
                    tile_log.error(f"Query_dataspace_by_tile received this error: {error}")

            titles = dataspace_products_all["title"].tolist()
            sizes = list()
            uuids = list()
            for elem in dataspace_products_all.itertuples(index=False):
                sizes.append(elem[-2]["download"]["size"])
                uuids.append(elem[-2]["download"]["url"].split("/")[-1])

            relative_orbit_numbers = dataspace_products_all["relativeOrbitNumber"].tolist()
            processing_levels = dataspace_products_all["processingLevel"].tolist()
            transformed_levels = ['Level-1C' if level == 'S2MSI1C' else 'Level-2A' for level in processing_levels]
            cloud_covers = dataspace_products_all["cloudCover"].tolist()
            begin_positions = dataspace_products_all["startDate"].tolist()
            statuses = dataspace_products_all["status"].tolist()
            scihub_compatible_df = pd.DataFrame({"title": titles,
                                                "size": sizes,
                                                "beginposition": begin_positions,
                                                "relativeorbitnumber": relative_orbit_numbers,
                                                "cloudcoverpercentage": cloud_covers,
                                                "processinglevel": transformed_levels,
                                                "uuid": uuids,
                                                "status": statuses})

            # check granule sizes on the server
            scihub_compatible_df["size"] = scihub_compatible_df["size"].apply(
                lambda x: round(float(x) * 1e-6, 2)
                )
            # reassign to match the scihub variable
            df_all = scihub_compatible_df


        if download_source == "scihub":

            try:
                products_all = queries_and_downloads.check_for_s2_data_by_date(
                    config_dict["tile_dir"],
                    start_date,
                    end_date,
                    conf=credentials_dict,
                    cloud_cover=cloud_cover,
                    tile_id=tile_to_process,
                    producttype=None,
                )

            except Exception as error:
                tile_log.error("check_for_s2_data_by_date failed:  {}".format(error))

            tile_log.info(
                "--> Found {} L1C and L2A products for the change detection:".format(
                    len(products_all)
                )
            )

            df_all = pd.DataFrame.from_dict(products_all, orient="index")

            # check granule sizes on the server
            df_all["size"] = (
                df_all["size"]
                .str.split(" ")
                .apply(lambda x: float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]])
            )

        if download_source == "scihub":
            min_granule_size = faulty_granule_threshold
        else:
            min_granule_size = 0  # Required for dataspace API which doesn't 
            # report size correctly (often reported as zero)

        df = df_all.query("size >= " + str(min_granule_size))

        tile_log.info(
            "Removed {} faulty scenes <{}MB in size from the list".format(
                len(df_all) - len(df), min_granule_size
            )
        )
        # find < threshold sizes, report to log
        df_faulty = df_all.query("size < " + str(min_granule_size))
        for r in range(len(df_faulty)):
            tile_log.info(
                "   {} MB: {}".format(
                    df_faulty.iloc[r, :]["size"], df_faulty.iloc[r, :]["title"]
                )
            )

        l1c_products = df[df.processinglevel == "Level-1C"]
        l2a_products = df[df.processinglevel == "Level-2A"]
        tile_log.info("    {} L1C products".format(l1c_products.shape[0]))
        tile_log.info("    {} L2A products".format(l2a_products.shape[0]))


        rel_orbits = np.unique(l1c_products["relativeorbitnumber"])
        if len(rel_orbits) > 0:
            if l1c_products.shape[0] > max_image_number / len(rel_orbits):
                tile_log.info(
                    "Capping the number of L1C products to {}".format(max_image_number)
                )
                tile_log.info(
                    "Relative orbits found covering tile: {}".format(rel_orbits)
                )
                uuids = []
                for orb in rel_orbits:
                    uuids = uuids + list(
                        l1c_products.loc[
                            l1c_products["relativeorbitnumber"] == orb
                        ].sort_values(by=["cloudcoverpercentage"], ascending=True)[
                            "uuid"
                        ][
                            : int(max_image_number / len(rel_orbits))
                        ]
                    )
                # keeps least cloudy n (max image number)
                l1c_products = l1c_products[l1c_products["uuid"].isin(uuids)]
                tile_log.info(
                    "    {} L1C products remain:".format(l1c_products.shape[0])
                )
                for product in l1c_products["title"]:
                    tile_log.info("       {}".format(product))
                tile_log.info("Number of L1C products for dataspace is {}".format(len(l1c_products['title'])))

        rel_orbits = np.unique(l2a_products["relativeorbitnumber"])
        if len(rel_orbits) > 0:
            if l2a_products.shape[0] > max_image_number / len(rel_orbits):
                tile_log.info(
                    "Capping the number of L2A products to {}".format(max_image_number)
                )
                tile_log.info(
                    "Relative orbits found covering tile: {}".format(rel_orbits)
                )
                uuids = []
                for orb in rel_orbits:
                    uuids = uuids + list(
                        l2a_products.loc[
                            l2a_products["relativeorbitnumber"] == orb
                        ].sort_values(by=["cloudcoverpercentage"], ascending=True)[
                            "uuid"
                        ][
                            : int(max_image_number / len(rel_orbits))
                        ]
                    )
                l2a_products = l2a_products[l2a_products["uuid"].isin(uuids)]
                tile_log.info(
                    "    {} L2A products remain:".format(l2a_products.shape[0])
                )
                for product in l2a_products["title"]:
                    tile_log.info("       {}".format(product))
                tile_log.info("Number of L2A products for dataspace is {}".format(
                    len(l2a_products['title'])
                    )
                )

        if l1c_products.shape[0] > 0 and l2a_products.shape[0] > 0:
            tile_log.info(
                "Filtering out L1C products that have the same 'beginposition'"+
                " time stamp as an existing L2A product."
            )
           
            if download_source == "scihub":
                (l1c_products,l2a_products,) = queries_and_downloads.filter_unique_l1c_and_l2a_data(
                                                    df,
                                                    log=tile_log
                                                    )

            if download_source == "dataspace":
                l1c_products = queries_and_downloads.filter_unique_dataspace_products(
                                l1c_products=l1c_products,
                                l2a_products=l2a_products, 
                                log=tile_log
                                )

        df = None

        tile_log.info(" {} L1C products for the change detection".format(
            len(l1c_products['title']))
            )
        tile_log.info(" {} L2A products for the change detection".format(
            len(l2a_products['title']))
            )
        tile_log.info("Successfully queried the L1C and L2A products for "+
                      "the change detection.")

        # Search the local directories, images/L2A and L1C, checking if 
        #    scenes have already been downloaded and/or processed whilst 
        #    checking their dir sizes
        if download_source == "scihub":
            if l1c_products.shape[0] > 0:
                tile_log.info(
                    "Checking for already downloaded and zipped L1C or L2A "+
                    "products and availability of matching L2A products for "+
                    "download."
                    )
                n = len(l1c_products)
                drop = []
                add = []
                for r in range(n):
                    id = l1c_products.iloc[r, :]["title"]
                    search_term = (
                        id.split("_")[2]
                        + "_"
                        + id.split("_")[3]
                        + "_"
                        + id.split("_")[4]
                        + "_"
                        + id.split("_")[5]
                    )
                    tile_log.info(
                        "Searching locally for file names containing: {}.".format(
                            search_term
                        )
                    )
                    file_list = (
                        [
                            os.path.join(l1_image_dir, f)
                            for f in os.listdir(l1_image_dir)
                        ]
                        + [
                            os.path.join(l2_image_dir, f)
                            for f in os.listdir(l2_image_dir)
                        ]
                        + [
                            os.path.join(l2_masked_image_dir, f)
                            for f in os.listdir(l2_masked_image_dir)
                        ]
                    )
                    for f in file_list:
                        if search_term in f:
                            tile_log.info("  Product already downloaded: {}".format(f))
                            drop.append(l1c_products.index[r])
                    search_term = (
                        "*"
                        + id.split("_")[2]
                        + "_"
                        + id.split("_")[3]
                        + "_"
                        + id.split("_")[4]
                        + "_"
                        + id.split("_")[5]
                        + "*"
                    )


                    tile_log.info(
                        "Searching on the data hub for files containing: {}.".format(
                            search_term
                        )
                    )
                    matching_l2a_products = queries_and_downloads._file_api_query(
                        user=sen_user,
                        passwd=sen_pass,
                        start_date=start_date,
                        end_date=end_date,
                        filename=search_term,
                        cloud=cloud_cover,
                        producttype="S2MSI2A",
                    )

                    matching_l2a_products_df = pd.DataFrame.from_dict(
                        matching_l2a_products, orient="index"
                    )

                    # 07/03/2023: Matt - Applied Ali's fix for converting 
                    # product size to MB to compare against faulty_grandule_threshold
                    if (
                        len(matching_l2a_products_df) == 1
                        and [
                            float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            for x in [matching_l2a_products_df["size"][0].split(" ")]
                        ][0]
                        > faulty_granule_threshold
                    ):
                        tile_log.info("Replacing L1C {} with L2A product:".format(id))
                        tile_log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )

                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                    if len(matching_l2a_products_df) == 0:
                        pass
                    if len(matching_l2a_products_df) > 1:
                        # check granule sizes on the server
                        matching_l2a_products_df["size"] = (
                            matching_l2a_products_df["size"]
                            .str.split(" ")
                            .apply(
                                lambda x: float(x[0])
                                * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            )
                        )
                        matching_l2a_products_df = matching_l2a_products_df.query(
                            "size >= " + str(faulty_granule_threshold)
                        )
                        if (
                            matching_l2a_products_df.iloc[0, :]["size"]
                            .str.split(" ")
                            .apply(
                                lambda x: float(x[0])
                                * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            )
                            > faulty_granule_threshold
                        ):
                            tile_log.info("Replacing L1C {} with L2A product:".format(id))
                            tile_log.info(
                                "              {}".format(
                                    matching_l2a_products_df.iloc[0, :]["title"]
                                )
                            )
                            drop.append(l1c_products.index[r])
                            add.append(matching_l2a_products_df.iloc[0, :])
                if len(drop) > 0:
                    l1c_products = l1c_products.drop(index=drop)
                if len(add) > 0:
                    # l2a_products = l2a_products.append(add)
                    add = pd.DataFrame(add)
                    l2a_products = pd.concat([l2a_products, add])

                tile_log.info("Successfully searched for the L2A counterparts for\
                              the L1C products for the change detection.")
                
            # here, dataspace and scihub derived l1c_products and l2a_products lists are the "same"
            l2a_products = l2a_products.drop_duplicates(subset="title")
            tile_log.info(
                "    {} L1C products remaining for download".format(
                    l1c_products.shape[0]
                )
            )
            tile_log.info(
                "    {} L2A products remaining for download".format(
                    l2a_products.shape[0]
                )
            )

        ##################################
        # Download the L1C images
        ##################################
        if l1c_products.shape[0] > 0:
            tile_log.info("Downloading Sentinel-2 L1C products from {}".format(download_source))

            if download_source == "scihub":

                queries_and_downloads.download_s2_data_from_df(
                    l1c_products,
                    l1_image_dir,
                    l2_image_dir,
                    source="scihub",
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )

            if download_source == "dataspace":

                queries_and_downloads.download_s2_data_from_dataspace(
                    product_df=l1c_products,
                    l1c_directory=l1_image_dir,
                    l2a_directory=l2_image_dir,
                    dataspace_username=sen_user,
                    dataspace_password=sen_pass,
                    log=tile_log
                )

            tile_log.info("Successfully downloaded the Sentinel-2 L1C products")

            tile_log.info("Atmospheric correction of L1C image products with sen2cor.")
            raster_manipulation.atmospheric_correction(
                l1_image_dir,
                l2_image_dir,
                sen2cor_path,
                delete_unprocessed_image=False,
                log=tile_log
            )

        tile_log.info("Successful atmospheric correction of the Sentinel-2 L1C"+
                      " products to L2A.")

        # Download the L2A images
        if l2a_products.shape[0] > 0:
            tile_log.info("Downloading Sentinel-2 L2A products.")

            if download_source == "scihub":

                queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    l1_image_dir,
                    l2_image_dir,
                    source="scihub",
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
            if download_source == "dataspace":

                queries_and_downloads.download_s2_data_from_dataspace(
                    product_df=l2a_products,
                    l1c_directory=l1_image_dir,
                    l2a_directory=l2_image_dir,
                    dataspace_username=sen_user,
                    dataspace_password=sen_pass,
                    log=tile_log
                )

        # check for incomplete L2A downloads
        incomplete_downloads, sizes = raster_manipulation.find_small_safe_dirs(
            l2_image_dir, threshold=faulty_granule_threshold * 1024 * 1024
        )
        if len(incomplete_downloads) > 0:
            for index, safe_dir in enumerate(incomplete_downloads):
                if sizes[
                    index
                ] / 1024 / 1024 < faulty_granule_threshold and os.path.exists(safe_dir):
                    tile_log.warning("Found likely incomplete download of size {} MB: {}".format(
                            str(round(sizes[index] / 1024 / 1024)), safe_dir))

        tile_log.info(
            "---------------------------------------------------------------"
            )
        tile_log.info("Image download and atmospheric correction for change "+
                      "detection is complete.")
        tile_log.info(
            "---------------------------------------------------------------"
            )

        # Housekeeping
        if config_dict["do_delete"]:
            tile_log.info(
                "---------------------------------------------------------------"
                )
            tile_log.info("Deleting downloaded L1C images for change detection.")
            tile_log.info("Keeping only derived L2A products.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            directory = l1_image_dir
            tile_log.info("Deleting {}".format(directory))
            shutil.rmtree(directory)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Deletion of L1C images complete.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
        else:
            if config_dict["do_zip"]:
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Zipping downloaded L1C images for change detection after atmospheric correction")
                tile_log.info("---------------------------------------------------------------")
                zip_contents(l1_image_dir)
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Zipping complete")
                tile_log.info("---------------------------------------------------------------")

        tile_log.info("Housekeeping after download successfully finished")

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Applying simple cloud, cloud shadow and haze mask based on")
        tile_log.info("SCL files and stacking the masked band raster files.")
        tile_log.info("---------------------------------------------------------------")

        directory = l2_masked_image_dir
        masked_file_paths = [
            f
            for f in os.listdir(directory)
            if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
        ]

        directory = l2_image_dir
        l2a_zip_file_paths = [f for f in os.listdir(directory) if f.endswith(".zip")]

        if len(l2a_zip_file_paths) > 0:
            for f in l2a_zip_file_paths:
                # check whether the zipped file has already been cloud masked
                zip_timestamp = filesystem_utilities.get_image_acquisition_time(
                    os.path.basename(f)
                ).strftime("%Y%m%dT%H%M%S")
                if any(zip_timestamp in f for f in masked_file_paths):
                    continue
                else:
                    # extract it if not
                    unzip_contents(
                        os.path.join(l2_image_dir, f),
                        ifstartswith="S2",
                        ending=".SAFE",
                    )

        directory = l2_image_dir
        l2a_safe_file_paths = [
            f
            for f in os.listdir(directory)
            if f.endswith(".SAFE") and os.path.isdir(os.path.join(directory, f))
        ]

        files_for_cloud_masking = []
        if len(l2a_safe_file_paths) > 0:
            for f in l2a_safe_file_paths:
                # check whether the L2A SAFE file has already been cloud masked
                safe_timestamp = filesystem_utilities.get_image_acquisition_time(
                    os.path.basename(f)
                ).strftime("%Y%m%dT%H%M%S")
                if any(safe_timestamp in f for f in masked_file_paths):
                    continue
                else:
                    # add it to the list of files to do if it has not been cloud masked yet
                    files_for_cloud_masking = files_for_cloud_masking + [f]
    
        if len(files_for_cloud_masking) == 0:
            tile_log.info("No L2A images found for cloud masking. They may already have been done.")
        else:
            raster_manipulation.apply_scl_cloud_mask(
                l2_image_dir,
                l2_masked_image_dir,
                scl_classes=[0, 1, 2, 3, 8, 9, 10, 11],
                buffer_size=buffer_size,
                bands=bands,
                out_resolution=out_resolution,
                haze=None,
                epsg=epsg,
                skip_existing=skip_existing)
            tile_log.info("Cloud-masking successfully finished")

        '''
        Apply Processing Baseline Offset
        Before Sentinel-2 imagery is provided to the user as L1C or L2A formats, the raw imagery (L0) 
        are processed by the ESA Copernicus Ground Segment. The algorithms used in the processing 
        baseline, are indicated by the field N0XXX in the product title and the changes introduced 
        by each processing baseline iteration are listed here.
        The advent of processing baseline N0400 introduced an offset of -1000 in the spectral 
        reflectance values. Therefore, to ensure that the spectral reflectance of imagery before 
        and after N0400 can be compared, we apply the offset correction of +1000.
        '''

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting cloud masked L2A images for change detection.")
        tile_log.info("---------------------------------------------------------------")

        raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
            l2_masked_image_dir, l2_masked_image_dir)

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting of cloud masked L2A images for change detection complete.")
        tile_log.info("---------------------------------------------------------------")


        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Producing quicklooks.")
            tile_log.info("---------------------------------------------------------------")
            dirs_for_quicklooks = [l2_masked_image_dir]
            for main_dir in dirs_for_quicklooks:
                files = [
                    f.path
                    for f in os.scandir(main_dir)
                    if f.is_file() and os.path.basename(f).endswith(".tif")
                ]
                if len(files) == 0:
                    tile_log.warning("No images found in {}.".format(main_dir))
                else:
                    for f in files:
                        quicklook_path = os.path.join(
                            quicklook_dir,
                            os.path.basename(f).split(".")[0] + ".png",
                        )
                        if os.path.isfile(quicklook_path) and skip_existing:
                            tile_log.info(
                                "Quicklook already exists - skipping: "+
                                f"{quicklook_path}"
                                )
                        else:
                            tile_log.info(f"Creating quicklook: {quicklook_path}")
                            raster_manipulation.create_quicklook(
                                f,
                                quicklook_path,
                                width=512,
                                height=512,
                                format="PNG",
                                bands=[3, 2, 1],
                                scale_factors=[[0, 2000, 0, 255]],
                            )
            tile_log.info("Quicklooks complete.")
        else:
            tile_log.info("Quicklook option disabled in ini file.")


        if config_dict["do_zip"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info(
                "Zipping downloaded L2A images for change detection after cloud masking and band stacking"
            )
            tile_log.info(
                "---------------------------------------------------------------"
            )
            zip_contents(l2_image_dir)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Zipping complete")
            tile_log.info(
                "---------------------------------------------------------------"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
        # Classify each L2A image and the baseline composite

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tile_log.info(
                "---------------------------------------------------------------"
                )
            tile_log.info(
                "Classify a land cover map for each L2A image and composite "+
                "image using a saved model"
            )
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Model used: {}".format(model_path))
            if skip_existing:
                tile_log.info("Skipping existing classification images if found.")
            
            classification.classify_directory(
                composite_dir,
                model_path,
                categorised_image_dir,
                prob_out_dir=None,
                apply_mask=False,
                out_type="GTiff",
                chunks=config_dict["chunks"],
                skip_existing=skip_existing,
                log=tile_log
            )
            classification.classify_directory(
                l2_masked_image_dir,
                model_path,
                categorised_image_dir,
                prob_out_dir=None,
                apply_mask=False,
                out_type="GTiff",
                chunks=config_dict["chunks"],
                skip_existing=skip_existing,
                log=tile_log
            )
    
            for root, dirs, files in os.walk(categorised_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
            if len(all_tiffs) > 0:
                tile_log.info(
                    "Compressing tiff files in directory {} and all subdirectories".format(
                        categorised_image_dir
                    )
                )
                for this_tiff in all_tiffs:
                    raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), 
                        os.path.join(root, this_tiff),
                        tile_log
                    )
    
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Classification of all images is complete.")
            tile_log.info("---------------------------------------------------------------")

        if config_dict["do_quicklooks"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Producing quicklooks.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            dirs_for_quicklooks = [categorised_image_dir]
            for main_dir in dirs_for_quicklooks:
                files = [
                    f.path
                    for f in os.scandir(main_dir)
                    if f.is_file()
                    and os.path.basename(f).endswith(".tif")
                    and "class" in os.path.basename(f)
                ]  # do classification images only
                if len(files) == 0:
                    tile_log.warning("No images found in {}.".format(main_dir))
                else:
                    for f in files:
                        quicklook_path = os.path.join(
                            quicklook_dir,
                            os.path.basename(f).split(".")[0] + ".png",
                            )
                        if os.path.isfile(quicklook_path) and skip_existing:
                            tile_log.info(
                                "Quicklook already exists - skipping: "+
                                f"{quicklook_path}"
                                )  
                        else:
                            tile_log.info(f"Creating quicklook: {quicklook_path}")
                            raster_manipulation.create_quicklook(
                                f, 
                                quicklook_path, 
                                width=512, 
                                height=512,
                                format="PNG"
                                )
            tile_log.info("Quicklook creation complete.")

        # ------------------------------------------------------------------------
        # Pair up the class images with the composite baseline map
        # and identify all pixels with the change between groups of classes of interest.
        # Optionally applies a sieve filter to the class images if specified in the ini file.
        # Confirms detected changes by NDVI differencing.
        #
        # The overall Change Detection can be summarised as this:
        # - PyEO first looks for the composite and change imagery classifications 
        #   and orders them by most recent.
        # - Then, it searches for existing report files created from previous 
        #   PyEO runs and archive them, moving them to an archived folder.
        # - Then it creates the change report by sequentially comparing the 
        #   classified change imagery against the classified baseline composite.
        # - Once finished, PyEO does some housekeeping, compressing unneeded files.
        #
        # To perform Change Detection, we take the Classified Change Imagery 
        # and compare it with the Classified Baseline Composite.
        # Because are concerned with monitoring deforestation for our Change 
        # Detection, `pyeo` examines whether any forest classes (*classes 1, 11
        # and 12*) change to non-forest classes (*classes 3, 4, 5 and 13*).
        # As new change imagery becomes available (*as deforestation monitoring
        # is an iterative process through time*), these change images are 
        # classified and compared to the baseline again.
        #

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Creating change layers from stacked class images.")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Changes of interest:")
        tile_log.info("  from any of the classes {}".format(from_classes))
        tile_log.info("  to   any of the classes {}".format(to_classes))

        # optionally sieve the class images
        if sieve > 0:
            tile_log.info("Applying sieve to classification outputs.")
            raster_manipulation.sieve_directory(
                in_dir=categorised_image_dir,
                out_dir=sieved_image_dir,
                neighbours=8,
                sieve=sieve,
                out_type="GTiff",
                skip_existing=skip_existing,
            )
            # if sieve was chosen, work with the sieved class images
            class_image_dir = sieved_image_dir
        else:
            # if sieve was not chosen, work with the original class images
            class_image_dir = categorised_image_dir
    
        # get all image paths in the classification maps directory except the class composites
        class_image_paths = [
            f.path
            for f in os.scandir(class_image_dir)
            if f.is_file() and f.name.endswith(".tif") and not "composite_" in f.name
        ]
        if len(class_image_paths) == 0:
            log.warning(f"No class tif images found in {class_image_dir}")
        else:
            # sort class images by image acquisition date
            class_image_paths = list(
                filter(filesystem_utilities.get_image_acquisition_time, class_image_paths)
            )
            class_image_paths.sort(
                key=lambda x: filesystem_utilities.get_image_acquisition_time(x)
            )
            for index, image in enumerate(class_image_paths):
                tile_log.info(f"{index}: {image}")
        
            # find the latest available composite
            try:
                latest_composite_name = filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(composite_dir)
                        if image_name.endswith(".tif")
                    ],
                    recent_first=True,
                )[0]
                latest_composite_path = os.path.join(composite_dir, latest_composite_name)
                tile_log.info("Most recent existing composite:")
                tile_log.info(f"  {latest_composite_path}")
            except IndexError:
                tile_log.critical(
                    "Latest composite not found. The first time you run this script, you need to include the "
                    "--build-composite flag to create a base composite to work off. If you have already done this,"
                    "check that the earliest dated image in your images/merged folder is later than the earliest"
                    " dated image in your composite/ folder."
                )
                sys.exit(1)
        
            latest_class_composite_path = os.path.join(
                class_image_dir,
                [
                    f.path
                    for f in os.scandir(class_image_dir)
                    if f.is_file()
                    and os.path.basename(latest_composite_path)[:-4] in f.name
                    and f.name.endswith(".tif")
                ][0],
            )
        
            if not os.path.exists(latest_class_composite_path):
                tile_log.critical(
                    "Latest class composite not found. The first time you run this script, you need to include the "
                    "--build-composite flag to create a base composite to work off. If you have already done this,"
                    "check that the earliest dated image in your images/merged folder is later than the earliest"
                    " dated image in your composite/ folder. Then, you need to run the --classify option."
                )
                sys.exit(1)
        
            # set the name of the report file
            before_timestamp = filesystem_utilities.get_change_detection_dates(
                os.path.basename(latest_class_composite_path)
            )[0]
            # Timestamp report with the date of most recent classified image that 
            #   contributes to it
            after_timestamp = filesystem_utilities.get_image_acquisition_time(
                os.path.basename(class_image_paths[-1])
            )
            # Previously:
            # gets timestamp of the earliest change image of those available in class_image_path
            # after_timestamp  = pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(class_image_paths[0]))
            output_product = os.path.join(
                probability_image_dir,
                "report_{}_{}_{}.tif".format(
                    before_timestamp.strftime("%Y%m%dT%H%M%S"),
                    tile_to_process,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
    
            # check whether a report file of that name already exists
            if os.path.isfile(output_product):
                tile_log.info(f"Report file already exists:  {output_product}")
            else:
                tile_log.info(f"Report file name will be:  {output_product}")
    
            # if an earlier report file exists, archive it
            n_report_files = len(
                [
                    f
                    for f in os.scandir(probability_image_dir)
                    if f.is_file()
                    and f.name.startswith("report_")
                    and f.name.endswith(".tif")
                ]
            )
    
            if n_report_files > 0:
                output_products_existing = [
                    f.path
                    for f in os.scandir(probability_image_dir)
                    if f.is_file()
                    and f.name.startswith("report_")
                    and f.name.endswith(".tif")
                ]
                for output_product_existing in output_products_existing:
                    # do not archive the current report image file
                    if output_product_existing != output_product:
                        tile_log.info(
                            "Found existing earlier report image product: "+
                            f" {output_product_existing}"
                            )
            
                        output_product_existing_archived = os.path.join(
                            os.path.dirname(output_product_existing),
                            "archived_" + os.path.basename(output_product_existing),
                        )
                        tile_log.info(
                            "Renaming existing report image product to: "+
                            f" {output_product_existing_archived}"
                        )
                        os.rename(
                            output_product_existing, 
                            output_product_existing_archived
                            )
        
            # find change patterns in the stack of classification images
            for index, image in enumerate(class_image_paths):
                before_timestamp = filesystem_utilities.get_change_detection_dates(
                    os.path.basename(latest_class_composite_path)
                )[0]
                after_timestamp = filesystem_utilities.get_image_acquisition_time(
                    os.path.basename(image)
                )
                tile_log.info(
                    f"Processing class image {index+1} of "+
                    f"{len(class_image_paths)} - filename: {image}"
                    )
                #tile_log.info("  early time stamp: {}".format(before_timestamp))
                #tile_log.info("  late  time stamp: {}".format(after_timestamp))
                change_raster = os.path.join(
                    probability_image_dir,
                    "change_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_to_process,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                
                # skip if change maps already exist and skip_existing is True
                if os.path.isfile(change_raster) and skip_existing:
                    tile_log.info(
                        f"  Change raster file already exists:   {change_raster}"
                    )
                    tile_log.info("    Skipping change raster production.")
                else:            
                    tile_log.info(
                        f"  Change raster file to be created:   {change_raster}"
                    )
        
                    dNDVI_raster = os.path.join(
                        probability_image_dir,
                        "dNDVI_{}_{}_{}.tif".format(
                            before_timestamp.strftime("%Y%m%dT%H%M%S"),
                            tile_to_process,
                            after_timestamp.strftime("%Y%m%dT%H%M%S"),
                        ),
                    )
                    tile_log.info(
                        "  dNDVI raster file to be created: {}".format(dNDVI_raster)
                    )
        
                    NDVI_raster = os.path.join(
                        probability_image_dir,
                        "NDVI_{}_{}_{}.tif".format(
                            before_timestamp.strftime("%Y%m%dT%H%M%S"),
                            tile_to_process,
                            after_timestamp.strftime("%Y%m%dT%H%M%S"),
                        ),
                    )
                    #tile_log.info(
                    #    "  I.R. NDVI raster file of change image to be created: {}".format(
                    #        NDVI_raster
                    #    )
                    #)
                    
                    # The following function call looks for changes from class 
                    #   'change_from' in the composite to any of the 
                    #   'change_to_classes' in the change images. Pixel values are 
                    #   the acquisition date of the detected change of interest or 
                    #   zero.
                    # Applying check whether dNDVI < -0.2, i.e. greenness has 
                    #   decreased over changed areas
            
                    # TODO: In change_from_class_maps(), add a flag (e.g. -1) whether a pixel was a cloud in the later image.
                    tile_log.info("Update of the report image product based on change detection image.")
                    raster_manipulation.change_from_class_maps(
                        old_class_path=latest_class_composite_path,
                        new_class_path=image,
                        change_raster=change_raster,
                        dNDVI_raster=dNDVI_raster,
                        NDVI_raster=NDVI_raster,
                        change_from=from_classes,
                        change_to=to_classes,
                        report_path=output_product,
                        skip_existing=skip_existing,
                        old_image_dir=composite_dir,
                        new_image_dir=l2_masked_image_dir,
                        viband1=4,
                        viband2=3,
                        dNDVI_threshold=-0.2,
                        log=tile_log,
                    )
        
            for root, dirs, files in os.walk(probability_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
            if len(all_tiffs) > 0:
                tile_log.info(
                    f"Compressing tiff files in {probability_image_dir} and all subdirectories"
                    )
                for this_tiff in all_tiffs:
                    raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), 
                        os.path.join(root, this_tiff),
                        tile_log
                    )
        
            for root, dirs, files in os.walk(sieved_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
            if len(all_tiffs) > 0:
                tile_log.info(
                    f"Compressing tiff files in {sieved_image_dir} and all subdirectories"
                    )
                for this_tiff in all_tiffs:
                    raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), 
                        os.path.join(root, this_tiff),
                        tile_log
                    )
    
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Post-classification change detection complete.")
            tile_log.info("---------------------------------------------------------------")
        
            '''    
            #do_dev is now depracated
            if not config_dict["do_dev"]:
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info(
                    "Creating aggregated report file. Deprecated in the development version."
                )
                tile_log.info(
                    "---------------------------------------------------------------"
                )
    
                date_image_paths = [
                    f.path
                    for f in os.scandir(probability_image_dir)
                    if f.is_file() and f.name.endswith(".tif") and "change_" in f.name
                ]
                if len(date_image_paths) == 0:
                    raise FileNotFoundError(
                        "No class images found in {}.".format(categorised_image_dir)
                    )
        
                before_timestamp = filesystem_utilities.get_change_detection_dates(
                    os.path.basename(latest_class_composite_path)
                )[0]
                after_timestamp = filesystem_utilities.get_image_acquisition_time(
                    os.path.basename(class_image_paths[-1])
                )
                output_product = os.path.join(
                    probability_image_dir,
                    "report_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_to_process,
                        # tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                tile_log.info("Combining date maps: {}".format(date_image_paths))
                raster_manipulation.combine_date_maps(date_image_paths, output_product)
            '''    
        
            tile_log.info(
                f"Report image product completed / updated: {output_product}"
            )
                
            if config_dict["do_all"] or config_dict["do_vectorise"]:
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Starting Vectorisation of the Change Report Rasters " +
                              f"of tile: {tile_to_process}")
                tile_log.info("---------------------------------------------------------------")
                # get all report tif file names that are within the root_dir with 
                #   search pattern from the probabilities subdirectory
                report_tif_pattern = f"{os.path.join(probability_image_dir, 'report*.tif')}"
                tile_log.info(f"Searching for {report_tif_pattern}...")
                change_report_paths = glob.glob(report_tif_pattern)
                #tile_log.info(f"change report paths from first search: {change_report_paths}")
                '''                
                # ... and from the reports subdirectory
                report_tif_pattern = f"{os.path.join(reports_dir, 'report*.tif')}"
                tile_log.info(f"Searching for {report_tif_pattern}...")
                for g in glob.glob(report_tif_pattern):
                    change_report_paths.append(g)
                '''                
                if len(change_report_paths) == 0:
                    tile_log.error("No change report image path(s) found.")
                    tile_log.error("They may have been zipped.")
                else:
                    tile_log.info("Change report image path(s) found:")
                    for change_report_path in change_report_paths:
                        tile_log.info(f"  {change_report_path}")
                    tile_log.info("")
        
                    # iterate over the list of report image files
                    for change_report_path in change_report_paths:
                        #tile_log.info(f"change_report_path = {change_report_path}")
        
                        # skip if vector file exists?
                        #if os.path.exists(change_report_path[:-4]+".shp"):
                        #    tile_log.info(f"Skipping. Found {change_report_path[:-4]}.shp")
                        #else:
        
                        #tile_log.info("calling vectorise_from_band")
        
                        # band 15 in pyeo 1.0 was band=6 in pyeo 0.9
                        path_vectorised_binary = vectorise_from_band(
                            change_report_path=change_report_path,
                            band=15,
                            log=tile_log
                        )
        
                        #tile_log.info("calling clean_zero_nodata...")
        
                        path_vectorised_binary_filtered = clean_zero_nodata_vectorised_band(
                            vectorised_band_path=path_vectorised_binary,
                            log=tile_log
                        )
                
                        #tile_log.info(f"vectorised_file_path = {path_vectorised_binary_filtered}")
        
                        #tile_log.info("calling zonal_stats")
        
                        # band 5 in pyeo 1.0 was band 2 in pyeo 0.9
                        rb_ndetections_zstats_df = zonal_statistics(
                            raster_path=change_report_path,
                            shapefile_path=path_vectorised_binary_filtered,
                            report_band=5,
                            log=tile_log
                            )
                    
                        # band 9 in pyeo 1.0 was band 5 in pyeo 0.9
                        rb_confidence_zstats_df = zonal_statistics(
                            raster_path=change_report_path,
                            shapefile_path=path_vectorised_binary_filtered,
                            report_band=9,
                            log=tile_log
                        )
                    
                        # band 4 in pyeo 1.0 was band 7 in pyeo 0.9
                        rb_first_changedate_zstats_df = zonal_statistics(
                            raster_path=change_report_path,
                            shapefile_path=path_vectorised_binary_filtered,
                            report_band=4,
                            log=tile_log
                        )

                        empty_zonal_stats = False
                        if rb_ndetections_zstats_df is None:
                            tile_log.warning("Empty zonal statistics dataframe"+
                                             " for number of detections.")
                            empty_zonal_stats = True
                        if rb_confidence_zstats_df is None:
                            tile_log.warning("Empty zonal statistics dataframe"+
                                             " for confidence layer.")
                            empty_zonal_stats = True
                        if rb_first_changedate_zstats_df is None:
                            tile_log.warning("Empty zonal statistics dataframe."+
                                             "for first date of change detected.")
                            empty_zonal_stats = True

                        if empty_zonal_stats:
                            pass
                        else:
                            # table joins, area, lat lon, county
                            #tile_log.info("calling merge_and_calculate_spatial")
                            output_vector_products = merge_and_calculate_spatial(
                                    rb_ndetections_zstats_df=rb_ndetections_zstats_df,
                                    rb_confidence_zstats_df=rb_confidence_zstats_df,
                                    rb_first_changedate_zstats_df=rb_first_changedate_zstats_df,
                                    path_to_vectorised_binary_filtered=path_vectorised_binary_filtered,
                                    write_csv=False,
                                    write_shapefile=True,
                                    write_kml=False,
                                    write_pkl=False,
                                    change_report_path=change_report_path,
                                    log=tile_log,
                                    epsg=epsg,
                                    level_1_boundaries_path=level_1_boundaries_path,
                                    tileid=tile_to_process,
                                    delete_intermediates=True
                                    )
            
                            #tile_log.info(f"Returned output vector file paths: {output_vector_products}")
        
                    tile_log.info("Compressing the report image.")
                    raster_manipulation.compress_tiff(output_product, 
                                                      output_product,
                                                      tile_log
                                                      )
        
                    # log output file names and move from probabilities dir to reports dir
                    # move all shapefile related files and tables, pickle objects,
                    #    kml files and csv tables to reports dir if they exist
                    if len(output_vector_products) > 0:
                        file_endings = ["prj","shp","shx","dbf","cpg", "kml", "pkl", "csv"]
                        for i, f in enumerate(output_vector_products):
                            filepath_start = f.split(".")[0]
                            for fe in file_endings:
                                this_file =  os.path.join(probability_image_dir, 
                                                          filepath_start+"."+fe
                                                          ) 
                                if os.path.isfile(this_file):
                                    shutil.move(
                                        this_file,
                                        reports_dir
                                        )
                                    tile_log.info(f"  Moved {f} to reports dir: {this_file}")
                    else:
                        tile_log.error("  List of created output vector files is empty.")
    
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Vectorisation complete")
                tile_log.info("---------------------------------------------------------------")
    
            if config_dict["do_delete"]:
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info(
                    "Deleting intermediate class images used in change detection."
                )
                tile_log.info(
                    "They can be recreated from the cloud-masked, band-stacked " + 
                    "L2A images and the saved model."
                )
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                directories = [
                    categorised_image_dir,
                    sieved_image_dir,
                    probability_image_dir,
                ]
                for directory in directories:
                    paths = [f for f in os.listdir(directory)]
                    for f in paths:
                        # keep the classified composite layers and the report image
                        #   product for the next change detection
                        if not f.startswith("composite_") and not f.startswith("report_"):
                            tile_log.info("Deleting {}".format(os.path.join(directory, f)))
                            if os.path.isdir(os.path.join(directory, f)):
                                shutil.rmtree(os.path.join(directory, f))
                            else:
                                os.remove(os.path.join(directory, f))
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Deletion of intermediate file products complete.")
                tile_log.info(
                    "---------------------------------------------------------------"
                )
            else:
                if config_dict["do_zip"]:
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    tile_log.info(
                        "Zipping intermediate class images used in change detection"
                    )
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    directories = [categorised_image_dir, sieved_image_dir]
                    for directory in directories:
                        zip_contents(
                            directory, notstartswith=["composite_", "report_"]
                        )
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    tile_log.info("Zipping complete")
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
        
            tile_log.info("---------------------------------------------------------------")
            tile_log.info(
                "Change detection and report image product updating, file "+
                "compression, zipping and deletion of intermediate file products "+
                "(if selected) are complete."
                )
            tile_log.info("---------------------------------------------------------------")
        
            if config_dict["do_delete"]:
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Deleting temporary directories starting with 'tmp*'")
                tile_log.info("These can be left over from interrupted processing runs.")
                tile_log.info("---------------------------------------------------------------")
                for root, dirs, files in os.walk(tile_dir):
                    temp_dirs = [d for d in dirs if d.startswith("tmp")]
                    for temp_dir in temp_dirs:
                        tile_log.info("Deleting {}".format(os.path.join(root, temp_dir)))
                        if os.path.isdir(os.path.join(tile_dir, f)):
                            shutil.rmtree(os.path.join(tile_dir, f))
                        else:
                            tile_log.warning(
                                "This should not have happened. "+
                                f"{os.path.join(root, temp_dir)} is not a "+
                                "directory. Skipping deletion."
                            )
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Deletion of temporary directories complete.")
                tile_log.info("---------------------------------------------------------------")

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(f"---  TILE PROCESSING END: {tile_to_process} ---")
        tile_log.info("---------------------------------------------------------------")

        # process the next tile if more than one tile are specified at this point (for loop)

    # after the for loop ends, log that all tiles are complete:
    log.info("---------------------------------------------------------------")
    log.info("---                  ALL PROCESSING END                      ---")
    log.info("---------------------------------------------------------------")
    return tile_dir


if __name__ == "__main__":
    # save runtime statistics of this code
    profiler = cProfile.Profile()
    profiler.enable()

    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Downloads and preprocesses Sentinel 2 images for change detection.\
            and classifies them using a machine learning model. Performs change\
            detection against a baseline median image composite. Generates a\
            report image file and optionally vectorises it if selected in the\
            ini file."
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
        help="Overrides the geojson location with a Sentinel-2 tile ID location",
    )

    args = parser.parse_args()

    tile_dir = detect_change(**vars(args))
        
    profiler.disable()
    f = os.path.join(tile_dir, "detect_change")
    i = 1
    if os.path.exists(f+".prof"):
        while os.path.exists(f+"_"+str(i)+".prof"):
            i = i + 1
        f = f+"_"+str(i)
    f = f + ".prof"
    profiler.dump_stats(f)
    print(f"runtime analysis saved to {f}")
    print("Run snakeviz over the profile file to interact with the profile information:")
    print(f">snakeviz {f}")