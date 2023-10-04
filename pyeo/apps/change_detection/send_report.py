"""
detect_change
-------------------------------------
An app for the detection of changes between new images and the 
initial cloud-free median composite from Sentinel-2, i.e. the baseline map.
It uses some of the ini file parameters but not the do_x flags.
"""

import argparse
import configparser
import datetime
import geopandas as gpd
import pandas as pd
import json
import numpy as np
import os
from osgeo import gdal
import shutil
import sys
import warnings
import zipfile
from pyeo import (filesystem_utilities, queries_and_downloads, raster_manipulation)
from pyeo.acd_national import (acd_initialisation,
                                 acd_config_to_log,
                                 acd_roi_tile_intersection)

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

    def zip_contents(directory, notstartswith=None):
        '''
        A function that compresses all files in a directory in zip file format.
    
        Args:
    
            directory : string with the full path to the files that will all be compressed in separate zip files
    
            notstartswith (optional) : string specifying the start of filenames that will be skipped and not zipped
    
        '''
        paths = [f for f in os.listdir(directory) if not f.endswith(".zip")]
        for f in paths:
            do_it = True
            if notstartswith is not None:
                for i in notstartswith:
                    if f.startswith(i):
                        do_it = False
                        log.info(
                            "Skipping file that starts with '{}':   {}".format(i, f)
                        )
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
    
    
    def unzip_contents(directory, ifstartswith=None, ending=None):
        '''
        A function that uncompresses a zip directory or file.
    
        Args:
    
            directory : string with the full path to the zip directory or file that will be decompressed
    
            ifstartswith (optional) : string specifying a pattern for the start of a filename
    
            ending (optional) : string specifying the ending that will be attached to the directory name 
                                of the unzipped directory in case ifstartswith is also defined
    
        '''
        dirpath = directory[:-4]  # cut away the  .zip ending
        if ifstartswith is not None and ending is not None:
            if dirpath.startswith(ifstartswith):
                dirpath = dirpath + ending
        log.info("Unzipping {}".format(directory))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if os.path.exists(dirpath):
            if os.path.exists(directory):
                shutil.unpack_archive(
                    filename=directory, extract_dir=dirpath, format="zip"
                )
                os.remove(directory)
        else:
            log.error("Unzipping failed")
        return

    # read the ini file contents into a dictionary
    configparser.ConfigParser(allow_no_value=True)
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)

    config_dict, log = acd_initialisation(config_path)
    acd_config_to_log(config_dict, log)

    try:
        os.chdir(config_dict["pyeo_dir"]) # ensures pyeo is looking in the correct directory
        #start_date = config_dict["start_date"]
        end_date = config_dict["end_date"]
        if end_date == "TODAY":
            end_date = datetime.date.today().strftime("%Y%m%d")
        composite_start_date = config_dict["composite_start"]
        composite_end_date = config_dict["composite_end"]
        cloud_cover = config_dict["cloud_cover"]
        #cloud_certainty_threshold = config_dict["cloud_certainty_threshold"]
        #model_path = config_dict["model_path"]
        tile_dir = config_dict["tile_dir"]
        sen2cor_path = config_dict["sen2cor_path"]
        epsg = config_dict["epsg"]
        bands = config_dict["bands"]
        #resolution = config_dict["resolution_string"]
        out_resolution = config_dict["output_resolution"]
        #buffer_size = config_dict["buffer_size_cloud_masking"]
        buffer_size_composite = config_dict["buffer_size_cloud_masking_composite"]
        max_image_number = config_dict["download_limit"]
        faulty_granule_threshold = config_dict["faulty_granule_threshold"]
        #download_limit = config_dict["download_limit"]
        skip_existing = config_dict["do_skip_existing"]
        #sieve = config_dict["sieve"]
        #from_classes = config_dict["from_classes"]
        #to_classes = config_dict["to_classes"]
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
            composite_l1_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L1C")
            composite_l2_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"L2A")
            composite_l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"composite", r"cloud_masked")
            #change_image_dir = os.path.join(individual_tile_directory_path, r"images")
            #l1_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L1C")
            #l2_image_dir = os.path.join(individual_tile_directory_path, r"images", r"L2A")
            #l2_masked_image_dir = os.path.join(individual_tile_directory_path, r"images", r"cloud_masked")
            #categorised_image_dir = os.path.join(individual_tile_directory_path, r"output", r"classified")
            #probability_image_dir = os.path.join(individual_tile_directory_path, r"output", r"probabilities")
            quicklook_dir = os.path.join(individual_tile_directory_path, r"output", r"quicklooks")
            log.info("Successfully created the subdirectory paths for this tile")
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
        tile_log.info("Folder structure build successfully finished for tile "+tile_to_process)
        tile_log.info("  Directory path created: "+individual_tile_directory_path)
        tile_log.info("\n")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("---   PROCESSING START: {}   ---".format(tile_dir))
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Making an image composite as a baseline map for the change detection.")
        tile_log.info("List of image bands: {}".format(bands))
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Creating an initial cloud-free median composite from Sentinel-2 as a baseline map"
        )
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Searching for images for initial composite.")

        if download_source == "dataspace":

            # convert date string to YYYY-MM-DD
            date_object = datetime.datetime.strptime(composite_start_date, "%Y%m%d")
            dataspace_composite_start = date_object.strftime("%Y-%m-%d")
            date_object = datetime.datetime.strptime(composite_end_date, "%Y%m%d")
            dataspace_composite_end = date_object.strftime("%Y-%m-%d")

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
                geometry = geometry.representative_point()

                # attempt a geometry based query
                try:
                    dataspace_composite_products_all = queries_and_downloads.query_dataspace_by_polygon(
                        max_cloud_cover=cloud_cover,
                        start_date=dataspace_composite_start,
                        end_date=dataspace_composite_end,
                        area_of_interest=geometry,
                        max_records=100,
                        log=tile_log
                    )
                except Exception as error:
                    tile_log.error("Query_dataspace_by_polygon received this error: {}".format(error))
            else:
                # attempt a tile ID based query
                try:
                    dataspace_composite_products_all = queries_and_downloads.query_dataspace_by_tile_id(
                        max_cloud_cover=cloud_cover,
                        start_date=dataspace_composite_start,
                        end_date=dataspace_composite_end,
                        tile_id=tile_id,
                        max_records=100,
                        log=tile_log
                    )
                except Exception as error:
                    tile_log.error("Query_dataspace_by_tile received this error: {}".format(error))

            titles = dataspace_composite_products_all["title"].tolist()
            sizes = list()
            uuids = list()
            for elem in dataspace_composite_products_all.itertuples(index=False):
                sizes.append(elem[-2]["download"]["size"])
                uuids.append(elem[-2]["download"]["url"].split("/")[-1])

            relative_orbit_numbers = dataspace_composite_products_all["relativeOrbitNumber"].tolist()
            processing_levels = dataspace_composite_products_all["processingLevel"].tolist()
            transformed_levels = ['Level-1C' if level == 'S2MSI1C' else 'Level-2A' for level in processing_levels]
            cloud_covers = dataspace_composite_products_all["cloudCover"].tolist()
            begin_positions = dataspace_composite_products_all["startDate"].tolist()
            statuses = dataspace_composite_products_all["status"].tolist()

            scihub_compatible_df = pd.DataFrame({"title": titles,
                                                "size": sizes,
                                                "beginposition": begin_positions,
                                                "relativeorbitnumber": relative_orbit_numbers,
                                                "cloudcoverpercentage": cloud_covers,
                                                "processinglevel": transformed_levels,
                                                "uuid": uuids,
                                                "status": statuses})

            # check granule sizes on the server
            scihub_compatible_df["size"] = scihub_compatible_df["size"].apply(lambda x: round(float(x) * 1e-6, 2))
            # reassign to match the scihub variable
            df_all = scihub_compatible_df


        if download_source == "scihub":

            try:
                composite_products_all = queries_and_downloads.check_for_s2_data_by_date(
                    config_dict["tile_dir"],
                    composite_start_date,
                    composite_end_date,
                    conf=credentials_dict,
                    cloud_cover=cloud_cover,
                    tile_id=tile_to_process,
                    producttype=None,
                )

            except Exception as error:
                tile_log.error("check_for_s2_data_by_date failed:  {}".format(error))

            tile_log.info(
                "--> Found {} L1C and L2A products for the composite:".format(
                    len(composite_products_all)
                )
            )

            df_all = pd.DataFrame.from_dict(composite_products_all, orient="index")

            # check granule sizes on the server
            df_all["size"] = (
                df_all["size"]
                .str.split(" ")
                .apply(lambda x: float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]])
            )

        if download_source == "scihub":
            min_granule_size = faulty_granule_threshold
        else:
            min_granule_size = 0  # Required for dataspace API which doesn't report size correctly (often reported as zero)

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
                tile_log.info("dataspace branch reaches here")
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
        tile_log.info(" {} L1C products for the Composite".format(
            len(l1c_products['title']))
            )
        tile_log.info(" {} L2A products for the Composite".format(
            len(l2a_products['title']))
            )
        tile_log.info("Successfully queried the L1C and L2A products for "+
                      "the Composite")

        # Search the local directories, composite/L2A and L1C, checking if 
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
                            os.path.join(composite_l1_image_dir, f)
                            for f in os.listdir(composite_l1_image_dir)
                        ]
                        + [
                            os.path.join(composite_l2_image_dir, f)
                            for f in os.listdir(composite_l2_image_dir)
                        ]
                        + [
                            os.path.join(composite_l2_masked_image_dir, f)
                            for f in os.listdir(composite_l2_masked_image_dir)
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
                        start_date=composite_start_date,
                        end_date=composite_end_date,
                        filename=search_term,
                        cloud=cloud_cover,
                        producttype="S2MSI2A",
                    )

                    matching_l2a_products_df = pd.DataFrame.from_dict(
                        matching_l2a_products, orient="index"
                    )
                    # 07/03/2023: Matt - Applied Ali's fix for converting product size to MB to compare against faulty_grandule_threshold
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

                tile_log.info("\n Successfully searched for the L2A counterparts for the L1C products for the Composite")
                
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
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    source="scihub",
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )

            if download_source == "dataspace":

                queries_and_downloads.download_s2_data_from_dataspace(
                    product_df=l1c_products,
                    l1c_directory=composite_l1_image_dir,
                    l2a_directory=composite_l2_image_dir,
                    dataspace_username=sen_user,
                    dataspace_password=sen_pass,
                    log=tile_log
                )

            tile_log.info("Successfully downloaded the Sentinel-2 L1C products")

            tile_log.info("Atmospheric correction of L1C image products with sen2cor.")
            raster_manipulation.atmospheric_correction(
                composite_l1_image_dir,
                composite_l2_image_dir,
                sen2cor_path,
                delete_unprocessed_image=False,
                log=tile_log
            )

        tile_log.info("Successful atmospheric correction of the Sentinel-2 L1C products to L2A.")

        # Download the L2A images
        if l2a_products.shape[0] > 0:
            tile_log.info("Downloading Sentinel-2 L2A products.")

            if download_source == "scihub":

                queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    source="scihub",
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
            if download_source == "dataspace":

                queries_and_downloads.download_s2_data_from_dataspace(
                    product_df=l2a_products,
                    l1c_directory=composite_l1_image_dir,
                    l2a_directory=composite_l2_image_dir,
                    dataspace_username=sen_user,
                    dataspace_password=sen_pass,
                    log=tile_log
                )

        # check for incomplete L2A downloads
        incomplete_downloads, sizes = raster_manipulation.find_small_safe_dirs(
            composite_l2_image_dir, threshold=faulty_granule_threshold * 1024 * 1024
        )
        if len(incomplete_downloads) > 0:
            for index, safe_dir in enumerate(incomplete_downloads):
                if sizes[
                    index
                ] / 1024 / 1024 < faulty_granule_threshold and os.path.exists(safe_dir):
                    tile_log.warning("Found likely incomplete download of size {} MB: {}".format(
                            str(round(sizes[index] / 1024 / 1024)), safe_dir))

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Image download and atmospheric correction for composite is complete.")
        tile_log.info("---------------------------------------------------------------")

        # Housekeeping
        if config_dict["do_delete"]:
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Deleting downloaded L1C images for composite, keeping only derived L2A products")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            directory = composite_l1_image_dir
            tile_log.info("Deleting {}".format(directory))
            shutil.rmtree(directory)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Deletion of L1C images complete. Keeping only L2A images.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
        else:
            if config_dict["do_zip"]:
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Zipping downloaded L1C images for composite after atmospheric correction")
                tile_log.info("---------------------------------------------------------------")
                filesystem_utilities.zip_contents(composite_l1_image_dir)
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Zipping complete")
                tile_log.info("---------------------------------------------------------------")

        tile_log.info("Housekeeping after download successfully finished")

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files.")
        tile_log.info("---------------------------------------------------------------")

        directory = composite_l2_masked_image_dir
        masked_file_paths = [
            f
            for f in os.listdir(directory)
            if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
        ]

        directory = composite_l2_image_dir
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
                    filesystem_utilities.unzip_contents(
                        os.path.join(composite_l2_image_dir, f),
                        ifstartswith="S2",
                        ending=".SAFE",
                    )

        directory = composite_l2_image_dir
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
                composite_l2_image_dir,
                composite_l2_masked_image_dir,
                scl_classes=[0, 1, 2, 3, 8, 9, 10, 11],
                buffer_size=buffer_size_composite,
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
        tile_log.info("Offsetting cloud masked L2A images for composite.")
        tile_log.info("---------------------------------------------------------------")

        raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
            composite_l2_masked_image_dir, composite_l2_masked_image_dir)

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting of cloud masked L2A images for composite complete.")
        tile_log.info("---------------------------------------------------------------")


        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Producing quicklooks.")
            tile_log.info("---------------------------------------------------------------")
            dirs_for_quicklooks = [composite_l2_masked_image_dir]
            for main_dir in dirs_for_quicklooks:
                files = [
                    f.path
                    for f in os.scandir(main_dir)
                    if f.is_file() and os.path.basename(f).endswith(".tif")
                ]
                # files = [ f.path for f in os.scandir(main_dir) if f.is_file() and os.path.basename(f).endswith(".tif") and "class" in os.path.basename(f) ] # do classification images only
                if len(files) == 0:
                    tile_log.warning("No images found in {}.".format(main_dir))
                else:
                    for f in files:
                        quicklook_path = os.path.join(
                            quicklook_dir,
                            os.path.basename(f).split(".")[0] + ".png",
                        )
                        tile_log.info("Creating quicklook: {}".format(quicklook_path))
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
                "Zipping downloaded L2A images for composite after cloud masking and band stacking"
            )
            tile_log.info(
                "---------------------------------------------------------------"
            )
            filesystem_utilities.zip_contents(composite_l2_image_dir)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Zipping complete")
            tile_log.info(
                "---------------------------------------------------------------"
            )

        '''
        Create Composite from the Baseline Imagery
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            tile_log.info("---------------------------------------------------------------")
            tile_log.info(
                "Building initial cloud-free median composite from directory {}".format(
                    composite_l2_masked_image_dir
                )
            )
            tile_log.info("---------------------------------------------------------------")
            directory = composite_l2_masked_image_dir
            masked_file_paths = [
                f
                for f in os.listdir(directory)
                if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
            ]

            if len(masked_file_paths) > 0:
                raster_manipulation.clever_composite_directory(
                    composite_l2_masked_image_dir,
                    composite_dir,
                    chunks=config_dict["chunks"],
                    generate_date_images=True,
                    missing_data_value=0,
                )
                tile_log.info("---------------------------------------------------------------")
                tile_log.info("Baseline composite complete.")
                tile_log.info("---------------------------------------------------------------")

        '''
        Create Quicklook of the Composite
        '''
        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info("---------------------------------------------------------------")
            tile_log.info("Producing quicklook.")
            tile_log.info("---------------------------------------------------------------")
            dirs_for_quicklooks = [composite_dir]
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
                        tile_log.info(
                            "Creating quicklook: {}".format(quicklook_path)
                        )
                        raster_manipulation.create_quicklook(
                            f,
                            quicklook_path,
                            width=512,
                            height=512,
                            format="PNG",
                            bands=[3, 2, 1],
                            scale_factors=[[0, 2000, 0, 255]],
                        )
            tile_log.info("Quicklook complete.")

        '''
        Final Housekeeping
        Now that we have created our composite and produced any quicklooks, we tell pyeo 
            to delete or compress the cloud-masked L2A images that the composite was derived from.
        '''

        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            if config_dict["do_delete"]:
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info(
                    "Deleting intermediate cloud-masked L2A images used for the baseline composite"
                )
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                f = composite_l2_masked_image_dir
                tile_log.info("Deleting {}".format(f))
                shutil.rmtree(f)
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Intermediate file products have been deleted.")
                tile_log.info("They can be reprocessed from the downloaded L2A images.")
                tile_log.info(
                    "---------------------------------------------------------------"
                )
            else:
                if config_dict["do_zip"]:
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    tile_log.info(
                        "Zipping cloud-masked L2A images used for the baseline composite"
                    )
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    filesystem_utilities.zip_contents(composite_l2_masked_image_dir)
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )
                    tile_log.info("Zipping complete")
                    tile_log.info(
                        "---------------------------------------------------------------"
                    )

            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info(
                "Compressing tiff files in directory {} and all subdirectories".format(
                    composite_dir
                )
            )
            tile_log.info(
                "---------------------------------------------------------------"
            )
            for root, dirs, files in os.walk(composite_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
                for this_tiff in all_tiffs:
                    raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                    )

            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info(
                "Baseline image composite, file compression, zipping and deletion of"
            )
            tile_log.info("intermediate file products (if selected) are complete.")
            tile_log.info(
                "---------------------------------------------------------------"
            )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("---             TILE PROCESSING END                           ---")
        tile_log.info("---------------------------------------------------------------")

        # process the next tile if more than one tile are specified at this point (for loop)

    # after the for loop ends, log that all tiles are complete:
    log.info("---------------------------------------------------------------")
    log.info("---                  ALL PROCESSING END                      ---")
    log.info("---------------------------------------------------------------")
    return



if __name__ == "__main__":

    # Geopandas can throw an error with the proj installation if multiple subdirectories called proj are within the environment directory:
    #    PROJ: internal_proj_identify: /home/h/hb91/miniconda3/envs/pyeo_env/share/proj/proj.db lacks DATABASE.LAYOUT.VERSION.MAJOR / DATABASE.LAYOUT.VERSION.MINOR metadata. It comes from another PROJ installation.
    print(os.environ["PROJ_LIB"])

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

    detect_change(**vars(args))