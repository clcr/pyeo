import configparser
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
from pyeo import (acd_national, classification, filesystem_utilities,
                    queries_and_downloads, raster_manipulation)


def acd_by_tile_raster(config_path: str,
                       tile: str
                       ) -> None:
    """

    This function:

        - Downloads Images for the Composite
            - Converts any L1C images to L2A using Sen2Cor
            - Applies Offset Correction
            - SCL Cloud Mask
            - Temporal Median

        - Downloads Change Images
            - Converts any L1C images to L2A using Sen2Cor
            - Applies Offset Correction
            - SCL Cloud Mask

        - Classifies the Composite and Change Imagery

        - Runs Change Detection between the Composite and all Change images

        - Produces a Change Report as a Raster

    Parameters
    ----------
    config_path : str
        path to pyeo.ini
    tile : str
        Sentinel-2 tile name to process

    Returns
    ----------
    None

    """

    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)

    # changes directory to pyeo_dir, enabling the use of relative paths from the config file
    os.chdir(config_dict["pyeo_dir"])

    # wrap the whole process in a try block
    # try:
    # get path where the tiles are downloaded to
    tile_directory_path = config_dict["tile_dir"]

    # check for and create the folder structure pyeo expects
    individual_tile_directory_path = os.path.join(tile_directory_path, tile)
    if not os.path.exists(individual_tile_directory_path):
        # log.info(
        #     f"individual tile directory path  : {individual_tile_directory_path}"
        # )
        filesystem_utilities.create_folder_structure_for_tiles(
            individual_tile_directory_path
        )
    # else:
    # log.info(
    #     f"This individual tile directory already exists  : {individual_tile_directory_path}"
    # )

    # changes directory to pyeo_dir, enabling the use of relative paths from the config file
    os.chdir(config_dict["pyeo_dir"])

    # create per tile log file
    tile_log = filesystem_utilities.init_log_acd(
        log_path=os.path.join(individual_tile_directory_path, "log", tile + "_log.log"),
        logger_name=tile  #"pyeo"
    )

    # print config parameters to the tile log
    acd_national.acd_config_to_log(config_dict=config_dict, log=tile_log)

    # create per tile directory variables
    tile_log.info("Creating the directory paths")
    tile_root_dir = individual_tile_directory_path

    change_image_dir = os.path.join(tile_root_dir, "images")
    l1_image_dir = os.path.join(tile_root_dir, f"images{os.sep}L1C")
    l2_image_dir = os.path.join(tile_root_dir, f"images{os.sep}L2A")
    l2_masked_image_dir = os.path.join(tile_root_dir, f"images{os.sep}cloud_masked")
    categorised_image_dir = os.path.join(tile_root_dir, f"output{os.sep}classified")
    probability_image_dir = os.path.join(tile_root_dir, f"output{os.sep}probabilities")
    sieved_image_dir = os.path.join(tile_root_dir, f"output{os.sep}sieved")
    composite_dir = os.path.join(tile_root_dir, "composite")
    composite_l1_image_dir = os.path.join(tile_root_dir, f"composite{os.sep}L1C")
    composite_l2_image_dir = os.path.join(tile_root_dir, f"composite{os.sep}L2A")
    composite_l2_masked_image_dir = os.path.join(tile_root_dir, f"composite{os.sep}cloud_masked")
    quicklook_dir = os.path.join(tile_root_dir, f"output{os.sep}quicklooks")

    start_date = config_dict["start_date"]
    end_date = config_dict["end_date"]
    composite_start_date = config_dict["composite_start"]
    composite_end_date = config_dict["composite_end"]
    cloud_cover = config_dict["cloud_cover"]
    cloud_certainty_threshold = config_dict["cloud_certainty_threshold"]
    model_path = config_dict["model_path"]
    sen2cor_path = config_dict["sen2cor_path"]
    epsg = config_dict["epsg"]
    bands = config_dict["bands"]
    resolution = config_dict["resolution_string"]
    out_resolution = config_dict["output_resolution"]
    buffer_size = config_dict["buffer_size_cloud_masking"]
    buffer_size_composite = config_dict["buffer_size_cloud_masking_composite"]
    max_image_number = config_dict["download_limit"]
    faulty_granule_threshold = config_dict["faulty_granule_threshold"]
    download_limit = config_dict["download_limit"]

    skip_existing = config_dict["do_skip_existing"]
    sieve = config_dict["sieve"]
    from_classes = config_dict["from_classes"]
    to_classes = config_dict["to_classes"]

    download_source = config_dict["download_source"]
    if download_source == "scihub":
        tile_log.info("scihub API is the download source")
    if download_source == "dataspace":
        tile_log.info("dataspace API is the download source")

    tile_log.info(f"Faulty Granule Threshold is set to   : {config_dict['faulty_granule_threshold']}")
    tile_log.info("    Files below this threshold will not be downloaded")
        
    credentials_path = config_dict["credentials_path"]
    if not os.path.exists(credentials_path):
        tile_log.error(f"The credentials path does not exist  :{credentials_path}")
        tile_log.error(f"Current working directory :{os.getcwd()}")
        tile_log.error("Exiting raster pipeline")
        sys.exit(1)

    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(credentials_path)
    credentials_dict = {}
    # credentials_dict is made because functions further in the pipeline want a dictionary

    # ------------------------------------------------------------------------
    # Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map
    # ------------------------------------------------------------------------

    ## Setup credentials for data API
    if download_source == "dataspace":

        tile_log.info(f'Running download handler for {download_source}')

        credentials_dict["sent_2"] = {}
        credentials_dict["sent_2"]["user"] = conf["dataspace"]["user"]
        credentials_dict["sent_2"]["pass"] = conf["dataspace"]["pass"]
        sen_user = credentials_dict["sent_2"]["user"]
        sen_pass = credentials_dict["sent_2"]["pass"]

    if download_source == "scihub":

        tile_log.info(f'Running download handler for {download_source}')

        credentials_dict["sent_2"] = {}
        credentials_dict["sent_2"]["user"] = conf["sent_2"]["user"]
        credentials_dict["sent_2"]["pass"] = conf["sent_2"]["pass"]
        sen_user = credentials_dict["sent_2"]["user"]
        sen_pass = credentials_dict["sent_2"]["pass"]

    if config_dict["build_composite"] or config_dict["do_all"]:
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Creating an initial cloud-free median composite from Sentinel-2 as a baseline map")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Searching for images for initial composite.")

        if download_source == "dataspace":

            try:
                tiles_geom_path = os.path.join(config_dict["pyeo_dir"], os.path.join(config_dict["geometry_dir"], config_dict["s2_tiles_filename"]))
                tile_log.info(f"Absolute path to S2 tile geometry: {os.path.abspath(tiles_geom_path)}")
                tiles_geom = gpd.read_file(os.path.abspath(tiles_geom_path))
            except FileNotFoundError:
                # tile_log.error(f"Path to the S2 tile geometry does not exist, the path is :{tiles_geom_path}")
                tile_log.error(f"Path to S2 tile geometry does not exist, absolute path given: {os.path.abspath(tiles_geom_path)}")

            tile_geom = tiles_geom[tiles_geom["Name"] == tile]
            tile_geom = tile_geom.to_crs(epsg=4326)
            geometry = tile_geom["geometry"].iloc[0]
            geometry = geometry.representative_point()
            
            # convert date string to YYYY-MM-DD
            date_object = datetime.strptime(composite_start_date, "%Y%m%d")
            dataspace_composite_start = date_object.strftime("%Y-%m-%d")
            date_object = datetime.strptime(composite_end_date, "%Y%m%d")
            dataspace_composite_end = date_object.strftime("%Y-%m-%d")

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
                tile_log.error(f"query_dataspace_by_polygon received this error: {error}")
            
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
            # tile_log.info(f"this is the before size df amendment : {scihub_compatible_df['size']}")
            # sys.exit(1)
            scihub_compatible_df["size"] = scihub_compatible_df["size"].apply(lambda x: round(float(x) * 1e-6, 2))
            # reassign to match the scihub variable
            df_all = scihub_compatible_df


        if download_source == "scihub":

            try:
                composite_products_all = queries_and_downloads.check_for_s2_data_by_date(
                    tile_root_dir,
                    composite_start_date,
                    composite_end_date,
                    conf=credentials_dict,
                    cloud_cover=cloud_cover,
                    tile_id=tile,
                    producttype=None,  # "S2MSI2A" or "S2MSI1C"
                )

            except Exception as error:
                tile_log.error(
                    f"check_for_s2_data_by_date failed, got this error :  {error}"
                )
        
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
                tile_log.info(f"len of L1C products for dataspace is {len(l1c_products['title'])}")

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
                tile_log.info(f"len of L2A products for dataspace is {len(l2a_products['title'])}")

        if l1c_products.shape[0] > 0 and l2a_products.shape[0] > 0:
            tile_log.info(
                "Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product."
            )
            if download_source == "scihub":
                (l1c_products,l2a_products,) = queries_and_downloads.filter_unique_l1c_and_l2a_data(df,log=tile_log)

            if download_source == "dataspace":
                l1c_products = queries_and_downloads.filter_unique_dataspace_products(l1c_products=l1c_products, l2a_products=l2a_products, log=tile_log)
                                                                                                      
            # tile_log.info(
            #     "--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(
            #         l1c_products.shape[0] + l2a_products.shape[0]
            #     )
            # )
    
    
        df = None
        tile_log.info(f" {len(l1c_products['title'])} L1C products for the Composite")
        tile_log.info(f" {len(l2a_products['title'])} L2A products for the Composite")

        # Search the local directories, composite/L2A and L1C, checking if scenes have already been downloaded and/or processed whilst checking their dir sizes
        if download_source == "scihub":
            if l1c_products.shape[0] > 0:
                tile_log.info("Checking for already downloaded and zipped L1C or L2A products and")
                tile_log.info("  availability of matching L2A products for download.")
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

        ######### above is querying

        ######## below is downloading
        # if L1C products remain after matching for L2As, then download the unmatched L1Cs
        if l1c_products.shape[0] > 0:
            tile_log.info(f"Downloading Sentinel-2 L1C products from {download_source}:")

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
            
            tile_log.info("Atmospheric correction with sen2cor.")
            raster_manipulation.atmospheric_correction(
                composite_l1_image_dir,
                composite_l2_image_dir,
                sen2cor_path,
                delete_unprocessed_image=False,
                log=tile_log,
            )

        if l2a_products.shape[0] > 0:
            tile_log.info("Downloading Sentinel-2 L2A products.")

            if download_source == "scihub":

                queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    source="scihub",
                    # download_source,
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
                    tile_log.warning(
                        "Found likely incomplete download of size {} MB: {}".format(
                            str(round(sizes[index] / 1024 / 1024)), safe_dir
                        )
                    )
                    # shutil.rmtree(safe_dir)

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Image download and atmospheric correction for composite is complete."
        )
        tile_log.info("---------------------------------------------------------------")

        if config_dict["do_delete"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info(
                "Deleting downloaded L1C images for composite, keeping only derived L2A products"
            )
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
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info(
                    "Zipping downloaded L1C images for composite after atmospheric correction"
                )
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                filesystem_utilities.zip_contents(composite_l1_image_dir)
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Zipping complete")
                tile_log.info(
                    "---------------------------------------------------------------"
                )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files."
        )
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
            tile_log.info(
                "No L2A images found for cloud masking. They may already have been done."
            )
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
                skip_existing=skip_existing,
            )
        # I.R. 20220607 START
        # Apply offset to any images of processing baseline 0400 in the composite cloud_masked folder
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting cloud masked L2A images for composite.")
        tile_log.info("---------------------------------------------------------------")

        raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
            composite_l2_masked_image_dir, composite_l2_masked_image_dir
        )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting of cloud masked L2A images for composite complete.")
        tile_log.info("---------------------------------------------------------------")
        # I.R. 20220607 END

        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Producing quicklooks.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
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
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Baseline composite complete.")
            tile_log.info(
                "---------------------------------------------------------------"
            )

            if config_dict["do_quicklooks"] or config_dict["do_all"]:
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Producing quicklooks.")
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                dirs_for_quicklooks = [composite_dir]
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
                tile_log.info("Quicklooks complete.")

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

        else:
            tile_log.error(
                "No cloud-masked L2A image products found in {}.".format(
                    composite_l2_image_dir
                )
            )
            tile_log.error(
                "Cannot produce a median composite. Download and cloud-mask some images first."
            )

    # ------------------------------------------------------------------------
    # Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C)
    # ------------------------------------------------------------------------
    if config_dict["do_all"] or config_dict["do_download"]:
        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Downloading change detection images between {} and {} with cloud cover <= {}".format(
                start_date, end_date, cloud_cover
            )
        )
        tile_log.info("---------------------------------------------------------------")
        if download_source == "dataspace":

            try:
                tiles_geom_path = os.path.join(config_dict["pyeo_dir"], os.path.join(config_dict["geometry_dir"], config_dict["s2_tiles_filename"]))
                tiles_geom = gpd.read_file(os.path.abspath(tiles_geom_path))
                
            except FileNotFoundError:
                tile_log.error(f"tiles_geom does not exist, the path is :{tiles_geom_path}")

            tile_geom = tiles_geom[tiles_geom["Name"] == tile]
            tile_geom = tile_geom.to_crs(epsg=4326)
            geometry = tile_geom["geometry"].iloc[0]
            geometry = geometry.representative_point()
            
            # convert date string to YYYY-MM-DD
            date_object = datetime.strptime(start_date, "%Y%m%d")
            dataspace_change_start = date_object.strftime("%Y-%m-%d")
            date_object = datetime.strptime(end_date, "%Y%m%d")
            dataspace_change_end = date_object.strftime("%Y-%m-%d")

            try:
                dataspace_change_products_all = queries_and_downloads.query_dataspace_by_polygon(
                    max_cloud_cover=cloud_cover,
                    start_date=dataspace_change_start,
                    end_date=dataspace_change_end,
                    area_of_interest=geometry,
                    max_records=100,
                    log=tile_log
                )
            except Exception as error:
                tile_log.error(f"query_by_polygon received this error: {error}")
                
            titles = dataspace_change_products_all["title"].tolist()
            sizes = list()
            uuids = list()
            for elem in dataspace_change_products_all.itertuples(index=False):
                sizes.append(elem[-2]["download"]["size"])
                uuids.append(elem[-2]["download"]["url"].split("/")[-1])
            

            relative_orbit_numbers = dataspace_change_products_all["relativeOrbitNumber"].tolist()
            processing_levels = dataspace_change_products_all["processingLevel"].tolist()
            transformed_levels = ['Level-1C' if level == 'S2MSI1C' else 'Level-2A' for level in processing_levels]
            cloud_covers = dataspace_change_products_all["cloudCover"].tolist()
            begin_positions = dataspace_change_products_all["startDate"].tolist()
            statuses = dataspace_change_products_all["status"].tolist()

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
            products_all = queries_and_downloads.check_for_s2_data_by_date(
                tile_root_dir,
                start_date,
                end_date,
                credentials_dict,
                cloud_cover=cloud_cover,
                tile_id=tile,
                producttype=None,  # "S2MSI2A" or "S2MSI1C"
            )
            tile_log.info(
                "--> Found {} L1C and L2A products for change detection:".format(
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

        # here the main call (from if download_source == "scihub" branch) is resumed
        df = df_all.query("size >= " + str(faulty_granule_threshold))
        tile_log.info(
            "Removed {} faulty scenes <{}MB in size from the list:".format(
                len(df_all) - len(df), faulty_granule_threshold
            )
        )
        df_faulty = df_all.query("size < " + str(faulty_granule_threshold))
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
        
        if l1c_products.shape[0] > 0 and l2a_products.shape[0] > 0:
            tile_log.info(
                "Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product."
            )
            if download_source == "scihub":
                (l1c_products,l2a_products,) = queries_and_downloads.filter_unique_l1c_and_l2a_data(df,log=tile_log)

            if download_source == "dataspace":
                l1c_products = queries_and_downloads.filter_unique_dataspace_products(l1c_products=l1c_products, l2a_products=l2a_products, log=tile_log)
                                                                                                      
            tile_log.info(
                "--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(
                    l1c_products.shape[0] + l2a_products.shape[0]
                )
            )
        
        df = None
        tile_log.info(f" {len(l1c_products['title'])} L1C Change Images")
        tile_log.info(f" {len(l2a_products['title'])} L2A Change Images")

        # TODO: Before the next step, search the composite/L2A and L1C directories whether the scenes have already been downloaded and/or processed and check their dir sizes
        # Remove those already obtained from the list
        if download_source == "scihub":    
            if l1c_products.shape[0] > 0:
                tile_log.info(
                    "Checking for availability of L2A products to minimise download and atmospheric correction of L1C products."
                )
                n = len(l1c_products)
                drop = []
                add = []
                for r in range(n):
                    id = l1c_products.iloc[r, :]["title"]
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
                    tile_log.info("Search term: {}.".format(search_term))
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
                    if len(matching_l2a_products_df) == 1:
                        tile_log.info(matching_l2a_products_df.iloc[0, :]["size"])
                        matching_l2a_products_df["size"] = (
                            matching_l2a_products_df["size"]
                            .str.split(" ")
                            .apply(
                                lambda x: float(x[0])
                                * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            )
                        )
                        if (
                            matching_l2a_products_df.iloc[0, :]["size"]
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
                        tile_log.info("Found no match for L1C: {}.".format(id))
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
                        if (
                            matching_l2a_products_df.iloc[0, :]["size"]
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
                    if config_dict["do_dev"]:
                        add = pd.DataFrame(add)
                        l2a_products = pd.concat([l2a_products, add])
                        # TODO: test the above fix for:
                        # pyeo/pyeo/apps/change_detection/tile_based_change_detection_from_cover_maps.py:456: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
                    else:
                        add = pd.DataFrame(add)
                        l2a_products = pd.concat([l2a_products, add])

                tile_log.info(
                    "    {} L1C products remaining for download".format(
                        l1c_products.shape[0]
                    )
                )
        l2a_products = l2a_products.drop_duplicates(subset="title")
        # I.R.
        tile_log.info(
            "    {} L2A products remaining for download".format(
                l2a_products.shape[0]
            )
        )
        if l1c_products.shape[0] > 0:

            tile_log.info(f"Downloading Sentinel-2 L1C products from {download_source}")
            
            if download_source == "scihub":
                queries_and_downloads.download_s2_data_from_df(
                    l1c_products,
                    l1_image_dir,
                    l2_image_dir,
                    download_source,
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
            elif download_source == "dataspace":
                    queries_and_downloads.download_s2_data_from_dataspace(
                    product_df=l1c_products,
                    l1c_directory=l1_image_dir,
                    l2a_directory=l2_image_dir,
                    dataspace_username=sen_user,
                    dataspace_password=sen_pass,
                    log=tile_log
                )
            else:
                tile_log.error(f"download source specified did not match 'scihub' or 'dataspace'")
                tile_log.error(f"download source supplied was  :  {download_source}")
                tile_log.error("exiting pipeline...")
                sys.exit(1)

            tile_log.info("Atmospheric correction with sen2cor.")
            raster_manipulation.atmospheric_correction(
                l1_image_dir,
                l2_image_dir,
                sen2cor_path,
                delete_unprocessed_image=False,
                log=tile_log,
            )
        if l2a_products.shape[0] > 0:
            tile_log.info(f"Downloading Sentinel-2 L2A products from {download_source}")

            if download_source == "scihub":
                queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    l1_image_dir,
                    l2_image_dir,
                    download_source,
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

        # check for incomplete L2A downloads and remove them
        incomplete_downloads, sizes = raster_manipulation.find_small_safe_dirs(
            l2_image_dir, threshold=faulty_granule_threshold * 1024 * 1024
        )
        if len(incomplete_downloads) > 0:
            for index, safe_dir in enumerate(incomplete_downloads):
                if sizes[
                    index
                ] / 1024 / 1024 < faulty_granule_threshold and os.path.exists(safe_dir):
                    tile_log.warning(
                        "Found likely incomplete download of size {} MB: {}".format(
                            str(round(sizes[index] / 1024 / 1024)), safe_dir
                        )
                    )
                    # shutil.rmtree(safe_dir)

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Image download and atmospheric correction for change detection images is complete."
        )
        tile_log.info("---------------------------------------------------------------")
        # TODO: delete L1C images if do_delete is True
        if config_dict["do_delete"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Deleting L1C images downloaded for change detection.")
            tile_log.info(
                "Keeping only the derived L2A images after atmospheric correction."
            )
            tile_log.info(
                "---------------------------------------------------------------"
            )
            directory = l1_image_dir
            tile_log.info("Deleting {}".format(directory))
            shutil.rmtree(directory)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Deletion complete")
            tile_log.info(
                "---------------------------------------------------------------"
            )
        else:
            if config_dict["do_zip"]:
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Zipping L1C images downloaded for change detection")
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                filesystem_utilities.zip_contents(l1_image_dir)
                tile_log.info(
                    "---------------------------------------------------------------"
                )
                tile_log.info("Zipping complete")
                tile_log.info(
                    "---------------------------------------------------------------"
                )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files."
        )
        tile_log.info("---------------------------------------------------------------")
        # l2a_paths = [ f.path for f in os.scandir(l2_image_dir) if f.is_dir() ]
        # tile_log.info("  l2_image_dir: {}".format(l2_image_dir))
        # tile_log.info("  l2_masked_image_dir: {}".format(l2_masked_image_dir))
        # tile_log.info("  bands: {}".format(bands))
        raster_manipulation.apply_scl_cloud_mask(
            l2_image_dir,
            l2_masked_image_dir,
            scl_classes=[0, 1, 2, 3, 8, 9, 10, 11],
            buffer_size=buffer_size,
            bands=bands,
            out_resolution=out_resolution,
            haze=None,
            epsg=epsg,
            skip_existing=skip_existing,
        )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Cloud masking and band stacking of new L2A images are complete.")
        tile_log.info("---------------------------------------------------------------")

        # I.R. 20220607 START
        # Apply offset to any images of processing baseline 0400 in the composite cloud_masked folder
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting cloud masked L2A images.")
        tile_log.info("---------------------------------------------------------------")

        raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
            l2_masked_image_dir, l2_masked_image_dir
        )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Offsetting of cloud masked L2A images complete.")
        tile_log.info("---------------------------------------------------------------")
        # I.R. 20220607 END

        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Producing quicklooks.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            dirs_for_quicklooks = [l2_masked_image_dir]
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

        if config_dict["do_zip"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Zipping L2A images downloaded for change detection")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            filesystem_utilities.zip_contents(l2_image_dir)
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Zipping complete")
            tile_log.info(
                "---------------------------------------------------------------"
            )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Compressing tiff files in directory {} and all subdirectories".format(
                l2_masked_image_dir
            )
        )
        tile_log.info("---------------------------------------------------------------")
        for root, dirs, files in os.walk(l2_masked_image_dir):
            all_tiffs = [
                image_name for image_name in files if image_name.endswith(".tif")
            ]
            for this_tiff in all_tiffs:
                raster_manipulation.compress_tiff(
                    os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Pre-processing of change detection images, file compression, zipping"
        )
        tile_log.info(
            "and deletion of intermediate file products (if selected) are complete."
        )
        tile_log.info("---------------------------------------------------------------")

    # ------------------------------------------------------------------------
    # Step 3: Classify each L2A image and the baseline composite
    # ------------------------------------------------------------------------
    if config_dict["do_all"] or config_dict["do_classify"]:
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Classifying composite & change images using Random Forest Model")
        tile_log.info("Model Provided: {}".format(model_path))
        tile_log.info("---------------------------------------------------------------")

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
        )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Compressing tiff files in directory {} and all subdirectories".format(categorised_image_dir))
        tile_log.info("---------------------------------------------------------------")
        
        for root, dirs, files in os.walk(categorised_image_dir):
            all_tiffs = [
                image_name for image_name in files if image_name.endswith(".tif")
            ]
            for this_tiff in all_tiffs:
                raster_manipulation.compress_tiff(
                    os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Classification of all images is complete.")
        tile_log.info("---------------------------------------------------------------")

        if config_dict["do_quicklooks"] or config_dict["do_all"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info("Producing quicklooks.")
            tile_log.info(
                "---------------------------------------------------------------"
            )
            dirs_for_quicklooks = [categorised_image_dir]
            for main_dir in dirs_for_quicklooks:
                # files = [ f.path for f in os.scandir(main_dir) if f.is_file() and os.path.basename(f).endswith(".tif") ]
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
                        tile_log.info("Creating quicklook: {}".format(quicklook_path))
                        raster_manipulation.create_quicklook(
                            f, quicklook_path, width=512, height=512, format="PNG"
                        )
        tile_log.info("Quicklooks complete.")

    # ------------------------------------------------------------------------
    # Step 4: Pair up the class images with the composite baseline map
    # and identify all pixels with the change between groups of classes of interest.
    # Optionally applies a sieve filter to the class images if specified in the ini file.
    # Confirms detected changes by NDVI differencing.
    # ------------------------------------------------------------------------

    if config_dict["do_all"] or config_dict["do_change"]:
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Creating change layers from stacked class images.")
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Changes of interest:")
        tile_log.info(
            "  from any of the classes {}".format(config_dict["from_classes"])
        )
        tile_log.info("  to   any of the classes {}".format(config_dict["to_classes"]))

        # optionally sieve the class images
        if sieve > 0:
            tile_log.info("Applying sieve to classification outputs.")
            sieved_paths = raster_manipulation.sieve_directory(
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
            raise FileNotFoundError(
                "No class images found in {}.".format(class_image_dir)
            )

        # sort class images by image acquisition date
        class_image_paths = list(
            filter(filesystem_utilities.get_image_acquisition_time, class_image_paths)
        )
        class_image_paths.sort(
            key=lambda x: filesystem_utilities.get_image_acquisition_time(x)
        )
        for index, image in enumerate(class_image_paths):
            tile_log.info("{}: {}".format(index, image))

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
            tile_log.info("Most recent composite at {}".format(latest_composite_path))
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

        tile_log.info(
            "Most recent class composite at {}".format(latest_class_composite_path)
        )
        if not os.path.exists(latest_class_composite_path):
            tile_log.critical(
                "Latest class composite not found. The first time you run this script, you need to include the "
                "--build-composite flag to create a base composite to work off. If you have already done this,"
                "check that the earliest dated image in your images/merged folder is later than the earliest"
                " dated image in your composite/ folder. Then, you need to run the --classify option."
            )
            sys.exit(1)

        if config_dict[
            "do_dev"
        ]:  # set the name of the report file in the development version run
            before_timestamp = filesystem_utilities.get_change_detection_dates(
                os.path.basename(latest_class_composite_path)
            )[0]
            # I.R. 20220611 START
            ## Timestamp report with the date of most recent classified image that contributes to it
            after_timestamp = filesystem_utilities.get_image_acquisition_time(
                os.path.basename(class_image_paths[-1])
            )
            ## ORIGINAL
            # gets timestamp of the earliest change image of those available in class_image_path
            # after_timestamp  = pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(class_image_paths[0]))
            # I.R. 20220611 END
            output_product = os.path.join(
                probability_image_dir,
                "report_{}_{}_{}.tif".format(
                    before_timestamp.strftime("%Y%m%dT%H%M%S"),
                    tile,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
            tile_log.info("I.R. Report file name will be {}".format(output_product))

            # if a report file exists, archive it  ( I.R. Changed from 'rename it to show it has been updated')
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
                # I.R. ToDo: Should iterate over output_product_existing in case more than one report file is present (though unlikely)
                output_product_existing = [
                    f.path
                    for f in os.scandir(probability_image_dir)
                    if f.is_file()
                    and f.name.startswith("report_")
                    and f.name.endswith(".tif")
                ][0]
                tile_log.info(
                    "Found existing report image product: {}".format(
                        output_product_existing
                    )
                )
                # I.R. 20220610 START
                ## Mark existing reports as 'archive_'
                ## - do not try and extend upon existing reports
                ## - calls to __change_from_class_maps below will build a report incorporating all new AND pre-existing change maps
                ## - this might be the cause of the error in report generation that caused over-range and periodicity in the histogram - as reported to Heiko by email
                # report_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(output_product_existing))[1]
                # if report_timestamp < after_timestamp:
                # tile_log.info("Report timestamp {}".format(report_timestamp.strftime("%Y%m%dT%H%M%S")))
                # tile_log.info(" is earlier than {}".format(after_timestamp.strftime("%Y%m%dT%H%M%S")))
                # tile_log.info("Updating its file name to: {}".format(output_product))
                # os.rename(output_product_existing, output_product)

                # Renaming any pre-existing report file with prefix 'archive_'
                ## it will therefore not be detected in __change_from_class_maps which will therefore create a new report file

                output_product_existing_archived = os.path.join(
                    os.path.dirname(output_product_existing),
                    "archived_" + os.path.basename(output_product_existing),
                )
                tile_log.info(
                    "Renaming existing report image product to: {}".format(
                        output_product_existing_archived
                    )
                )
                os.rename(output_product_existing, output_product_existing_archived)

                # I.R. 20220610 END

        # find change patterns in the stack of classification images

        for index, image in enumerate(class_image_paths):
            tile_log.info("")
            tile_log.info("")
            tile_log.info(f"  printing index, image   : {index}, {image}")
            tile_log.info("")
            tile_log.info("")
            before_timestamp = filesystem_utilities.get_change_detection_dates(
                os.path.basename(latest_class_composite_path)
            )[0]
            after_timestamp = filesystem_utilities.get_image_acquisition_time(
                os.path.basename(image)
            )
            # I.R. 20220612 START
            tile_log.info(
                "*** PROCESSING CLASSIFIED IMAGE: {} of {} filename: {} ***".format(
                    index, len(class_image_paths), image
                )
            )
            # I.R. 20220612 END
            tile_log.info("  early time stamp: {}".format(before_timestamp))
            tile_log.info("  late  time stamp: {}".format(after_timestamp))
            change_raster = os.path.join(
                probability_image_dir,
                "change_{}_{}_{}.tif".format(
                    before_timestamp.strftime("%Y%m%dT%H%M%S"),
                    tile,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
            tile_log.info(
                "  Change raster file to be created: {}".format(change_raster)
            )

            dNDVI_raster = os.path.join(
                probability_image_dir,
                "dNDVI_{}_{}_{}.tif".format(
                    before_timestamp.strftime("%Y%m%dT%H%M%S"),
                    tile,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
            tile_log.info(
                "  I.R. dNDVI raster file to be created: {}".format(dNDVI_raster)
            )

            NDVI_raster = os.path.join(
                probability_image_dir,
                "NDVI_{}_{}_{}.tif".format(
                    before_timestamp.strftime("%Y%m%dT%H%M%S"),
                    tile,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
            tile_log.info(
                "  I.R. NDVI raster file of change image to be created: {}".format(
                    NDVI_raster
                )
            )

            if config_dict["do_dev"]:
                # This function looks for changes from class 'change_from' in the composite to any of the 'change_to_classes'
                # in the change images. Pixel values are the acquisition date of the detected change of interest or zero.
                # TODO: In change_from_class_maps(), add a flag (e.g. -1) whether a pixel was a cloud in the later image.
                # Applying check whether dNDVI < -0.2, i.e. greenness has decreased over changed areas

                tile_log.info(
                    "Update of the report image product based on change detection image."
                )
                raster_manipulation.__change_from_class_maps(
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
            else:
                raster_manipulation.change_from_class_maps(
                    latest_class_composite_path,
                    image,
                    change_raster,
                    change_from=from_classes,
                    change_to=to_classes,
                    skip_existing=skip_existing,
                )

        # I.R. ToDo: Function compute additional layers derived from set of layers in report file generated in __change_from_class_maps()
        # pyeo.raster_manipulation.computed_report_layer_generation(report_path = output_product)

        # I.R. ToDo: Function to generate 3D time series array of classified (+NDVI?) images over full date range
        ## and save it to disk as a layered GeoTIFF (or numpy array)
        ## (Build into above loop that generates report...?)
        # pyeo.raster_manipulation.time_series_construction(classified_image_dir = classified_image_dir, change_from = from_classes,change_to = to_classes)

        # I.R. ToDo: Insert function to perform time series analysis on 3D classified (+NDVI?) time series array and generate forest alert outputs
        ## in a GeoTIFF file
        # pyeo.raster_manipulation.time_series_analysis(report_path = output_product)
        #
        # I.R. ToDo: Alternatively.. implement sliding buffer to scan through classified (and/or NDVI) image set so that FIR, IIR and State-Machine
        ## filters can be implemented to generate forest alerts
        ## e.g. 5 layers to hold rotating buffer of classification and/or NDVI state plus additional layers to hold state variables
        ## Use state to record a run of n consecutive change_from classes,
        ## detect transition to bare earth class with a simultaneous NDVI drop of > threshold
        ## record time point as first change date
        ## detect subsequent change to grassland class as bare earth re-greens with new growth
        ## count detection of multiple such cycles if they occur
        ## Use the above as input for temporal classification of pixel by land use e.g. multiple season correlated cycles as a signature of cropland
        ### and thus establish an expectation of variation for that pixel in the future
        ## Extend use of above to incorporate spatial analysis over multiple pixel neighbourhoods

        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Post-classification change detection complete.")
        tile_log.info("---------------------------------------------------------------")

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Compressing tiff files in directory {} and all subdirectories".format(
                probability_image_dir
            )
        )
        tile_log.info("---------------------------------------------------------------")
        for root, dirs, files in os.walk(probability_image_dir):
            all_tiffs = [
                image_name for image_name in files if image_name.endswith(".tif")
            ]
            for this_tiff in all_tiffs:
                raster_manipulation.compress_tiff(
                    os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                )

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Compressing tiff files in directory {} and all subdirectories".format(
                sieved_image_dir
            )
        )
        tile_log.info("---------------------------------------------------------------")
        for root, dirs, files in os.walk(sieved_image_dir):
            all_tiffs = [
                image_name for image_name in files if image_name.endswith(".tif")
            ]
            for this_tiff in all_tiffs:
                raster_manipulation.compress_tiff(
                    os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                )

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
            # combine all change layers into one output raster with two layers:
            #   (1) pixels show the earliest change detection date (expressed as the number of days since 1/1/2000)
            #   (2) pixels show the number of change detection dates (summed up over all change images in the folder)
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
                    tile,
                    # tile_id,
                    after_timestamp.strftime("%Y%m%dT%H%M%S"),
                ),
            )
            tile_log.info("Combining date maps: {}".format(date_image_paths))
            raster_manipulation.combine_date_maps(date_image_paths, output_product)

        tile_log.info("---------------------------------------------------------------")
        tile_log.info(
            "Report image product completed / updated: {}".format(output_product)
        )
        tile_log.info("Compressing the report image.")
        tile_log.info("---------------------------------------------------------------")
        raster_manipulation.compress_tiff(output_product, output_product)

        if config_dict["do_delete"]:
            tile_log.info(
                "---------------------------------------------------------------"
            )
            tile_log.info(
                "Deleting intermediate class images used in change detection."
            )
            tile_log.info(
                "They can be recreated from the cloud-masked, band-stacked L2A images and the saved model."
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
                    # keep the classified composite layers and the report image product for the next change detection
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
                    filesystem_utilities.zip_contents(
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
            "Change detection and report image product updating, file compression, zipping"
        )
        tile_log.info(
            "and deletion of intermediate file products (if selected) are complete."
        )
        tile_log.info("---------------------------------------------------------------")

    if config_dict["do_delete"]:
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Deleting temporary directories starting with 'tmp*'")
        tile_log.info("These can be left over from interrupted processing runs.")
        tile_log.info("---------------------------------------------------------------")
        directory = tile_root_dir
        for root, dirs, files in os.walk(directory):
            temp_dirs = [d for d in dirs if d.startswith("tmp")]
            for temp_dir in temp_dirs:
                tile_log.info("Deleting {}".format(os.path.join(root, temp_dir)))
                if os.path.isdir(os.path.join(directory, f)):
                    shutil.rmtree(os.path.join(directory, f))
                else:
                    tile_log.warning(
                        "This should not have happened. {} is not a directory. Skipping deletion.".format(
                            os.path.join(root, temp_dir)
                        )
                    )
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Deletion of temporary directories complete.")
        tile_log.info("---------------------------------------------------------------")

    # ------------------------------------------------------------------------
    # Step 5: Update the baseline composite with the reflectance values of only the changed pixels.
    #         Update last_date of the baseline composite.
    # ------------------------------------------------------------------------

    if config_dict["do_update"] or config_dict["do_all"]:
        tile_log.warning(
            "---------------------------------------------------------------"
        )
        tile_log.warning(
            "Updating of the baseline composite with new imagery is deprecated and will be ignored."
        )
        tile_log.warning(
            "---------------------------------------------------------------"
        )
        # Matt 11/05/23: the below code is kept for historical reasons, i.e. if a programmer wants to develop
        # the update baseline composite method, they can follow the code below to see the thought process.
        """
        tile_log.info("---------------------------------------------------------------")
        tile_log.info("Updating baseline composite with new imagery.")
        tile_log.info("---------------------------------------------------------------")
        # get all composite file paths
        composite_paths = [ f.path for f in os.scandir(composite_dir) if f.is_file() ]
        if len(composite_paths) == 0:
            raise FileNotFoundError("No composite images found in {}.".format(composite_dir))
        tile_log.info("Sorting composite image list by time stamp.")
        composite_images = \
            pyeo.filesystem_utilities.sort_by_timestamp(
                [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
                recent_first=False
            )
        try:
            latest_composite_name = \
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
                    recent_first=True
                )[0]
            latest_composite_path = os.path.join(composite_dir, latest_composite_name)
            latest_composite_timestamp = pyeo.filesystem_utilities.get_sen_2_image_timestamp(os.path.basename(latest_composite_path))
            tile_log.info("Most recent composite at {}".format(latest_composite_path))
        except IndexError:
            tile_log.critical("Latest composite not found. The first time you run this script, you need to include the "
                            "--build-composite flag to create a base composite to work off. If you have already done this,"
                            "check that the earliest dated image in your images/merged folder is later than the earliest"
                            " dated image in your composite/ folder.")
            sys.exit(1)

        # Find all categorised images
        categorised_paths = [ f.path for f in os.scandir(categorised_image_dir) if f.is_file() ]
        if len(categorised_paths) == 0:
            raise FileNotFoundError("No categorised images found in {}.".format(categorised_image_dir))
        tile_log.info("Sorting categorised image list by time stamp.")
        categorised_images = \
            pyeo.filesystem_utilities.sort_by_timestamp(
                [image_name for image_name in os.listdir(categorised_image_dir) if image_name.endswith(".tif")],
                recent_first=False
            )
        # Drop the categorised images that were made before the most recent composite date
        latest_composite_timestamp_datetime = pyeo.filesystem_utilities.get_image_acquisition_time(latest_composite_name)
        categorised_images = [image for image in categorised_images \
                                if pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(image))[1] > latest_composite_timestamp_datetime ]

        # Find all L2A images
        l2a_paths = [ f.path for f in os.scandir(l2_masked_image_dir) if f.is_file() ]
        if len(l2a_paths) == 0:
            raise FileNotFoundError("No images found in {}.".format(l2_masked_image_dir))
        tile_log.info("Sorting masked L2A image list by time stamp.")
        l2a_images = \
            pyeo.filesystem_utilities.sort_by_timestamp(
                [image_name for image_name in os.listdir(l2_masked_image_dir) if image_name.endswith(".tif")],
                recent_first=False
            )

        tile_log.info("Updating most recent composite with new imagery over detected changed areas.")
        for categorised_image in categorised_images:
            # Find corresponding L2A file
            timestamp = pyeo.filesystem_utilities.get_change_detection_date_strings(os.path.basename(categorised_image))
            before_time = timestamp[0]
            after_time = timestamp[1]
            granule = pyeo.filesystem_utilities.get_sen_2_image_tile(os.path.basename(categorised_image))
            l2a_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.tif".format(after_time, granule)
            tile_log.info("Searching for image name pattern: {}".format(l2a_glob))
            l2a_image = glob.glob(os.path.join(l2_masked_image_dir, l2a_glob))
            if len(l2a_image) == 0:
                tile_log.warning("Matching L2A file not found for categorised image {}".format(categorised_image))
            else:
                l2a_image = l2a_image[0]
            tile_log.info("Categorised image: {}".format(categorised_image))
            tile_log.info("Matching stacked masked L2A file: {}".format(l2a_image))

            # Extract all reflectance values from the pixels with the class of interest in the classified image
            with TemporaryDirectory(dir=os.getcwd()) as td:
                tile_log.info("Creating mask file from categorised image {} for class: {}".format(os.path.join(categorised_image_dir, categorised_image), class_of_interest))
                mask_path = os.path.join(td, categorised_image.split(sep=".")[0]+".msk")
                tile_log.info("  at {}".format(mask_path))
                pyeo.raster_manipulation.create_mask_from_class_map(os.path.join(categorised_image_dir, categorised_image),
                                                                    mask_path, [class_of_interest], buffer_size=0, out_resolution=None)
                masked_image_path = os.path.join(td, categorised_image.split(sep=".")[0]+"_change.tif")
                pyeo.raster_manipulation.apply_mask_to_image(mask_path, l2a_image, masked_image_path)
                new_composite_path = os.path.join(composite_dir, "composite_{}.tif".format(
                                                    pyeo.filesystem_utilities.get_sen_2_image_timestamp(os.path.basename(l2a_image))))
                # Update pixel values in the composite over the selected pixel locations where values are not missing
                tile_log.info("  {}".format(latest_composite_path))
                tile_log.info("  {}".format([l2a_image]))
                tile_log.info("  {}".format(new_composite_path))
                # todo generate_date_image=True currently produces a type error
                pyeo.raster_manipulation.update_composite_with_images(
                                                                        latest_composite_path,
                                                                        [masked_image_path],
                                                                        new_composite_path,
                                                                        generate_date_image=False,
                                                                        missing=0
                                                                        )
            latest_composite_path = new_composite_path
        """

    tile_log.info("---------------------------------------------------------------")
    tile_log.info("---                  PROCESSING END                         ---")
    tile_log.info("---------------------------------------------------------------")

    # except Exception as error:
    #     # this log needs to stay as the "main" log, which is `log`
    #     log.error(f"Could not complete ACD Raster Process for Tile  {tile}")
    #     log.error(f"error received   :  {error}")


# if run from terminal, do this:
if __name__ == "__main__":
    # assuming argv[0] is script name, config_path passed as 1 and tile string as 2
    acd_by_tile_raster(config_path=sys.argv[1], tile=sys.argv[2])
