"""
rolling_composite_s2_change_detection
-------------------------------------
An app for providing continuous change detection. Runs the following algorithm

 Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map

 Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C).
         Preprocess all L1C images with Sen2Cor to make a cloud mask and atmospherically correct it to L2A.
         For each L2A image, get the directory paths of the separate band raster files.

 Step 3: For each L2A image, pair it with the composite baseline map and classify a change map using a saved model

 Step 4: Update the baseline composite with the reflectance values of only the changed pixels.
         Update last_date of the baseline composite.

 """
import shutil
import sys

import pyeo.classification
import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities
from pyeo.filesystem_utilities import get_filenames


import configparser
import copy
import argparse
import glob
import numpy as np
import os
from osgeo import gdal
import pandas as pd
import datetime as dt
from tempfile import TemporaryDirectory

gdal.UseExceptions()


def rolling_detection(
    config_path,
    arg_start_date=None,
    arg_end_date=None,
    tile_id=None,
    num_chunks=None,
    build_composite=False,
    do_download=False,
    download_source="scihub",
    build_prob_image=False,
    do_classify=False,
    do_update=False,
    do_quicklooks=False,
    do_delete=False,
):
    # If any processing step args are present, do not assume that we want to do all steps
    do_all = True
    if (
        build_composite
        or do_download
        or do_classify
        or do_update
        or do_delete
        or do_quicklooks
    ) == True:
        do_all = False
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(config_path)
    sen_user = conf["sent_2"]["user"]
    sen_pass = conf["sent_2"]["pass"]
    root_dir = conf["forest_sentinel"]["root_dir"]
    tile_root_dir = os.path.join(root_dir, tile_id)
    composite_start_date = conf["forest_sentinel"]["composite_start"]
    composite_end_date = conf["forest_sentinel"]["composite_end"]
    start_date = conf["forest_sentinel"]["start_date"]
    end_date = conf["forest_sentinel"]["end_date"]
    cloud_cover = conf["forest_sentinel"]["cloud_cover"]
    cloud_certainty_threshold = int(
        conf["forest_sentinel"]["cloud_certainty_threshold"]
    )
    model_path = conf["forest_sentinel"]["model"]
    sen2cor_path = conf["sen2cor"]["path"]
    epsg = int(conf["forest_sentinel"]["epsg"])

    pyeo.filesystem_utilities.create_folder_structure_for_tiles(tile_root_dir)
    log = pyeo.filesystem_utilities.init_log(
        os.path.join(tile_root_dir, "log", tile_id + "_log.txt")
    )
    log.info("Options:")
    if do_all:
        log.info("  --do_all")
    if build_composite:
        log.info("  --build_composite for baseline composite")
        log.info("  --download_source = {}".format(download_source))
    if do_download:
        log.info("  --do_download for change detection images")
        log.info("  --download_source = {}".format(download_source))
    if do_classify:
        log.info(
            "  --do_classify to apply the random forest model and create classification layers"
        )
    if build_prob_image:
        log.info("  --build_prob_image to save classification probability layers")
    if do_update:
        log.info("  --do_update to update the baseline composite with new observations")
    if do_quicklooks:
        log.info("  --do_quicklooks to create image quicklooks")
    if do_delete:
        log.info(
            "  --do_delete to remove the downloaded L1C, L2A and cloud-masked composite layers after use"
        )

    log.info("Creating the directory structure if not already present")

    try:
        l1_image_dir = os.path.join(tile_root_dir, r"images/L1C")
        l2_image_dir = os.path.join(tile_root_dir, r"images/L2A")
        l2_masked_image_dir = os.path.join(tile_root_dir, r"images/cloud_masked")
        categorised_image_dir = os.path.join(tile_root_dir, r"output/classified")
        probability_image_dir = os.path.join(tile_root_dir, r"output/probabilities")
        composite_dir = os.path.join(tile_root_dir, r"composite")
        composite_l1_image_dir = os.path.join(tile_root_dir, r"composite/L1C")
        composite_l2_image_dir = os.path.join(tile_root_dir, r"composite/L2A")
        composite_l2_masked_image_dir = os.path.join(
            tile_root_dir, r"composite/cloud_masked"
        )
        quicklook_dir = os.path.join(tile_root_dir, r"output/quicklooks")

        if arg_start_date == "LATEST":
            # Returns the yyyymmdd string of the latest classified image
            start_date = pyeo.filesystem_utilities.get_image_acquisition_time(
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(categorised_image_dir)
                        if image_name.endswith(".tif")
                    ],
                    recent_first=True,
                )[0]
            ).strftime("%Y%m%d")
        elif arg_start_date:
            start_date = arg_start_date

        if arg_end_date == "TODAY":
            end_date = dt.date.today().strftime("%Y%m%d")
        elif arg_end_date:
            end_date = arg_end_date

        # ------------------------------------------------------------------------
        # Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map
        # ------------------------------------------------------------------------

        if build_composite or do_all:
            log.info("---------------------------------------------------------------")
            log.info(
                "Creating an initial cloud-free median composite from Sentinel-2 as a baseline map"
            )
            log.info("---------------------------------------------------------------")
            log.info("Searching for images for initial composite.")

            """ 
            # could use this as a shortcut
            
            test1 = api.query(tileid = tile_id, platformname = 'Sentinel-2', processinglevel = 'Level-1C')
            test2 = api.query(tileid = tile_id, platformname = 'Sentinel-2', processinglevel = 'Level-2A')
            
            """

            composite_products_all = (
                pyeo.queries_and_downloads.check_for_s2_data_by_date(
                    root_dir,
                    composite_start_date,
                    composite_end_date,
                    conf,
                    cloud_cover=cloud_cover,
                    tile_id=tile_id,
                    producttype=None,  # "S2MSI2A" or "S2MSI1C"
                )
            )
            log.info(
                "--> Found {} L1C and L2A products for the composite:".format(
                    len(composite_products_all)
                )
            )
            df = pd.DataFrame.from_dict(composite_products_all, orient="index")
            l1c_products = df[df.processinglevel == "Level-1C"]
            l2a_products = df[df.processinglevel == "Level-2A"]
            log.info("    {} L1C products".format(l1c_products.shape[0]))
            log.info("    {} L2A products".format(l2a_products.shape[0]))

            # during compositing stage, limit the number of images to download
            if l1c_products.shape[0] > max_image_number:
                log.info(
                    "Capping the number of L1C products to {}".format(max_image_number)
                )
                log.info(
                    "Cloud cover per image in ascending order: {}".format(
                        l1c_products.sort_values(
                            by=["cloudcoverpercentage"], ascending=True
                        )["cloudcoverpercentage"]
                    )
                )
                l1c_products = l1c_products.sort_values(
                    by=["cloudcoverpercentage"], ascending=True
                )[:max_image_number]
                log.info("    {} L1C products remain".format(l1c_products.shape[0]))
            if l2a_products.shape[0] > max_image_number:
                log.info(
                    "Capping the number of L2A products to {}".format(max_image_number)
                )
                log.info(
                    "Cloud cover per image in ascending order: {}".format(
                        l2a_products.sort_values(
                            by=["cloudcoverpercentage"], ascending=True
                        )["cloudcoverpercentage"]
                    )
                )
                l2a_products = l2a_products.sort_values(
                    by=["cloudcoverpercentage"], ascending=True
                )[:max_image_number]
                log.info("    {} L2A products remain".format(l2a_products.shape[0]))

            if l1c_products.shape[0] > 0 and l2a_products.shape[0] > 0:
                log.info(
                    "Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product."
                )
                (
                    l1c_products,
                    l2a_products,
                ) = pyeo.queries_and_downloads.filter_unique_l1c_and_l2a_data(df)
                log.info(
                    "--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(
                        l1c_products.shape[0] + l2a_products.shape[0]
                    )
                )
                log.info("    {} L1C products".format(l1c_products.shape[0]))
                log.info("    {} L2A products".format(l2a_products.shape[0]))
            df = None

            if l1c_products.shape[0] > 0:
                log.info(
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
                    log.info("Search term: {}.".format(search_term))
                    matching_l2a_products = (
                        pyeo.queries_and_downloads._file_api_query(
                            user=sen_user,
                            passwd=sen_pass,
                            start_date=composite_start_date,
                            end_date=composite_end_date,
                            filename=search_term,
                            cloud=cloud_cover,
                            producttype="S2MSI2A",
                        )
                    )
                    matching_l2a_products_df = pd.DataFrame.from_dict(
                        matching_l2a_products, orient="index"
                    )
                    if len(matching_l2a_products_df) == 1:
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                    if len(matching_l2a_products_df) == 0:
                        log.info("Found no match for L1C: {}.".format(id))
                    if len(matching_l2a_products_df) > 1:
                        log.warning("Several matches found for L1C product.")
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                if len(drop) > 0:
                    l1c_products = l1c_products.drop(index=drop)
                if len(add) > 0:
                    l2a_products = l2a_products.append(add)
                l2a_products = l2a_products.drop_duplicates(subset="title")
                log.info(
                    "    {} L1C products remaining for download".format(
                        l1c_products.shape[0]
                    )
                )
                log.info(
                    "    {} L2A products remaining for download".format(
                        l2a_products.shape[0]
                    )
                )

                log.info("Downloading Sentinel-2 L1C products.")
                # TODO: Need to collect the response from download_from_scihub function and check whether the download succeeded
                pyeo.queries_and_downloads.download_s2_data_from_df(
                    l1c_products,
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    download_source,
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
                log.info("Atmospheric correction with sen2cor.")
                pyeo.raster_manipulation.atmospheric_correction(
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    sen2cor_path,
                    delete_unprocessed_image=False,
                )
            if l2a_products.shape[0] > 0:
                log.info("Downloading Sentinel-2 L2A products.")
                pyeo.queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    composite_l1_image_dir,
                    composite_l2_image_dir,
                    download_source,
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
            log.info("---------------------------------------------------------------")
            log.info(
                "Image download and atmospheric correction for composite is complete."
            )
            log.info("---------------------------------------------------------------")

            l2a_paths = [
                f.path for f in os.scandir(composite_l2_image_dir) if f.is_dir()
            ]
            # raster_paths = pyeo.filesystem_utilities.get_raster_paths(l2a_paths, filepatterns=bands, dirpattern=resolution) # don't really need to know these
            # scl_raster_paths = pyeo.filesystem_utilities.get_raster_paths(l2a_paths, filepatterns=["SCL"], dirpattern="20m") # don't really need to know these

            log.info(
                "Applying simple cloud and cloud shadow mask based on SCL files and stacking the masked band raster files."
            )
            pyeo.raster_manipulation.apply_scl_cloud_mask(
                composite_l2_image_dir,
                composite_l2_masked_image_dir,
                scl_classes=[0, 1, 2, 3, 8, 9, 10, 11],
                buffer_size=buffer_size_composite,
                bands=bands,
                out_resolution=10,
            )

            log.info(
                "Building initial cloud-free median composite from directory {}".format(
                    composite_l2_masked_image_dir
                )
            )
            pyeo.raster_manipulation.clever_composite_directory(
                composite_l2_masked_image_dir,
                composite_dir,
                chunks=5,
                generate_date_images=True,
                missing_data_value=0,
            )

            log.info("---------------------------------------------------------------")
            log.info("Baseline image composite is complete.")
            log.info("---------------------------------------------------------------")

        # ------------------------------------------------------------------------
        # Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C)
        # ------------------------------------------------------------------------
        if do_all or do_download:
            log.info("---------------------------------------------------------------")
            log.info(
                "Downloading change detection images between {} and {} with cloud cover <= {}".format(
                    start_date, end_date, cloud_cover
                )
            )
            log.info("---------------------------------------------------------------")

            products_all = pyeo.queries_and_downloads.check_for_s2_data_by_date(
                root_dir,
                start_date,
                end_date,
                conf,
                cloud_cover=cloud_cover,
                tile_id=tile_id,
                producttype=None,  # "S2MSI2A" or "S2MSI1C"
            )
            log.info(
                "--> Found {} L1C and L2A products for change detection:".format(
                    len(products_all)
                )
            )
            df = pd.DataFrame.from_dict(products_all, orient="index")
            l1c_products = df[df.processinglevel == "Level-1C"]
            l2a_products = df[df.processinglevel == "Level-2A"]
            log.info("    {} L1C products".format(l1c_products.shape[0]))
            log.info("    {} L2A products".format(l2a_products.shape[0]))

            if l1c_products.shape[0] > 0 and l2a_products.shape[0] > 0:
                log.info(
                    "Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product."
                )
                (
                    l1c_products,
                    l2a_products,
                ) = pyeo.queries_and_downloads.filter_unique_l1c_and_l2a_data(df)
                log.info(
                    "--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(
                        l1c_products.shape[0] + l2a_products.shape[0]
                    )
                )
                log.info("    {} L1C products".format(l1c_products.shape[0]))
                log.info("    {} L2A products".format(l2a_products.shape[0]))
            df = None

            if l1c_products.shape[0] > 0:
                log.info(
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
                    log.info("Search term: {}.".format(search_term))
                    matching_l2a_products = (
                        pyeo.queries_and_downloads._file_api_query(
                            user=sen_user,
                            passwd=sen_pass,
                            start_date=start_date,
                            end_date=end_date,
                            filename=search_term,
                            cloud=cloud_cover,
                            producttype="S2MSI2A",
                        )
                    )
                    matching_l2a_products_df = pd.DataFrame.from_dict(
                        matching_l2a_products, orient="index"
                    )
                    if len(matching_l2a_products_df) == 1:
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                    if len(matching_l2a_products_df) == 0:
                        log.info("Found no match for L1C: {}.".format(id))
                    if len(matching_l2a_products_df) > 1:
                        log.warning("Several matches found for L1C product.")
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                if len(drop) > 0:
                    l1c_products = l1c_products.drop(index=drop)
                if len(add) > 0:
                    l2a_products = l2a_products.append(add)
                log.info(
                    "    {} L1C products remaining for download".format(
                        l1c_products.shape[0]
                    )
                )
                log.info(
                    "    {} L2A products remaining for download".format(
                        l2a_products.shape[0]
                    )
                )
                l2a_products = l2a_products.drop_duplicates(subset="title")
                log.info("Downloading Sentinel-2 L1C products.")
                pyeo.queries_and_downloads.download_s2_data_from_df(
                    l1c_products,
                    l1_image_dir,
                    l2_image_dir,
                    download_source,
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
                log.info("Atmospheric correction with sen2cor.")
                pyeo.raster_manipulation.atmospheric_correction(
                    l1_image_dir,
                    l2_image_dir,
                    sen2cor_path,
                    delete_unprocessed_image=False,
                )
            if l2a_products.shape[0] > 0:
                log.info("Downloading Sentinel-2 L2A products.")
                pyeo.queries_and_downloads.download_s2_data(
                    l2a_products.to_dict("index"),
                    l1_image_dir,
                    l2_image_dir,
                    download_source,
                    user=sen_user,
                    passwd=sen_pass,
                    try_scihub_on_fail=True,
                )
            log.info("---------------------------------------------------------------")
            log.info(
                "Image download and atmospheric correction for composite is complete."
            )
            log.info("---------------------------------------------------------------")
            log.info(
                "Applying simple cloud and cloud shadow mask based on SCL files and stacking the masked band raster files."
            )
            l2a_paths = [f.path for f in os.scandir(l2_image_dir) if f.is_dir()]
            log.info("  l2_image_dir: {}".format(l2_image_dir))
            log.info("  l2_masked_image_dir: {}".format(l2_masked_image_dir))
            log.info("  bands: {}".format(bands))
            pyeo.raster_manipulation.apply_scl_cloud_mask(
                l2_image_dir,
                l2_masked_image_dir,
                scl_classes=[0, 1, 2, 3, 8, 9, 10, 11],
                buffer_size=buffer_size,
                bands=bands,
                out_resolution=10,
            )

        # ------------------------------------------------------------------------
        # Step 3: For each L2A image, pair it with the composite baseline map and classify a change map using a saved model
        # ------------------------------------------------------------------------
        # TODO: Check why in some classification image file names the later timestamp comes before the earlier timestamp(?)
        if do_all or do_classify:
            log.info("---------------------------------------------------------------")
            log.info(
                "Classify a change detection for each L2A image using a saved model"
            )
            log.info("---------------------------------------------------------------")
            if skip_existing:
                log.info("Skipping existing classification images if found.")
            l2a_paths = [f.path for f in os.scandir(l2_masked_image_dir) if f.is_file()]
            if len(l2a_paths) == 0:
                raise FileNotFoundError(
                    "No images found in {}".format(l2_masked_image_dir)
                )

            log.info("Sorting masked L2A image list by time stamp.")
            images = pyeo.filesystem_utilities.sort_by_timestamp(
                [
                    image_name
                    for image_name in os.listdir(l2_masked_image_dir)
                    if image_name.endswith(".tif")
                ],
                recent_first=False,
            )

            try:
                latest_composite_name = pyeo.filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(composite_dir)
                        if image_name.endswith(".tif")
                    ],
                    recent_first=True,
                )[0]
                latest_composite_path = os.path.join(
                    composite_dir, latest_composite_name
                )
                log.info("Most recent composite at {}".format(latest_composite_path))
            except IndexError:
                log.critical(
                    "Latest composite not found. The first time you run this script, you need to include the "
                    "--build-composite flag to create a base composite to work off. If you have already done this,"
                    "check that the earliest dated image in your images/merged folder is later than the earliest"
                    " dated image in your composite/ folder."
                )
                sys.exit(1)

            for image in images:
                log.info("Change detection between images:")
                log.info("  Latest composite      : {}".format(latest_composite_path))
                log.info("  Change detection image: {}".format(image))
                # new_class_image should have the before and after dates/timestamps in the filename
                before_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        latest_composite_name
                    )
                )
                after_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        os.path.basename(image)
                    )
                )
                # only classify if the image is more recent than the latest composite
                if before_timestamp < after_timestamp:
                    before_timestamp_str = (
                        pyeo.filesystem_utilities.get_sen_2_image_timestamp(
                            latest_composite_name
                        )
                    )
                    after_timestamp_str = (
                        pyeo.filesystem_utilities.get_sen_2_image_timestamp(
                            os.path.basename(image)
                        )
                    )
                    processing_baseline_number = os.path.basename(image).split("_")[
                        3
                    ]  # Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
                    orbit = os.path.basename(image).split("_")[
                        4
                    ]  # ROOO: Relative Orbit number (R001 - R143)
                    tile = os.path.basename(image).split("_")[
                        5
                    ]  # Tile ID number (e.g. T21MVM)
                    new_class_image = os.path.join(
                        categorised_image_dir,
                        "class_{}_{}_{}_{}_{}.tif".format(
                            before_timestamp_str,
                            processing_baseline_number,
                            orbit,
                            tile,
                            after_timestamp_str,
                        ),
                    )
                    if build_prob_image:
                        new_prob_image = os.path.join(
                            probability_image_dir,
                            "prob_{}_{}_{}_{}_{}.tif".format(
                                before_timestamp_str,
                                processing_baseline_number,
                                orbit,
                                tile,
                                after_timestamp_str,
                            ),
                        )
                    else:
                        new_prob_image = None
                    pyeo.classification.change_from_composite(
                        os.path.join(l2_masked_image_dir, image),
                        latest_composite_path,
                        model_path,
                        new_class_image,
                        new_prob_image,
                        skip_existing=skip_existing,
                        apply_mask=False,
                    )
            log.info("End of classification.")

            # TODO: test this bit
            log.info(
                "Creating confidence layers based on {} repeated subsequent change detections.".format(
                    n_confirmations
                )
            )
            class_image_paths = [
                f.path
                for f in os.scandir(categorised_image_dir)
                if f.is_file() and f.name.endswith(".tif")
            ]
            if len(class_image_paths) == 0:
                raise FileNotFoundError(
                    "No categorised images found in {}.".format(categorised_image_dir)
                )

            # sort class images by change detection date (the second timestamp in the change detection image name)
            class_image_paths = list(
                filter(
                    pyeo.filesystem_utilities.get_change_detection_dates,
                    class_image_paths,
                )
            )
            class_image_paths.sort(
                key=lambda x: pyeo.filesystem_utilities.get_change_detection_dates(x)[
                    1
                ]
            )

            for index, image in enumerate(class_image_paths):
                log.info("{}: {}".format(index, image))
            for index, image in enumerate(class_image_paths):
                log.info(
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(image)
                    )[1]
                )

            # combine masks from n subsequent dates into a confirmed change detection image
            for index, image in enumerate(class_image_paths, start=n_confirmations):
                # submit all images from 0...n_confirmations to first round of verification,
                #   then increase the counter by one
                subsequent_images = class_image_paths[index - n_confirmations : index]
                # build output file name from earliest and latest time stamp
                #   (new_class_image should have the before and after dates/timestamps in the filename)
                before_timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(subsequent_images[0])
                    )[0]
                )
                after_timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(subsequent_images[-1])
                    )[1]
                )

                log.info("  subsequent images:")
                for im in subsequent_images:
                    log.info("  {}".format(im))
                log.info("  earliest time stamp: {}".format(before_timestamp))
                log.info("  latest   time stamp: {}".format(after_timestamp))

                verification_raster = os.path.join(
                    probability_image_dir,
                    "conf_{}_{}_{}_n{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                        n_confirmations,
                    ),
                )
                log.info(
                    "  verification raster file to be created: {}".format(
                        verification_raster
                    )
                )
                pyeo.raster_manipulation.verify_change_detections(
                    subsequent_images,
                    verification_raster,
                    [5],
                    buffer_size=0,
                    out_resolution=None,
                )

            log.info("Confidence layers done.")

        # ------------------------------------------------------------------------
        # Step 4: Update the baseline composite with the reflectance values of only the changed pixels.
        #         Update last_date of the baseline composite.
        # ------------------------------------------------------------------------

        if do_update or do_all:
            log.info("---------------------------------------------------------------")
            log.info("Updating baseline composite with new imagery.")
            log.info("---------------------------------------------------------------")
            # get all composite file paths
            composite_paths = [f.path for f in os.scandir(composite_dir) if f.is_file()]
            if len(composite_paths) == 0:
                raise FileNotFoundError(
                    "No composite images found in {}.".format(composite_dir)
                )
            log.info("Sorting composite image list by time stamp.")
            composite_images = pyeo.filesystem_utilities.sort_by_timestamp(
                [
                    image_name
                    for image_name in os.listdir(composite_dir)
                    if image_name.endswith(".tif")
                ],
                recent_first=False,
            )
            try:
                latest_composite_name = pyeo.filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(composite_dir)
                        if image_name.endswith(".tif")
                    ],
                    recent_first=True,
                )[0]
                latest_composite_path = os.path.join(
                    composite_dir, latest_composite_name
                )
                latest_composite_timestamp = (
                    pyeo.filesystem_utilities.get_sen_2_image_timestamp(
                        os.path.basename(latest_composite_path)
                    )
                )
                log.info("Most recent composite at {}".format(latest_composite_path))
            except IndexError:
                log.critical(
                    "Latest composite not found. The first time you run this script, you need to include the "
                    "--build-composite flag to create a base composite to work off. If you have already done this,"
                    "check that the earliest dated image in your images/merged folder is later than the earliest"
                    " dated image in your composite/ folder."
                )
                sys.exit(1)

            # Find all categorised images
            categorised_paths = [
                f.path for f in os.scandir(categorised_image_dir) if f.is_file()
            ]
            if len(categorised_paths) == 0:
                raise FileNotFoundError(
                    "No categorised images found in {}.".format(categorised_image_dir)
                )
            log.info("Sorting categorised image list by time stamp.")
            categorised_images = pyeo.filesystem_utilities.sort_by_timestamp(
                [
                    image_name
                    for image_name in os.listdir(categorised_image_dir)
                    if image_name.endswith(".tif")
                ],
                recent_first=False,
            )
            # Drop the categorised images that were made before the most recent composite date
            latest_composite_timestamp_datetime = (
                pyeo.filesystem_utilities.get_image_acquisition_time(
                    latest_composite_name
                )
            )
            categorised_images = [
                image
                for image in categorised_images
                if pyeo.filesystem_utilities.get_change_detection_dates(
                    os.path.basename(image)
                )[1]
                > latest_composite_timestamp_datetime
            ]

            # Find all L2A images
            l2a_paths = [f.path for f in os.scandir(l2_masked_image_dir) if f.is_file()]
            if len(l2a_paths) == 0:
                raise FileNotFoundError(
                    "No images found in {}.".format(l2_masked_image_dir)
                )
            log.info("Sorting masked L2A image list by time stamp.")
            l2a_images = pyeo.filesystem_utilities.sort_by_timestamp(
                [
                    image_name
                    for image_name in os.listdir(l2_masked_image_dir)
                    if image_name.endswith(".tif")
                ],
                recent_first=False,
            )

            log.info(
                "Updating most recent composite with new imagery over detected changed areas."
            )
            for categorised_image in categorised_images:
                # Find corresponding L2A file
                timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_date_strings(
                        os.path.basename(categorised_image)
                    )
                )
                before_time = timestamp[0]
                after_time = timestamp[1]
                granule = pyeo.filesystem_utilities.get_sen_2_image_tile(
                    os.path.basename(categorised_image)
                )
                l2a_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.tif".format(after_time, granule)
                log.info("Searching for image name pattern: {}".format(l2a_glob))
                l2a_image = glob.glob(os.path.join(l2_masked_image_dir, l2a_glob))
                if len(l2a_image) == 0:
                    log.warning(
                        "Matching L2A file not found for categorised image {}".format(
                            categorised_image
                        )
                    )
                else:
                    l2a_image = l2a_image[0]
                log.info("Categorised image: {}".format(categorised_image))
                log.info("Matching stacked masked L2A file: {}".format(l2a_image))

                # Extract all reflectance values from the pixels with the class of interest in the classified image
                with TemporaryDirectory(dir=os.getcwd()) as td:
                    log.info(
                        "Creating mask file from categorised image {} for class: {}".format(
                            os.path.join(categorised_image_dir, categorised_image),
                            class_of_interest,
                        )
                    )
                    mask_path = os.path.join(
                        td, categorised_image.split(sep=".")[0] + ".msk"
                    )
                    log.info("  at {}".format(mask_path))
                    pyeo.raster_manipulation.create_mask_from_class_map(
                        os.path.join(categorised_image_dir, categorised_image),
                        mask_path,
                        [class_of_interest],
                        buffer_size=0,
                        out_resolution=None,
                    )
                    masked_image_path = os.path.join(
                        td, categorised_image.split(sep=".")[0] + "_change.tif"
                    )
                    pyeo.raster_manipulation.apply_mask_to_image(
                        mask_path, l2a_image, masked_image_path
                    )
                    new_composite_path = os.path.join(
                        composite_dir,
                        "composite_{}.tif".format(
                            pyeo.filesystem_utilities.get_sen_2_image_timestamp(
                                os.path.basename(l2a_image)
                            )
                        ),
                    )
                    # Update pixel values in the composite over the selected pixel locations where values are not missing
                    log.info("  {}".format(latest_composite_path))
                    log.info("  {}".format([l2a_image]))
                    log.info("  {}".format(new_composite_path))
                    # TODO generate_date_image=True currently produces a type error
                    pyeo.raster_manipulation.update_composite_with_images(
                        latest_composite_path,
                        [masked_image_path],
                        new_composite_path,
                        generate_date_image=False,
                        missing=0,
                    )
                latest_composite_path = new_composite_path

        # ------------------------------------------------------------------------
        # Step 5: Create quicklooks for fast visualisation and quality assurance of output
        # ------------------------------------------------------------------------

        if do_quicklooks or do_all:
            log.info("---------------------------------------------------------------")
            log.info("Producing quicklooks.")
            log.info("---------------------------------------------------------------")
            dirs_for_quicklooks = [
                composite_dir,
                l2_masked_image_dir,
                categorised_image_dir,
                probability_image_dir,
            ]
            for main_dir in dirs_for_quicklooks:
                files = [
                    f.path
                    for f in os.scandir(main_dir)
                    if f.is_file() and os.path.basename(f).endswith(".tif")
                ]
                if len(files) == 0:
                    log.warning("No images found in {}.".format(main_dir))
                else:
                    for f in files:
                        log.info("Creating quicklook image from: {}".format(f))
                        quicklook_path = os.path.join(
                            quicklook_dir, os.path.basename(f).split(".")[0] + ".png"
                        )
                        log.info(
                            "                           at: {}".format(quicklook_path)
                        )
                        pyeo.raster_manipulation.create_quicklook(
                            f,
                            quicklook_path,
                            width=512,
                            height=512,
                            format="PNG",
                            bands=[3, 2, 1],
                            scale_factors=[[0, 2000, 0, 255]],
                        )
            log.info("Quicklooks complete.")

        # ------------------------------------------------------------------------
        # Step 6: Free up disk space by deleting all downloaded Sentinel-2 images and intermediate processing steps
        # ------------------------------------------------------------------------

        # Build new composite
        if do_delete:
            log.info("---------------------------------------------------------------")
            log.info(
                "Deleting downloaded images and intermediate products after use to free up disk space."
            )
            log.info("---------------------------------------------------------------")
            log.warning("This function is currently disabled.")
            """
            shutil.rmtree(l1_image_dir)
            shutil.rmtree(l2_image_dir)
            shutil.rmtree(l2_masked_image_dir)
            shutil.rmtree(composite_l1_image_dir)
            shutil.rmtree(composite_l2_image_dir)
            shutil.rmtree(composite_l2_masked_image_dir)
            """

        # ------------------------------------------------------------------------
        # End of processing
        # ------------------------------------------------------------------------
        log.info("---------------------------------------------------------------")
        log.info("---                  PROCESSING END                         ---")
        log.info("---------------------------------------------------------------")

    except Exception:
        log.exception("Fatal error in rolling_s2_composite chain")


if __name__ == "__main__":
    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Downloads, preprocesses and classifies Sentinel 2 images. A directory"
        "structure to contain preprocessed and downloaded files will be"
        "created at the root_dir location specified in the config file."
        "If any of the step flags are present, only those "
        "steps will be performed - otherwise all processing steps will be "
        "performed."
    )
    parser.add_argument(
        dest="config_path",
        action="store",
        default=r"change_detection.ini",
        help="A path to a .ini file containing the specification for the job. See "
        "pyeo/apps/change_detection/change_detection.ini for an example.",
    )
    parser.add_argument(
        "--start_date",
        dest="arg_start_date",
        help="Overrides the start date in the config file. Set to "
        "LATEST to get the date of the last merged accquistion",
    )
    parser.add_argument(
        "--end_date",
        dest="arg_end_date",
        help="Overrides the end date in the config file. Set to TODAY"
        "to get today's date",
    )
    parser.add_argument(
        "-b",
        "--build_composite",
        dest="build_composite",
        action="store_true",
        default=False,
        help="If present, creates a cloud-free (ish) composite between the two dates specified in the "
        "config file.",
    )
    parser.add_argument(
        "--tile",
        dest="tile_id",
        type=str,
        default="None",
        help="Overrides the geojson location with a" "Sentinel-2 tile ID location",
    )
    parser.add_argument(
        "--chunks",
        dest="num_chunks",
        type=int,
        default=10,
        help="Sets the number of chunks to split " "images to in ml processing",
    )
    parser.add_argument(
        "-d",
        "--download",
        dest="do_download",
        action="store_true",
        default=False,
        help="If present, perform the query and download level 1 images.",
    )
    parser.add_argument(
        "--download_source",
        default="scihub",
        help="Sets the download source, can be scihub " "(default) or aws",
    )
    parser.add_argument(
        "-c",
        "--classify",
        dest="do_classify",
        action="store_true",
        default=False,
        help="For each image in images/stacked, applies the classifier given in the .ini file. Saves"
        "the outputs in output/categories.",
    )
    parser.add_argument(
        "-p",
        "--build_prob_image",
        dest="build_prob_image",
        action="store_true",
        default=False,
        help="If present, build a confidence map of pixels. These tend to be large.",
    )
    parser.add_argument(
        "-u",
        "--update",
        dest="do_update",
        action="store_true",
        default=False,
        help="Builds a new cloud-free composite in composite/ from the latest image and mask"
        " in images/merged",
    )
    parser.add_argument(
        "-q",
        "--quicklooks",
        dest="do_quicklooks",
        action="store_true",
        default=False,
        help="Creates quicklooks for all composites, L2A change images, classified images and probability images.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        dest="do_delete",
        action="store_true",
        default=False,
        help="Not implemented. If present, removes all images in images/ to save space.",
    )

    args = parser.parse_args()

    # TODO: bands and resolution can be made flexible BUT the bands need to be at the same resolution
    bands = ["B02", "B03", "B04", "B08"]
    resolution = "10m"
    buffer_size = 30  # set buffer in number of pixels for dilating the SCL cloud mask (recommend 30 pixels of 10 m) for the change detection
    buffer_size_composite = 10  # set buffer in number of pixels for dilating the SCL cloud mask (recommend 10 pixels of 10 m) for the composite building
    max_image_number = 30  # maximum number of images to be downloaded for compositing, in order of least cloud cover
    class_of_interest = 5  # depends on the model used in the classification - 5 is the forest loss class from Valentin's model
    n_confirmations = (
        2  # number of subsequent change detections for verification purposes
    )
    skip_existing = False  # skip existing classification images

    rolling_detection(**vars(args))
