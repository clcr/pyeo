"""
rolling_composite_s2_change_detection
-------------------------------------
An app for providing continuous change detection based on classification of forest cover. Runs the following algorithm

 Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map

 Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C).
         Preprocess all L1C images with Sen2Cor to make a cloud mask and atmospherically correct it to L2A.
         For each L2A image, get the directory paths of the separate band raster files.

 Step 3: Classify each L2A image and the baseline composite

 Step 4: Pair up successive classified images with the composite baseline map and identify all pixels with the change between classes of interest, e.g. from class 1 to 2,3 or 8

 Step 5: Update the baseline composite with the reflectance values of only the changed pixels. Update last_date of the baseline composite.

 Step 6: Create quicklooks.

 """
import shutil
import sys

import pyeo.classification
import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities

# from pyeo.filesystem_utilities import get_filenames


import configparser

# import copy
import argparse

# import glob
import json
import numpy as np
import os
from osgeo import gdal
import pandas as pd
import datetime as dt
import zipfile

# from tempfile import TemporaryDirectory

gdal.UseExceptions()


def rolling_detection(
    config_path,
    arg_start_date=None,
    arg_end_date=None,
    tile_id=None,
    chunks=None,
    build_composite=False,
    do_download=False,
    download_source="scihub",
    build_prob_image=False,
    do_classify=False,
    do_change=False,
    do_dev=False,
    do_update=False,
    do_quicklooks=False,
    do_delete=False,
    do_zip=False,
    skip_existing=False,
):
    def zip_contents(directory, notstartswith=None):
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

    def unzip_contents(zippath, ifstartswith=None, ending=None):
        dirpath = zippath[:-4]  # cut away the  .zip ending
        if ifstartswith is not None and ending is not None:
            if dirpath.startswith(ifstartswith):
                dirpath = dirpath + ending
        log.info("Unzipping {}".format(zippath))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if os.path.exists(dirpath):
            if os.path.exists(zippath):
                shutil.unpack_archive(
                    filename=zippath, extract_dir=dirpath, format="zip"
                )
                os.remove(zippath)
        else:
            log.error("Unzipping failed")
        return

    # If any processing step args are present, do not assume that we want to do all steps
    do_all = True
    if (
        build_composite
        or do_download
        or do_classify
        or do_change
        or do_update
        or do_delete
        or do_zip
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
    bands = json.loads(conf.get("processing_parameters", "band_names"))
    resolution = conf["processing_parameters"]["resolution_string"]
    out_resolution = int(conf["processing_parameters"]["output_resolution"])
    buffer_size = int(conf["processing_parameters"]["buffer_size_cloud_masking"])
    buffer_size_composite = int(
        conf["processing_parameters"]["buffer_size_cloud_masking_composite"]
    )
    max_image_number = int(conf["processing_parameters"]["download_limit"])
    class_labels = json.loads(conf.get("processing_parameters", "class_labels"))
    from_classes = json.loads(conf.get("processing_parameters", "change_from_classes"))
    to_classes = json.loads(conf.get("processing_parameters", "change_to_classes"))
    faulty_granule_threshold = int(
        conf["processing_parameters"]["faulty_granule_threshold"]
    )
    sieve = int(conf["processing_parameters"]["sieve"])

    pyeo.filesystem_utilities.create_folder_structure_for_tiles(tile_root_dir)
    log = pyeo.filesystem_utilities.init_log(
        os.path.join(tile_root_dir, "log", tile_id + "_log.txt")
    )
    log.info("---------------------------------------------------------------")
    log.info("---   PROCESSING START: {}   ---".format(tile_root_dir))
    log.info("---------------------------------------------------------------")
    log.info("Options:")
    if do_dev:
        log.info(
            "  --dev Running in development mode, choosing development versions of functions where available"
        )
    else:
        log.info(
            "  Running in production mode, avoiding any development versions of functions."
        )
    if do_all:
        log.info("  --do_all")
    if build_composite:
        log.info("  --build_composite for baseline composite")
        log.info("  --download_source = {}".format(download_source))
    if do_download:
        log.info("  --download for change detection images")
        if not build_composite:
            log.info("  --download_source = {}".format(download_source))
    if do_classify:
        log.info(
            "  --classify to apply the random forest model and create classification layers"
        )
    if build_prob_image:
        log.info("  --build_prob_image to save classification probability layers")
    if do_change:
        log.info("  --change to produce change detection layers and report images")
    if do_update:
        log.info("  --update to update the baseline composite with new observations")
    if do_quicklooks:
        log.info("  --quicklooks to create image quicklooks")
    if do_delete:
        log.info("  --remove downloaded L1C images and intermediate image products")
        log.info(
            "           (cloud-masked band-stacked rasters, class images, change layers) after use."
        )
        log.info(
            "           Deletes remaining temporary directories starting with 'tmp' from interrupted processing runs."
        )
        log.info("           Keeps only L2A images, composites and report files.")
        log.info("           Overrides --zip for the above files. WARNING! FILE LOSS!")
    if do_zip:
        log.info(
            "  --zip archives L2A images, and if --remove is not selected also L1C,"
        )
        log.info(
            "           cloud-masked band-stacked rasters, class images and change layers after use."
        )

    log.info("List of image bands: {}".format(bands))
    log.info("Model used: {}".format(model_path))
    log.info("List of class labels:")
    for c, this_class in enumerate(class_labels):
        log.info("  {} : {}".format(c + 1, this_class))
    log.info("Detecting changes from any of the classes: {}".format(from_classes))
    log.info("                    to any of the classes: {}".format(to_classes))

    log.info("\nCreating the directory structure if not already present")

    try:
        change_image_dir = os.path.join(tile_root_dir, r"images")
        l1_image_dir = os.path.join(tile_root_dir, r"images/L1C")
        l2_image_dir = os.path.join(tile_root_dir, r"images/L2A")
        l2_masked_image_dir = os.path.join(tile_root_dir, r"images/cloud_masked")
        categorised_image_dir = os.path.join(tile_root_dir, r"output/classified")
        probability_image_dir = os.path.join(tile_root_dir, r"output/probabilities")
        sieved_image_dir = os.path.join(tile_root_dir, r"output/sieved")
        composite_dir = os.path.join(tile_root_dir, r"composite")
        composite_l1_image_dir = os.path.join(tile_root_dir, r"composite/L1C")
        composite_l2_image_dir = os.path.join(tile_root_dir, r"composite/L2A")
        composite_l2_masked_image_dir = os.path.join(
            tile_root_dir, r"composite/cloud_masked"
        )
        quicklook_dir = os.path.join(tile_root_dir, r"output/quicklooks")

        if arg_start_date == "LATEST":
            report_file_name = [
                f
                for f in os.listdir(probability_image_dir)
                if os.path.isfile(f) and f.startswith("report_") and f.endswith(".tif")
            ][0]
            report_file_path = os.path.join(probability_image_dir, report_file_name)
            after_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(
                os.path.basename(report_file_path)
            )[-1]
            after_timestamp.strftime(
                "%Y%m%d"
            )  # Returns the yyyymmdd string of the acquisition date from which the latest classified image was derived
        elif arg_start_date:
            start_date = arg_start_date

        if arg_end_date == "TODAY":
            end_date = dt.date.today().strftime("%Y%m%d")
        elif arg_end_date:
            end_date = arg_end_date

        # ------------------------------------------------------------------------
        # Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map
        # ------------------------------------------------------------------------

        # TODO: Make the download optional at the compositing stage, i.e. if do_download is not selected, skip it
        #      and only call the median compositing function. Should be a piece of cake.
        # if build_composite or do_all:
        #     if do_download or do_all:
        #         [...download the data for the composite...]
        #     [...calculate the median composite from the available data...]
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

            # TODO: retrieve metadata on nodata percentage and prioritise download of images with low values
            # This method currently only works for L2A products and needs expanding to L1C
            """
            composite_products_all = pyeo.queries_and_downloads.get_nodata_percentage(sen_user, sen_pass, composite_products_all)
            log.info("NO_DATA_PERCENTAGE:")
            for uuid, metadata in composite_products_all.items():
                log.info("{}: {}".format(metadata['title'], metadata['No_data_percentage']))
            """

            log.info(
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
            df = df_all.query("size >= " + str(faulty_granule_threshold))
            log.info(
                "Removed {} faulty scenes <{}MB in size from the list:".format(
                    len(df_all) - len(df), faulty_granule_threshold
                )
            )
            df_faulty = df_all.query("size < " + str(faulty_granule_threshold))
            for r in range(len(df_faulty)):
                log.info(
                    "   {} MB: {}".format(
                        df_faulty.iloc[r, :]["size"], df_faulty.iloc[r, :]["title"]
                    )
                )

            l1c_products = df[df.processinglevel == "Level-1C"]
            l2a_products = df[df.processinglevel == "Level-2A"]
            log.info("    {} L1C products".format(l1c_products.shape[0]))
            log.info("    {} L2A products".format(l2a_products.shape[0]))

            # during compositing stage, limit the number of images to download
            # to avoid only downloading partially covered granules with low cloud cover (which is calculated over the whole granule,
            # incl. missing values), we need to stratify our search for low cloud cover by relative orbit number

            rel_orbits = np.unique(l1c_products["relativeorbitnumber"])
            if len(rel_orbits) > 0:
                if l1c_products.shape[0] > max_image_number / len(rel_orbits):
                    log.info(
                        "Capping the number of L1C products to {}".format(
                            max_image_number
                        )
                    )
                    log.info(
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
                    l1c_products = l1c_products[l1c_products["uuid"].isin(uuids)]
                    log.info(
                        "    {} L1C products remain:".format(l1c_products.shape[0])
                    )
                    for product in l1c_products["title"]:
                        log.info("       {}".format(product))

            rel_orbits = np.unique(l2a_products["relativeorbitnumber"])
            if len(rel_orbits) > 0:
                if l2a_products.shape[0] > max_image_number / len(rel_orbits):
                    log.info(
                        "Capping the number of L2A products to {}".format(
                            max_image_number
                        )
                    )
                    log.info(
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
                    log.info(
                        "    {} L2A products remain:".format(l2a_products.shape[0])
                    )
                    for product in l2a_products["title"]:
                        log.info("       {}".format(product))

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

            # Before the next step, search the composite/L2A and L1C directories whether the scenes have already been downloaded and/or processed and check their dir sizes
            if l1c_products.shape[0] > 0:
                log.info(
                    "Checking for already downloaded and zipped L1C or L2A products and"
                )
                log.info("  availability of matching L2A products for download.")
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
                    log.info(
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
                            log.info("  Product already downloaded: {}".format(f))
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
                    log.info(
                        "Searching on the data hub for files containing: {}.".format(
                            search_term
                        )
                    )
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
                    # 07/03/2023: Applied Ali's fix for converting product size to MB to compare against faulty_grandule_threshold
                    if (
                        len(matching_l2a_products_df) == 1
                        and [
                            float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            for x in [matching_l2a_products_df["size"][0].split(" ")]
                        ][0]
                        > faulty_granule_threshold
                    ):
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info(
                            "              {}".format(
                                matching_l2a_products_df.iloc[0, :]["title"]
                            )
                        )
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0, :])
                    if len(matching_l2a_products_df) == 0:
                        # log.info("Found no match for L1C: {}.".format(id))
                        pass
                    # I.R. Bug Fixed 20230311: Was not handling case where multiple L2A products were returned - now filters by 'size' and sorts df so that largest alternative is downloaded
                    if len(matching_l2a_products_df) > 1:
                        # check granule sizes on the server
                        # print(f"1) matching_l2a_products_df {matching_l2a_products_df[['title', 'size']]}")
                        matching_l2a_products_df["size"] = (
                            matching_l2a_products_df["size"]
                            .str.split(" ")
                            .apply(
                                lambda x: float(x[0])
                                * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]]
                            )
                        )
                        # print(f"2) matching_l2a_products_df {matching_l2a_products_df[['title', 'size']]}")
                        matching_l2a_products_df = matching_l2a_products_df.query(
                            "size >= " + str(faulty_granule_threshold)
                        )
                        # matching_l2a_products_df.sort_values(by='size', ascending=False)
                        # print(f"3) matching_l2a_products_df {matching_l2a_products_df[['title', 'size']]}")
                        # if matching_l2a_products_df.iloc[0,:]['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]]) > faulty_granule_threshold:
                        if len(matching_l2a_products_df) > 0:
                            matching_l2a_products_df.sort_values(
                                by="size", ascending=False
                            )
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
                # I.R.
                log.info(
                    "    {} L2A products remaining for download".format(
                        l2a_products.shape[0]
                    )
                )
                # TODO: Need to collect the response from download_from_scihub function and check whether the download succeeded
                if l1c_products.shape[0] > 0:
                    log.info("Downloading Sentinel-2 L1C products.")
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

            # check for incomplete L2A downloads
            (
                incomplete_downloads,
                sizes,
            ) = pyeo.raster_manipulation.find_small_safe_dirs(
                composite_l2_image_dir, threshold=faulty_granule_threshold * 1024 * 1024
            )
            if len(incomplete_downloads) > 0:
                for index, safe_dir in enumerate(incomplete_downloads):
                    if sizes[
                        index
                    ] / 1024 / 1024 < faulty_granule_threshold and os.path.exists(
                        safe_dir
                    ):
                        log.warning(
                            "Found likely incomplete download of size {} MB: {}".format(
                                str(round(sizes[index] / 1024 / 1024)), safe_dir
                            )
                        )
                        # shutil.rmtree(safe_dir)

            log.info("---------------------------------------------------------------")
            log.info(
                "Image download and atmospheric correction for composite is complete."
            )
            log.info("---------------------------------------------------------------")

            if do_delete:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Deleting downloaded L1C images for composite, keeping only derived L2A products"
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                directory = composite_l1_image_dir
                log.info("Deleting {}".format(directory))
                shutil.rmtree(directory)
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Deletion of L1C images complete. Keeping only L2A images.")
                log.info(
                    "---------------------------------------------------------------"
                )
            else:
                if do_zip:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info(
                        "Zipping downloaded L1C images for composite after atmospheric correction"
                    )
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    zip_contents(composite_l1_image_dir)
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Zipping complete")
                    log.info(
                        "---------------------------------------------------------------"
                    )

            log.info("---------------------------------------------------------------")
            log.info(
                "Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files."
            )
            log.info("---------------------------------------------------------------")

            # Compare the time stamps of all .tif files in the composite_l2_masked_dir
            #   with the .SAFE directory time stamps in the composite_l2_dir, and
            #   the .zip files in the same directory
            #   Unzip files that have not been processed to cloud-masked yet

            directory = composite_l2_masked_image_dir
            masked_file_paths = [
                f
                for f in os.listdir(directory)
                if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
            ]

            directory = composite_l2_image_dir
            l2a_zip_file_paths = [
                f for f in os.listdir(directory) if f.endswith(".zip")
            ]

            if len(l2a_zip_file_paths) > 0:
                for f in l2a_zip_file_paths:
                    # check whether the zipped file has already been cloud masked
                    zip_timestamp = (
                        pyeo.filesystem_utilities.get_image_acquisition_time(
                            os.path.basename(f)
                        ).strftime("%Y%m%dT%H%M%S")
                    )
                    if any(zip_timestamp in f for f in masked_file_paths):
                        continue
                    else:
                        # extract it if not
                        unzip_contents(
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
                    safe_timestamp = (
                        pyeo.filesystem_utilities.get_image_acquisition_time(
                            os.path.basename(f)
                        ).strftime("%Y%m%dT%H%M%S")
                    )
                    if any(safe_timestamp in f for f in masked_file_paths):
                        continue
                    else:
                        # add it to the list of files to do if it has not been cloud masked yet
                        files_for_cloud_masking = files_for_cloud_masking + [f]

            if len(files_for_cloud_masking) == 0:
                log.info(
                    "No L2A images found for cloud masking. They may already have been done."
                )
            else:
                pyeo.raster_manipulation.apply_scl_cloud_mask(
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
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Offsetting cloud masked L2A images for composite.")
                log.info(
                    "---------------------------------------------------------------"
                )

                pyeo.raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
                    composite_l2_masked_image_dir, composite_l2_masked_image_dir
                )

                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Offsetting of cloud masked L2A images for composite complete."
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                # I.R. 20220607 END

                if do_quicklooks or do_all:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Producing quicklooks.")
                    log.info(
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
                        log.warning("No images found in {}.".format(main_dir))
                    else:
                        for f in files:
                            quicklook_path = os.path.join(
                                quicklook_dir,
                                os.path.basename(f).split(".")[0] + ".png",
                            )
                            log.info("Creating quicklook: {}".format(quicklook_path))
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

            if do_zip:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Zipping downloaded L2A images for composite after cloud masking and band stacking"
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                zip_contents(composite_l2_image_dir)
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Zipping complete")
                log.info(
                    "---------------------------------------------------------------"
                )

            log.info("---------------------------------------------------------------")
            log.info(
                "Building initial cloud-free median composite from directory {}".format(
                    composite_l2_masked_image_dir
                )
            )
            log.info("---------------------------------------------------------------")
            directory = composite_l2_masked_image_dir
            masked_file_paths = [
                f
                for f in os.listdir(directory)
                if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
            ]

            if len(masked_file_paths) > 0:
                pyeo.raster_manipulation.clever_composite_directory(
                    composite_l2_masked_image_dir,
                    composite_dir,
                    chunks=chunks,
                    generate_date_images=True,
                    missing_data_value=0,
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Baseline composite complete.")
                log.info(
                    "---------------------------------------------------------------"
                )

                if do_quicklooks or do_all:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Producing quicklooks.")
                    log.info(
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
                            log.warning("No images found in {}.".format(main_dir))
                        else:
                            for f in files:
                                quicklook_path = os.path.join(
                                    quicklook_dir,
                                    os.path.basename(f).split(".")[0] + ".png",
                                )
                                log.info(
                                    "Creating quicklook: {}".format(quicklook_path)
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

                if do_delete:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info(
                        "Deleting intermediate cloud-masked L2A images used for the baseline composite"
                    )
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    f = composite_l2_masked_image_dir
                    log.info("Deleting {}".format(f))
                    shutil.rmtree(f)
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Intermediate file products have been deleted.")
                    log.info("They can be reprocessed from the downloaded L2A images.")
                    log.info(
                        "---------------------------------------------------------------"
                    )
                else:
                    if do_zip:
                        log.info(
                            "---------------------------------------------------------------"
                        )
                        log.info(
                            "Zipping cloud-masked L2A images used for the baseline composite"
                        )
                        log.info(
                            "---------------------------------------------------------------"
                        )
                        zip_contents(composite_l2_masked_image_dir)
                        log.info(
                            "---------------------------------------------------------------"
                        )
                        log.info("Zipping complete")
                        log.info(
                            "---------------------------------------------------------------"
                        )

                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Compressing tiff files in directory {} and all subdirectories".format(
                        composite_dir
                    )
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                for root, dirs, files in os.walk(composite_dir):
                    all_tiffs = [
                        image_name
                        for image_name in files
                        if image_name.endswith(".tif")
                    ]
                    for this_tiff in all_tiffs:
                        pyeo.raster_manipulation.compress_tiff(
                            os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                        )

                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Baseline image composite, file compression, zipping and deletion of"
                )
                log.info("intermediate file products (if selected) are complete.")
                log.info(
                    "---------------------------------------------------------------"
                )

            else:
                log.error(
                    "No cloud-masked L2A image products found in {}.".format(
                        composite_l2_image_dir
                    )
                )
                log.error(
                    "Cannot produce a median composite. Download and cloud-mask some images first."
                )

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
            df_all = pd.DataFrame.from_dict(products_all, orient="index")

            # check granule sizes on the server
            df_all["size"] = (
                df_all["size"]
                .str.split(" ")
                .apply(lambda x: float(x[0]) * {"GB": 1e3, "MB": 1, "KB": 1e-3}[x[1]])
            )
            df = df_all.query("size >= " + str(faulty_granule_threshold))
            log.info(
                "Removed {} faulty scenes <{}MB in size from the list:".format(
                    len(df_all) - len(df), faulty_granule_threshold
                )
            )
            df_faulty = df_all.query("size < " + str(faulty_granule_threshold))
            for r in range(len(df_faulty)):
                log.info(
                    "   {} MB: {}".format(
                        df_faulty.iloc[r, :]["size"], df_faulty.iloc[r, :]["title"]
                    )
                )

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

            # TODO: Before the next step, search the composite/L2A and L1C directories whether the scenes have already been downloaded and/or processed and check their dir sizes
            # Remove those already obtained from the list

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
                        log.info(matching_l2a_products_df.iloc[0, :]["size"])
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
                    if do_dev:
                        add = pd.DataFrame(add)
                        log.info(
                            "Types for concatenation: {}, {}".format(
                                type(l2a_products), type(add)
                            )
                        )
                        l2a_products = pd.concat([l2a_products, add])
                        # TODO: test the above fix for:
                        # pyeo/pyeo/apps/change_detection/tile_based_change_detection_from_cover_maps.py:456: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
                    else:
                        l2a_products = l2a_products.append(add)

                log.info(
                    "    {} L1C products remaining for download".format(
                        l1c_products.shape[0]
                    )
                )
                l2a_products = l2a_products.drop_duplicates(subset="title")
                # I.R.
                log.info(
                    "    {} L2A products remaining for download".format(
                        l2a_products.shape[0]
                    )
                )
                if l1c_products.shape[0] > 0:
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

            # check for incomplete L2A downloads and remove them
            (
                incomplete_downloads,
                sizes,
            ) = pyeo.raster_manipulation.find_small_safe_dirs(
                l2_image_dir, threshold=faulty_granule_threshold * 1024 * 1024
            )
            if len(incomplete_downloads) > 0:
                for index, safe_dir in enumerate(incomplete_downloads):
                    if sizes[
                        index
                    ] / 1024 / 1024 < faulty_granule_threshold and os.path.exists(
                        safe_dir
                    ):
                        log.warning(
                            "Found likely incomplete download of size {} MB: {}".format(
                                str(round(sizes[index] / 1024 / 1024)), safe_dir
                            )
                        )
                        # shutil.rmtree(safe_dir)

            log.info("---------------------------------------------------------------")
            log.info(
                "Image download and atmospheric correction for change detection images is complete."
            )
            log.info("---------------------------------------------------------------")

            # TODO: delete L1C images if do_delete is True
            if do_delete:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Deleting L1C images downloaded for change detection.")
                log.info(
                    "Keeping only the derived L2A images after atmospheric correction."
                )
                log.info(
                    "---------------------------------------------------------------"
                )
                directory = l1_image_dir
                log.info("Deleting {}".format(directory))
                shutil.rmtree(directory)
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Deletion complete")
                log.info(
                    "---------------------------------------------------------------"
                )
            else:
                if do_zip:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Zipping L1C images downloaded for change detection")
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    zip_contents(l1_image_dir)
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Zipping complete")
                    log.info(
                        "---------------------------------------------------------------"
                    )

            log.info("---------------------------------------------------------------")
            log.info(
                "Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files."
            )
            log.info("---------------------------------------------------------------")
            # l2a_paths = [ f.path for f in os.scandir(l2_image_dir) if f.is_dir() ]
            # log.info("  l2_image_dir: {}".format(l2_image_dir))
            # log.info("  l2_masked_image_dir: {}".format(l2_masked_image_dir))
            # log.info("  bands: {}".format(bands))
            pyeo.raster_manipulation.apply_scl_cloud_mask(
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

            log.info("---------------------------------------------------------------")
            log.info("Cloud masking and band stacking of new L2A images are complete.")
            log.info("---------------------------------------------------------------")

            # I.R. 20220607 START
            # Apply offset to any images of processing baseline 0400 in the composite cloud_masked folder
            log.info("---------------------------------------------------------------")
            log.info("Offsetting cloud masked L2A images.")
            log.info("---------------------------------------------------------------")

            pyeo.raster_manipulation.apply_processing_baseline_offset_correction_to_tiff_file_directory(
                l2_masked_image_dir, l2_masked_image_dir
            )

            log.info("---------------------------------------------------------------")
            log.info("Offsetting of cloud masked L2A images complete.")
            log.info("---------------------------------------------------------------")
            # I.R. 20220607 END

            if do_quicklooks or do_all:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Producing quicklooks.")
                log.info(
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
                        log.warning("No images found in {}.".format(main_dir))
                    else:
                        for f in files:
                            quicklook_path = os.path.join(
                                quicklook_dir,
                                os.path.basename(f).split(".")[0] + ".png",
                            )
                            log.info("Creating quicklook: {}".format(quicklook_path))
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

            if do_zip:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Zipping L2A images downloaded for change detection")
                log.info(
                    "---------------------------------------------------------------"
                )
                zip_contents(l2_image_dir)
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Zipping complete")
                log.info(
                    "---------------------------------------------------------------"
                )

            log.info("---------------------------------------------------------------")
            log.info(
                "Compressing tiff files in directory {} and all subdirectories".format(
                    l2_masked_image_dir
                )
            )
            log.info("---------------------------------------------------------------")
            for root, dirs, files in os.walk(l2_masked_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
                for this_tiff in all_tiffs:
                    pyeo.raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                    )

            log.info("---------------------------------------------------------------")
            log.info(
                "Pre-processing of change detection images, file compression, zipping"
            )
            log.info(
                "and deletion of intermediate file products (if selected) are complete."
            )
            log.info("---------------------------------------------------------------")

        # ------------------------------------------------------------------------
        # Step 3: Classify each L2A image and the baseline composite
        # ------------------------------------------------------------------------
        if do_all or do_classify:
            log.info("---------------------------------------------------------------")
            log.info(
                "Classify a land cover map for each L2A image and composite image using a saved model"
            )
            log.info("---------------------------------------------------------------")
            log.info("Model used: {}".format(model_path))
            if skip_existing:
                log.info("Skipping existing classification images if found.")
            pyeo.classification.classify_directory(
                composite_dir,
                model_path,
                categorised_image_dir,
                prob_out_dir=None,
                apply_mask=False,
                out_type="GTiff",
                chunks=chunks,
                skip_existing=skip_existing,
            )
            pyeo.classification.classify_directory(
                l2_masked_image_dir,
                model_path,
                categorised_image_dir,
                prob_out_dir=None,
                apply_mask=False,
                out_type="GTiff",
                chunks=chunks,
                skip_existing=skip_existing,
            )

            log.info("---------------------------------------------------------------")
            log.info(
                "Compressing tiff files in directory {} and all subdirectories".format(
                    categorised_image_dir
                )
            )
            log.info("---------------------------------------------------------------")
            for root, dirs, files in os.walk(categorised_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
                for this_tiff in all_tiffs:
                    pyeo.raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                    )

            log.info("---------------------------------------------------------------")
            log.info("Classification of all images is complete.")
            log.info("---------------------------------------------------------------")

            if do_quicklooks or do_all:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Producing quicklooks.")
                log.info(
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
                        log.warning("No images found in {}.".format(main_dir))
                    else:
                        for f in files:
                            quicklook_path = os.path.join(
                                quicklook_dir,
                                os.path.basename(f).split(".")[0] + ".png",
                            )
                            log.info("Creating quicklook: {}".format(quicklook_path))
                            pyeo.raster_manipulation.create_quicklook(
                                f, quicklook_path, width=512, height=512, format="PNG"
                            )
            log.info("Quicklooks complete.")

        # ------------------------------------------------------------------------
        # Step 4: Pair up the class images with the composite baseline map
        # and identify all pixels with the change between groups of classes of interest.
        # Optionally applies a sieve filter to the class images if specified in the ini file.
        # Confirms detected changes by NDVI differencing.
        # ------------------------------------------------------------------------

        if do_all or do_change:
            log.info("---------------------------------------------------------------")
            log.info("Creating change layers from stacked class images.")
            log.info("---------------------------------------------------------------")
            log.info("Changes of interest:")
            log.info("  from any of the classes {}".format(from_classes))
            log.info("  to   any of the classes {}".format(to_classes))

            # optionally sieve the class images
            if sieve > 0:
                log.info("Applying sieve to classification outputs.")
                sieved_paths = pyeo.raster_manipulation.sieve_directory(
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
                if f.is_file()
                and f.name.endswith(".tif")
                and not "composite_" in f.name
            ]
            if len(class_image_paths) == 0:
                raise FileNotFoundError(
                    "No class images found in {}.".format(class_image_dir)
                )

            # sort class images by image acquisition date
            class_image_paths = list(
                filter(
                    pyeo.filesystem_utilities.get_image_acquisition_time,
                    class_image_paths,
                )
            )
            class_image_paths.sort(
                key=lambda x: pyeo.filesystem_utilities.get_image_acquisition_time(x)
            )
            for index, image in enumerate(class_image_paths):
                log.info("{}: {}".format(index, image))

            # find the latest available composite
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

            log.info(
                "Most recent class composite at {}".format(latest_class_composite_path)
            )
            if not os.path.exists(latest_class_composite_path):
                log.critical(
                    "Latest class composite not found. The first time you run this script, you need to include the "
                    "--build-composite flag to create a base composite to work off. If you have already done this,"
                    "check that the earliest dated image in your images/merged folder is later than the earliest"
                    " dated image in your composite/ folder. Then, you need to run the --classify option."
                )
                sys.exit(1)

            if do_dev:  # set the name of the report file in the development version run
                before_timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(latest_class_composite_path)
                    )[0]
                )
                # I.R. 20220611 START
                ## Timestamp report with the date of most recent classified image that contributes to it
                after_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        os.path.basename(class_image_paths[-1])
                    )
                )
                ## ORIGINAL
                # gets timestamp of the earliest change image of those available in class_image_path
                # after_timestamp  = pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(class_image_paths[0]))
                # I.R. 20220611 END
                output_product = os.path.join(
                    probability_image_dir,
                    "report_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                log.info("I.R. Report file name will be {}".format(output_product))

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
                    log.info(
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
                    # log.info("Report timestamp {}".format(report_timestamp.strftime("%Y%m%dT%H%M%S")))
                    # log.info(" is earlier than {}".format(after_timestamp.strftime("%Y%m%dT%H%M%S")))
                    # log.info("Updating its file name to: {}".format(output_product))
                    # os.rename(output_product_existing, output_product)

                    # Renaming any pre-existing report file with prefix 'archive_'
                    ## it will therefore not be detected in __change_from_class_maps which will therefore create a new report file

                    output_product_existing_archived = os.path.join(
                        os.path.dirname(output_product_existing),
                        "archived_" + os.path.basename(output_product_existing),
                    )
                    log.info(
                        "Renaming existing report image product to: {}".format(
                            output_product_existing_archived
                        )
                    )
                    os.rename(output_product_existing, output_product_existing_archived)

                    # I.R. 20220610 END

            # find change patterns in the stack of classification images
            for index, image in enumerate(class_image_paths):
                before_timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(latest_class_composite_path)
                    )[0]
                )
                after_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        os.path.basename(image)
                    )
                )
                # I.R. 20220612 START
                log.info(
                    "*** PROCESSING CLASSIFIED IMAGE: {} of {} filename: {} ***".format(
                        index, len(class_image_paths), image
                    )
                )
                # I.R. 20220612 END
                log.info("  early time stamp: {}".format(before_timestamp))
                log.info("  late  time stamp: {}".format(after_timestamp))
                change_raster = os.path.join(
                    probability_image_dir,
                    "change_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                log.info("  Change raster file to be created: {}".format(change_raster))

                dNDVI_raster = os.path.join(
                    probability_image_dir,
                    "dNDVI_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                log.info(
                    "  I.R. dNDVI raster file to be created: {}".format(dNDVI_raster)
                )

                NDVI_raster = os.path.join(
                    probability_image_dir,
                    "NDVI_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                log.info(
                    "  I.R. NDVI raster file of change image to be created: {}".format(
                        NDVI_raster
                    )
                )

                if do_dev:
                    # This function looks for changes from class 'change_from' in the composite to any of the 'change_to_classes'
                    # in the change images. Pixel values are the acquisition date of the detected change of interest or zero.
                    # TODO: In change_from_class_maps(), add a flag (e.g. -1) whether a pixel was a cloud in the later image.
                    # Applying check whether dNDVI < -0.2, i.e. greenness has decreased over changed areas

                    log.info(
                        "Update of the report image product based on change detection image."
                    )
                    pyeo.raster_manipulation.__change_from_class_maps(
                        latest_class_composite_path,
                        image,
                        change_raster,
                        dNDVI_raster,
                        NDVI_raster,
                        change_from=from_classes,
                        change_to=to_classes,
                        report_path=output_product,
                        skip_existing=skip_existing,
                        old_image_dir=composite_dir,
                        new_image_dir=l2_masked_image_dir,
                        viband1=4,
                        viband2=3,
                        dNDVI_threshold=-0.2,
                    )
                else:
                    pyeo.raster_manipulation.change_from_class_maps(
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

            log.info("---------------------------------------------------------------")
            log.info("Post-classification change detection complete.")
            log.info("---------------------------------------------------------------")

            log.info("---------------------------------------------------------------")
            log.info(
                "Compressing tiff files in directory {} and all subdirectories".format(
                    probability_image_dir
                )
            )
            log.info("---------------------------------------------------------------")
            for root, dirs, files in os.walk(probability_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
                for this_tiff in all_tiffs:
                    pyeo.raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                    )

            log.info("---------------------------------------------------------------")
            log.info(
                "Compressing tiff files in directory {} and all subdirectories".format(
                    sieved_image_dir
                )
            )
            log.info("---------------------------------------------------------------")
            for root, dirs, files in os.walk(sieved_image_dir):
                all_tiffs = [
                    image_name for image_name in files if image_name.endswith(".tif")
                ]
                for this_tiff in all_tiffs:
                    pyeo.raster_manipulation.compress_tiff(
                        os.path.join(root, this_tiff), os.path.join(root, this_tiff)
                    )

            if not do_dev:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info(
                    "Creating aggregated report file. Deprecated in the development version."
                )
                log.info(
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

                before_timestamp = (
                    pyeo.filesystem_utilities.get_change_detection_dates(
                        os.path.basename(latest_class_composite_path)
                    )[0]
                )
                after_timestamp = (
                    pyeo.filesystem_utilities.get_image_acquisition_time(
                        os.path.basename(class_image_paths[-1])
                    )
                )
                output_product = os.path.join(
                    probability_image_dir,
                    "report_{}_{}_{}.tif".format(
                        before_timestamp.strftime("%Y%m%dT%H%M%S"),
                        tile_id,
                        after_timestamp.strftime("%Y%m%dT%H%M%S"),
                    ),
                )
                log.info("Combining date maps: {}".format(date_image_paths))
                pyeo.raster_manipulation.combine_date_maps(
                    date_image_paths, output_product
                )

            log.info("---------------------------------------------------------------")
            log.info(
                "Report image product completed / updated: {}".format(output_product)
            )
            log.info("Compressing the report image.")
            log.info("---------------------------------------------------------------")
            pyeo.raster_manipulation.compress_tiff(output_product, output_product)

            if do_delete:
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Deleting intermediate class images used in change detection.")
                log.info(
                    "They can be recreated from the cloud-masked, band-stacked L2A images and the saved model."
                )
                log.info(
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
                        if not f.startswith("composite_") and not f.startswith(
                            "report_"
                        ):
                            log.info("Deleting {}".format(os.path.join(directory, f)))
                            if os.path.isdir(os.path.join(directory, f)):
                                shutil.rmtree(os.path.join(directory, f))
                            else:
                                os.remove(os.path.join(directory, f))
                log.info(
                    "---------------------------------------------------------------"
                )
                log.info("Deletion of intermediate file products complete.")
                log.info(
                    "---------------------------------------------------------------"
                )
            else:
                if do_zip:
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info(
                        "Zipping intermediate class images used in change detection"
                    )
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    directories = [categorised_image_dir, sieved_image_dir]
                    for directory in directories:
                        zip_contents(directory, notstartswith=["composite_", "report_"])
                    log.info(
                        "---------------------------------------------------------------"
                    )
                    log.info("Zipping complete")
                    log.info(
                        "---------------------------------------------------------------"
                    )

            log.info("---------------------------------------------------------------")
            log.info(
                "Change detection and report image product updating, file compression, zipping"
            )
            log.info(
                "and deletion of intermediate file products (if selected) are complete."
            )
            log.info("---------------------------------------------------------------")

        if do_delete:
            log.info("---------------------------------------------------------------")
            log.info("Deleting temporary directories starting with 'tmp*'")
            log.info("These can be left over from interrupted processing runs.")
            log.info("---------------------------------------------------------------")
            directory = tile_root_dir
            for root, dirs, files in os.walk(directory):
                temp_dirs = [d for d in dirs if d.startswith("tmp")]
                for temp_dir in temp_dirs:
                    log.info("Deleting {}".format(os.path.join(root, temp_dir)))
                    if os.path.isdir(os.path.join(directory, f)):
                        shutil.rmtree(os.path.join(directory, f))
                    else:
                        log.warning(
                            "This should not have happened. {} is not a directory. Skipping deletion.".format(
                                os.path.join(root, temp_dir)
                            )
                        )
            log.info("---------------------------------------------------------------")
            log.info("Deletion of temporary directories complete.")
            log.info("---------------------------------------------------------------")

        # ------------------------------------------------------------------------
        # Step 5: Update the baseline composite with the reflectance values of only the changed pixels.
        #         Update last_date of the baseline composite.
        # ------------------------------------------------------------------------

        if do_update or do_all:
            log.warning(
                "---------------------------------------------------------------"
            )
            log.warning(
                "Updating of the baseline composite with new imagery is deprecated and will be ignored."
            )
            log.warning(
                "---------------------------------------------------------------"
            )
            """
            log.info("---------------------------------------------------------------")
            log.info("Updating baseline composite with new imagery.")
            log.info("---------------------------------------------------------------")
            # get all composite file paths
            composite_paths = [ f.path for f in os.scandir(composite_dir) if f.is_file() ]
            if len(composite_paths) == 0:
                raise FileNotFoundError("No composite images found in {}.".format(composite_dir))
            log.info("Sorting composite image list by time stamp.")
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
                log.info("Most recent composite at {}".format(latest_composite_path))
            except IndexError:
                log.critical("Latest composite not found. The first time you run this script, you need to include the "
                             "--build-composite flag to create a base composite to work off. If you have already done this,"
                             "check that the earliest dated image in your images/merged folder is later than the earliest"
                             " dated image in your composite/ folder.")
                sys.exit(1)

            # Find all categorised images
            categorised_paths = [ f.path for f in os.scandir(categorised_image_dir) if f.is_file() ]
            if len(categorised_paths) == 0:
                raise FileNotFoundError("No categorised images found in {}.".format(categorised_image_dir))
            log.info("Sorting categorised image list by time stamp.")
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
            log.info("Sorting masked L2A image list by time stamp.")
            l2a_images = \
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(l2_masked_image_dir) if image_name.endswith(".tif")],
                    recent_first=False
                )

            log.info("Updating most recent composite with new imagery over detected changed areas.")
            for categorised_image in categorised_images:
                # Find corresponding L2A file
                timestamp = pyeo.filesystem_utilities.get_change_detection_date_strings(os.path.basename(categorised_image))
                before_time = timestamp[0]
                after_time = timestamp[1]
                granule = pyeo.filesystem_utilities.get_sen_2_image_tile(os.path.basename(categorised_image))
                l2a_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.tif".format(after_time, granule)
                log.info("Searching for image name pattern: {}".format(l2a_glob))
                l2a_image = glob.glob(os.path.join(l2_masked_image_dir, l2a_glob))
                if len(l2a_image) == 0:
                    log.warning("Matching L2A file not found for categorised image {}".format(categorised_image))
                else:
                    l2a_image = l2a_image[0]
                log.info("Categorised image: {}".format(categorised_image))
                log.info("Matching stacked masked L2A file: {}".format(l2a_image))

                # Extract all reflectance values from the pixels with the class of interest in the classified image
                with TemporaryDirectory(dir=os.getcwd()) as td:
                    log.info("Creating mask file from categorised image {} for class: {}".format(os.path.join(categorised_image_dir, categorised_image), class_of_interest))
                    mask_path = os.path.join(td, categorised_image.split(sep=".")[0]+".msk")
                    log.info("  at {}".format(mask_path))
                    pyeo.raster_manipulation.create_mask_from_class_map(os.path.join(categorised_image_dir, categorised_image),
                                                                        mask_path, [class_of_interest], buffer_size=0, out_resolution=None)
                    masked_image_path = os.path.join(td, categorised_image.split(sep=".")[0]+"_change.tif")
                    pyeo.raster_manipulation.apply_mask_to_image(mask_path, l2a_image, masked_image_path)
                    new_composite_path = os.path.join(composite_dir, "composite_{}.tif".format(
                                                      pyeo.filesystem_utilities.get_sen_2_image_timestamp(os.path.basename(l2a_image))))
                    # Update pixel values in the composite over the selected pixel locations where values are not missing
                    log.info("  {}".format(latest_composite_path))
                    log.info("  {}".format([l2a_image]))
                    log.info("  {}".format(new_composite_path))
                    #TODO generate_date_image=True currently produces a type error
                    pyeo.raster_manipulation.update_composite_with_images(
                                                                         latest_composite_path,
                                                                         [masked_image_path],
                                                                         new_composite_path,
                                                                         generate_date_image=False,
                                                                         missing=0
                                                                         )
                latest_composite_path = new_composite_path
            """

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
        dest="chunks",
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
        "--dev",
        dest="do_dev",
        action="store_true",
        default=False,
        help="If selected, run the development version with experimental functions. Default is to run in production mode.",
    )
    parser.add_argument(
        "-x",
        "--change",
        dest="do_change",
        action="store_true",
        default=False,
        help="Produces change images by post-classification cross-tabulation.",
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
        help="Deprecated. Builds a new cloud-free composite in composite/ from the latest image and mask"
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
        help="Not currently available. If present, removes all intermediate images to save space.",
    )
    parser.add_argument(
        "-z",
        "--zip",
        dest="do_zip",
        action="store_true",
        default=False,
        help="If present, archives all intermediate images to save space.",
    )
    parser.add_argument(
        "-s",
        "--skip",
        dest="skip_existing",
        action="store_true",
        default=False,
        help="If chosen, existing output files will be skipped in the production process.",
    )

    args = parser.parse_args()

    rolling_detection(**vars(args))
