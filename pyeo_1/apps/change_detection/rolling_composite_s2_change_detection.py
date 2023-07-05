"""
rolling_composite_s2_change_detection
-------------------------------------
An app for providing continuous change detection. Runs the following algorithm

 Step 1: Create initial cloud-free composite with pixel values taken from the last date

 Step 2: Download images for the specific time window

 Step 3: Preprocess each image with Sen2Cor to make a cloud mask and atmospherically correct it

 Step 4: For each image, merge the separate band raster files into a single Geotiff file

 Step 5: Apply an external mask of pixels to be classified, e.g. a baseline map of land cover
 
 Step 6: For pairs of consecutive images, stack the bands of both images into a single Geotiff file

 Step 7: Mosaic the pairwise image stacks if multiple tiles are processed
 
 Step 8: Classify the mosaics using a saved model

 Step 9: Update the rolling cloud-free composite with the latest available cloud-free pixels

 Step 10: Update last_date of composite

 """
import shutil
import sys

import pyeo.classification
import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities


import configparser
import copy
import argparse
import os
from osgeo import gdal
import datetime as dt


def rolling_detection(
    config_path,
    arg_start_date=None,
    arg_end_date=None,
    build_composite=False,
    tile_id=None,
    num_chunks=None,
    download_source="scihub",
    flip_stacks=False,
    download_l2_data=True,
    build_prob_image=False,
    do_download=False,
    do_preprocess=False,
    do_merge=False,
    do_mask=False,
    do_stack=False,
    do_mosaic=False,
    do_classify=False,
    do_update=False,
    do_delete=False,
    skip_prob_image=False,
):
    # If any processing step args are present, do not assume that we want to do all steps
    do_all = True
    if (
        build_composite
        or do_download
        or do_preprocess
        or do_merge
        or do_stack
        or do_mosaic
        or do_mask
        or do_classify
    ) == True:
        do_all = False
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(config_path)
    sen_user = conf["sent_2"]["user"]
    sen_pass = conf["sent_2"]["pass"]
    project_root = conf["forest_sentinel"]["root_dir"]
    aoi_path = conf["forest_sentinel"]["aoi_path"]
    start_date = conf["forest_sentinel"]["start_date"]
    end_date = conf["forest_sentinel"]["end_date"]
    log_path = conf["forest_sentinel"]["log_path"]
    cloud_cover = conf["forest_sentinel"]["cloud_cover"]
    cloud_certainty_threshold = int(
        conf["forest_sentinel"]["cloud_certainty_threshold"]
    )
    model_path = conf["forest_sentinel"]["model"]
    mask_path = conf["forest_sentinel"]["mask_path"]
    sen2cor_path = conf["sen2cor"]["path"]
    composite_start_date = conf["forest_sentinel"]["composite_start"]
    composite_end_date = conf["forest_sentinel"]["composite_end"]
    epsg = int(conf["forest_sentinel"]["epsg"])

    pyeo.filesystem_utilities.create_file_structure(project_root)
    log = pyeo.filesystem_utilities.init_log(log_path)
    log.info("Creating the directory structure if not already present")

    try:
        l1_image_dir = os.path.join(project_root, r"images/L1C")
        l2_image_dir = os.path.join(project_root, r"images/L2A")
        planet_image_dir = os.path.join(project_root, r"images/planet")
        merged_image_dir = os.path.join(project_root, r"images/bandmerged")
        stacked_image_dir = os.path.join(project_root, r"images/stacked")
        mosaic_image_dir = os.path.join(project_root, r"images/stacked_mosaic")
        masked_stacked_image_dir = os.path.join(project_root, r"images/stacked_masked")
        catagorised_image_dir = os.path.join(project_root, r"output/classified")
        probability_image_dir = os.path.join(project_root, r"output/probabilities")
        composite_dir = os.path.join(project_root, r"composite")
        composite_l1_image_dir = os.path.join(project_root, r"composite/L1C")
        composite_l2_image_dir = os.path.join(project_root, r"composite/L2A")
        composite_merged_dir = os.path.join(project_root, r"composite/bandmerged")

        if skip_prob_image:
            probability_image_dir = None

        if arg_start_date == "LATEST":
            # This isn't nice, but returns the yyyymmdd string of the latest classified image
            start_date = pyeo.filesystem_utilities.get_image_acquisition_time(
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(catagorised_image_dir)
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

        # Download and build the initial composite.
        if build_composite:
            log.info("------------------------------------------")
            log.info("Building image composite")
            if do_download or download_l2_data or do_all:
                log.info("Searching for images for initial composite.")
                composite_products_all = (
                    pyeo.queries_and_downloads.check_for_s2_data_by_date(
                        aoi_path,
                        composite_start_date,
                        composite_end_date,
                        conf,
                        cloud_cover=cloud_cover,
                        tile_id=tile_id,
                    )
                )
                log.info(
                    "--> Found {} L1C or L2A products for the composite.".format(
                        len(composite_products_all)
                    )
                )

                if download_l2_data:
                    log.info(
                        "Restricting query results to include only matching L1C and L2A products."
                    )
                    composite_products = (
                        pyeo.queries_and_downloads.filter_non_matching_s2_data(
                            composite_products_all
                        )
                    )
                    log.info(
                        "--> Found {} L2A products for the composite.".format(
                            len(composite_products)
                        )
                    )
                    if len(composite_products) > 0:
                        log.info(
                            "{} matching products with both L1C and L2A remain.".format(
                                len(composite_products)
                            )
                        )
                        log.info(
                            "Downloading only matching Sentinel-2 L1C and L2A products."
                        )
                        pyeo.queries_and_downloads.download_s2_data(
                            composite_products,
                            composite_l1_image_dir,
                            composite_l2_image_dir,
                            download_source,
                            user=sen_user,
                            passwd=sen_pass,
                            try_scihub_on_fail=True,
                        )
                    else:  # download L1C if no L2A data are found
                        do_download = True
                        download_l2_data = False
                        composite_products = copy.deepcopy(composite_products_all)
                if (do_download or do_all) and not download_l2_data:
                    log.info("Searching for L1C products only.")
                    composite_products = pyeo.queries_and_downloads.filter_to_l1_data(
                        composite_products
                    )
                    log.info(
                        "--> Found {} L1C products for the composite.".format(
                            len(composite_products)
                        )
                    )
                    if len(composite_products) > 0:
                        log.error("Found no L1C data that match the search criteria.")
                    log.info("Downloading L1C products for the composite.")
                    pyeo.queries_and_downloads.download_s2_data(
                        composite_products,
                        composite_l1_image_dir,
                        composite_l2_image_dir,
                        source=download_source,
                        user=sen_user,
                        passwd=sen_pass,
                        try_scihub_on_fail=True,
                    )
                    log.info("Atmospheric correction with sen2cor for L1C products")
                    pyeo.raster_manipulation.atmospheric_correction(
                        composite_l1_image_dir,
                        composite_l2_image_dir,
                        sen2cor_path,
                        delete_unprocessed_image=False,
                    )
                log.info("Merging raster bands into single files for each image")
                pyeo.raster_manipulation.preprocess_sen2_images(
                    composite_l2_image_dir,
                    composite_merged_dir,
                    composite_l1_image_dir,
                    cloud_certainty_threshold,
                    epsg=epsg,
                    buffer_size=10,
                )

            log.info(
                "Building initial cloud-free composite from directory {}".format(
                    composite_dir
                )
            )
            pyeo.raster_manipulation.clever_composite_directory(
                composite_merged_dir,
                composite_dir,
                chunks=30,
                generate_date_images=True,
                missing_data_value=0,
            )

        else:
            # If build_composite is not set, query and download all images since the last composite was created or updated
            if do_download or download_l2_data or do_all:
                log.info(
                    "Searching for images for change detection between {} and {} with cloud cover <= {}".format(
                        start_date, end_date, cloud_cover
                    )
                )
                products = pyeo.queries_and_downloads.check_for_s2_data_by_date(
                    aoi_path,
                    start_date,
                    end_date,
                    conf,
                    cloud_cover=cloud_cover,
                    tile_id=tile_id,
                )
                if download_l2_data:
                    log.info(
                        "Restricting query results to include only matching L1C and L2A products."
                    )
                    products = pyeo.queries_and_downloads.filter_non_matching_s2_data(
                        products
                    )
                    log.info("{} L2A products remain".format(len(products)))
                    log.info("Downloading selected products.")
                    pyeo.queries_and_downloads.download_s2_data(
                        products,
                        l1_image_dir,
                        l2_image_dir,
                        download_source,
                        user=sen_user,
                        passwd=sen_pass,
                        try_scihub_on_fail=True,
                    )
                else:
                    log.info("Restricting query results to L1C products only.")
                    products = pyeo.queries_and_downloads.filter_to_l1_data(products)
                    log.info("Downloading selected products.")
                    pyeo.queries_and_downloads.download_s2_data(
                        products,
                        l1_image_dir,
                        l2_image_dir,
                        download_source,
                        user=sen_user,
                        passwd=sen_pass,
                        try_scihub_on_fail=True,
                    )
                    log.info("Applying sen2cor to downloaded L1C products.")
                    pyeo.raster_manipulation.atmospheric_correction(
                        l1_image_dir,
                        l2_image_dir,
                        sen2cor_path,
                        delete_unprocessed_image=False,
                    )

            # Aggregate single band raster files for the change detection into a single Geotiff files
            if do_merge or do_all:
                log.info("Merging all band files into a Geotiff file for each granule")
                pyeo.raster_manipulation.preprocess_sen2_images(
                    l2_image_dir,
                    merged_image_dir,
                    l1_image_dir,
                    cloud_certainty_threshold,
                    epsg=epsg,
                    buffer_size=10,
                )

            # Stack pairs of consecutive images for change detection into single files
            if do_stack or do_all:
                log.info("Stacking pairs of consecutive images into single files")
                log.info("Finding most recent image composite")
                try:
                    latest_composite_name = (
                        pyeo.filesystem_utilities.sort_by_timestamp(
                            [
                                image_name
                                for image_name in os.listdir(composite_dir)
                                if image_name.endswith(".tif")
                            ],
                            recent_first=True,
                        )[0]
                    )
                    latest_composite_path = os.path.join(
                        composite_dir, latest_composite_name
                    )
                    log.info(
                        "Most recent composite at {}".format(latest_composite_path)
                    )
                except IndexError:
                    log.critical(
                        "Latest composite not found. The first time you run this script, you need to include the "
                        "--build-composite flag to create a base composite to work off. If you have already done this,"
                        "check that the earliest dated image in your images/merged folder is later than the earliest"
                        " dated image in your composite/ folder."
                    )
                    sys.exit(1)

                log.info("Sorting image list")
                images = pyeo.filesystem_utilities.sort_by_timestamp(
                    [
                        image_name
                        for image_name in os.listdir(merged_image_dir)
                        if image_name.endswith(".tif")
                    ],
                    recent_first=False,
                )
                if not images:
                    raise FileNotFoundError(
                        "No images found in {}. Did your preprocessing complete?".format(
                            merged_image_dir
                        )
                    )
                log.info("Images to process: {}".format(images))

                for image in images:
                    new_image_path = os.path.join(merged_image_dir, image)
                    # Stack with preceding composite
                    try:
                        latest_composite_path = (
                            pyeo.filesystem_utilities.get_preceding_image_path(
                                new_image_path, composite_dir
                            )
                        )
                    except FileNotFoundError:
                        log.warning(
                            "No preceding composite found for {}, skipping.".format(
                                new_image_path
                            )
                        )
                        continue
                    log.info(
                        "Stacking image {} with latest available composite {} with bands from both dates".format(
                            new_image_path, latest_composite_path
                        )
                    )
                    log.info(
                        "New stacked image will be created at {}".format(new_image_path)
                    )
                    new_stack_path = (
                        pyeo.raster_manipulation.stack_image_with_composite(
                            new_image_path,
                            latest_composite_path,
                            stacked_image_dir,
                            invert_stack=flip_stacks,
                        )
                    )

        """
        # Mosaic stacked layers
        if do_mosaic or do_all:
            log.info("Mosaicking stacked multitemporal images across tiles")
            pyeo.raster_manipulation.mosaic_images(stacked_image_dir, mosaic_image_dir, format="GTiff", 
                                                   datatype=gdal.GDT_Int32, nodata=0)
        """

        # Classify images stacked with composite
        if do_classify or do_all:
            log.info("do_all={}".format(do_all))
            log.info("do_mask={}".format(do_mask))
            if do_mask or do_all:
                # Apply a mask of pixels to be classified to all images in the directory
                log.info("Applying the specified mask of pixels to be classified")
                log.info("Stacked image dir: {}".format(stacked_image_dir))
                log.info("Mask file: {}".format(mask_path))
                log.info("Masked image output dir: {}".format(masked_stacked_image_dir))
                pyeo.raster_manipulation.apply_mask_to_dir(
                    mask_path, stacked_image_dir, masked_stacked_image_dir
                )
                log.info(
                    "Copying corresponding cloud masks from: {}".format(
                        stacked_image_dir
                    )
                )
                log.info("  to: {}".format(masked_stacked_image_dir))
                cloudmask_files = [
                    os.path.join(stacked_image_dir, f)
                    for f in os.listdir(stacked_image_dir)
                    if f.endswith(".msk")
                ]
                for cloudmask_file in cloudmask_files:
                    log.info(
                        "  Copying {} to {}".format(
                            cloudmask_file,
                            os.path.join(
                                masked_stacked_image_dir,
                                os.path.basename(cloudmask_file).split(".")[0]
                                + "_masked.msk",
                            ),
                        )
                    )
                    shutil.copy(
                        cloudmask_file,
                        os.path.join(
                            masked_stacked_image_dir,
                            os.path.basename(cloudmask_file).split(".")[0]
                            + "_masked.msk",
                        ),
                    )

                log.info("Classifying masked image stacked with composite")
                masked_stacked_image_files = [
                    os.path.join(masked_stacked_image_dir, f)
                    for f in os.listdir(masked_stacked_image_dir)
                    if f.endswith(".tif") or f.endswith(".tiff")
                ]
                for masked_stacked_image_file in masked_stacked_image_files:
                    new_class_image = os.path.join(
                        catagorised_image_dir,
                        "class_{}".format(os.path.basename(masked_stacked_image_file)),
                    )
                    if build_prob_image:
                        new_prob_image = os.path.join(
                            probability_image_dir,
                            "prob_{}".format(
                                os.path.basename(masked_stacked_image_file)
                            ),
                        )
                    else:
                        new_prob_image = None
                    pyeo.classification.classify_image(
                        masked_stacked_image_file,
                        model_path,
                        new_class_image,
                        new_prob_image,
                        num_chunks=num_chunks,
                        skip_existing=True,
                        apply_mask=False,
                    )

            else:
                log.info("Classifying image stacked with composite")
                stacked_image_files = [
                    os.path.join(stacked_image_dir, f)
                    for f in os.listdir(stacked_image_dir)
                    if f.endswith(".tif") or f.endswith(".tiff")
                ]
                for stacked_image_file in stacked_image_files:
                    new_class_image = os.path.join(
                        catagorised_image_dir,
                        "class_{}".format(os.path.basename(stacked_image_file)),
                    )
                    if build_prob_image:
                        new_prob_image = os.path.join(
                            probability_image_dir,
                            "prob_{}".format(os.path.basename(stacked_image_file)),
                        )
                    else:
                        new_prob_image = None
                    pyeo.classification.classify_image(
                        stacked_image_file,
                        model_path,
                        new_class_image,
                        new_prob_image,
                        num_chunks=num_chunks,
                        skip_existing=True,
                        apply_mask=True,
                    )

        # Build new composite
        if do_update or do_all:
            log.info("Updating composite")
            new_composite_path = os.path.join(
                composite_dir,
                "composite_{}.tif".format(
                    pyeo.filesystem_utilities.get_sen_2_image_timestamp(
                        os.path.basename(image)
                    )
                ),
            )
            pyeo.raster_manipulation.composite_images_with_mask(
                (latest_composite_path, new_image_path),
                new_composite_path,
                generate_date_image=True,
            )
            latest_composite_path = new_composite_path

        log.info("***PROCESSING END***")

    except Exception:
        log.exception("Fatal error in rolling_s2_composite chain")


if __name__ == "__main__":
    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Downloads, preprocesses and classifies sentinel 2 images. A directory"
        "structure to contain preprocessed and downloaded files will be"
        "created at the aoi_root location specified in the config file."
        "If any of the step flags (d,p,m,a,s,c,u,r) are present, only those "
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
        "--download_source",
        default="scihub",
        help="Sets the download source, can be scihub " "(default) or aws",
    )
    parser.add_argument(
        "--flip_stacks",
        action="store_true",
        default=False,
        help="If present, stasks the classification stack as new(bgr), old(bgr). Default is"
        "old(bgr), new(bgr). For compatability with old models.",
    )
    parser.add_argument(
        "--download_l2_data",
        action="store_true",
        default=False,
        help="If present, skips sen2cor and instead downloads every image in the query for which"
        "both an L1C and L2A product are available",
    )
    parser.add_argument(
        "--build_prob_image",
        action="store_true",
        default=False,
        help="If present, build a confidence map of pixels. These tend to be large.",
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
        "-p",
        "--preprocess",
        dest="do_preprocess",
        action="store_true",
        default=False,
        help="Currently does nothing, because atmospheric correction is applied automatically after "
        "download of L1C products.",
    )
    #                   'If present, apply sen2cor to all .SAFE in images/L1. Stores the result in images/L2')
    parser.add_argument(
        "-m",
        "--merge",
        dest="do_merge",
        action="store_true",
        default=False,
        help="If present, merges the blue, green, red and NIR 10m rasters in each L2 safefile"
        " into a single 4-band raster. This will also mask and reproject"
        " the image to the requested projection. Stores the result in images/merged.",
    )
    parser.add_argument(
        "-a",
        "--mask",
        dest="do_mask",
        action="store_true",
        default=False,
        help="Applies an external mask file.",
    )
    parser.add_argument(
        "-s",
        "--stack",
        dest="do_stack",
        action="store_true",
        default=False,
        help="For each image in images/merged, stacks with the composite in composite/ that is most "
        "recent prior to the image. Stores an 8-band geotiff in images/stacked, where bands 1-4 "
        "are the B,G,R and NIR bands of the composite and band 5-8 are the B,G,R and NIR bands of"
        "the merged image.",
    )
    parser.add_argument(
        "-o",
        "--mosaic",
        dest="do_mosaic",
        action="store_true",
        default=False,
        help="All image stacks in images/stacked will be mosaicked into a single large raster file."
        "Warning: File size can become very large if multiple tiles are used.",
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
        "-u",
        "--update",
        dest="do_update",
        action="store_true",
        default=False,
        help="Builds a new cloud-free composite in composite/ from the latest image and mask"
        " in images/merged",
    )
    parser.add_argument(
        "-r",
        "--remove",
        dest="do_delete",
        action="store_true",
        default=False,
        help="Not implemented. If present, removes all images in images/ to save space.",
    )

    parser.add_argument(
        "--skip_prob_image",
        dest="skip_prob_image",
        action="store_true",
        default=False,
        help="",
    )

    args = parser.parse_args()

    rolling_detection(**vars(args))
