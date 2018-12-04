"""
 Right, let's think this through.
 Step -1: Create initial composite with last date (stored in filename?)
 Step 0: Load composite
 Step 1: Download images from last date in composite until present last date
 Step 2: Preprocess each image
 Step 3: Generate cloud mask for each image
For each preprocessed image:
    Step 4: Build stack with composite
    Step 5: Classify stack
    Step 6: Update composite with last cloud-free pixel based on cloud mask
    Step 7: Update last_date of composite
Step 8:
 """

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import configparser
import argparse
import os
import datetime as dt


if __name__ == "__main__":

    do_all = True

    # Reading in config file
    parser = argparse.ArgumentParser(description='Automatically detect and report on change')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'change_detection.ini',
                        help="Path to the .ini file specifying the job.")
    parser.add_argument('--start_date', dest='start_date', help="Overrides the start date in the config file. Set to "
                                                                "LATEST to get the date of the last merged accquistion")
    parser.add_argument('--end_date', dest='end_date', help="Overrides the end date in the config file. Set to TODAY"
                                                            "to get today's date")
    parser.add_argument('-b', '--build_composite', dest='build_composite', action='store_true', default=False)
    parser.add_argument('-d', '--download', dest='do_download', action='store_true', default=False)
    parser.add_argument('-p', '--preprocess', dest='do_preprocess', action='store_true',  default=False)
    parser.add_argument('-m', '--merge', dest='do_merge', action='store_true', default=False)
    parser.add_argument('-a', '--mask', dest='do_mask', action='store_true', default=False)
    parser.add_argument('-s', '--stack', dest='do_stack', action='store_true', default=False)
    parser.add_argument('-c', '--classify', dest='do_classify', action='store_true', default=False)
    parser.add_argument('-u', '--update', dest='do_update', action='store_true', default=False)
    parser.add_argument('-r', '--remove', dest='do_delete', action='store_true', default=False)

    args = parser.parse_args()

    # If any processing step args are present, do not assume that we want to do all steps
    if (args.do_download or args.do_preprocess or args.do_merge or args.do_stack or args.do_classify) == True:
        do_all = False

    conf = configparser.ConfigParser()
    conf.read(args.config_path)
    sen_user = conf['sent_2']['user']
    sen_pass = conf['sent_2']['pass']
    project_root = conf['forest_sentinel']['root_dir']
    aoi_path = conf['forest_sentinel']['aoi_path']
    start_date = conf['forest_sentinel']['start_date']
    end_date = conf['forest_sentinel']['end_date']
    log_path = conf['forest_sentinel']['log_path']
    cloud_cover = conf['forest_sentinel']['cloud_cover']
    cloud_certainty_threshold = int(conf['forest_sentinel']['cloud_certainty_threshold'])
    model_path = conf['forest_sentinel']['model']
    sen2cor_path = conf['sen2cor']['path']
    composite_start_date = conf['forest_sentinel']['composite_start']
    composite_end_date = conf['forest_sentinel']['composite_end']

    pyeo.create_file_structure(project_root)
    log = pyeo.init_log(log_path)

    l1_image_dir = os.path.join(project_root, r"images/L1")
    l2_image_dir = os.path.join(project_root, r"images/L2")
    planet_image_dir = os.path.join(project_root, r"images/planet")
    merged_image_dir = os.path.join(project_root, r"images/merged")
    stacked_image_dir = os.path.join(project_root, r"images/stacked")
    catagorised_image_dir = os.path.join(project_root, r"output/categories")
    probability_image_dir = os.path.join(project_root, r"output/probabilities")
    composite_dir = os.path.join(project_root, r"composite")
    composite_l1_image_dir = os.path.join(project_root, r"composite/L1")
    composite_l2_image_dir = os.path.join(project_root, r"composite/L2")
    composite_merged_dir = os.path.join(project_root, r"composite/merged")

    if args.start_date == "LATEST":
        # This isn't nice, but returns the yyyymmdd string of the latest stacked image
        start_date = pyeo.get_s2_image_acquisition_time(pyeo.sort_by_s2_timestamp(
            [image_name for image_name in os.listdir(stacked_image_dir) if image_name.endswith(".tif")],
            recent_first=True
        )[0]).strftime("%Y%m%d")
    elif args.start_date:
        start_date = args.start_date
    if args.end_date == "TODAY":
        end_date = dt.date.today().strftime("%Y%m%d")
    elif args.end_date:
        end_date = args.end_date

    # Download and build the initial composite. Does not do by default
    if args.build_composite:
        log.info("Downloading for initial composite between {} and {} with cloud cover <= ()".format(
            composite_start_date, composite_end_date, cloud_cover))
        composite_products = pyeo.check_for_s2_data_by_date(aoi_path, composite_start_date, composite_end_date,
                                                         conf)
        pyeo.download_new_s2_data(composite_products, composite_l1_image_dir, composite_l2_image_dir)
        log.info("Preprocessing composite products")
        pyeo.atmospheric_correction(composite_l1_image_dir, composite_l2_image_dir, sen2cor_path,
                                    delete_unprocessed_image=True)
        log.info("Aggregating composite layers")
        pyeo.aggregate_and_mask_10m_bands(composite_l2_image_dir, composite_merged_dir, cloud_certainty_threshold)
        log.info("Building initial cloud-free composite")
        pyeo.composite_directory(composite_merged_dir, composite_dir)

    # Query and download all images since last composite
    if args.do_download or do_all:
        products = pyeo.check_for_s2_data_by_date(aoi_path, start_date, end_date, conf)
        log.info("Downloading")
        pyeo.download_new_s2_data(products, l1_image_dir)

    # Atmospheric correction
    if args.do_preprocess or do_all:
        log.info("Applying sen2cor")
        pyeo.atmospheric_correction(l1_image_dir, l2_image_dir, sen2cor_path, delete_unprocessed_image=True)

    # Aggregating layers into single image
    if args.do_merge or do_all:
        log.info("Aggregating layers")
        pyeo.aggregate_and_mask_10m_bands(l2_image_dir, merged_image_dir, cloud_certainty_threshold)

    log.info("Finding most recent composite")
    latest_composite_name = \
        pyeo.sort_by_timestamp(
            [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
            recent_first=True
        )[0]
    latest_composite_path = os.path.join(composite_dir, latest_composite_name)
    log.info("Most recent composite at {}".format(latest_composite_path))

    log.info("Sorting image list")
    images = \
        pyeo.sort_by_timestamp(
            [image_name for image_name in os.listdir(merged_image_dir) if image_name.endswith(".tif")],
            recent_first=True
        )
    log.info("Images to process: {}".format(images))

    for image in images:
        log.info("Detecting change for {}".format(image))
        new_image_path = os.path.join(merged_image_dir, image)

        # Stack with composite
        if args.do_stack or do_all:
            log.info("Stacking {} with composite {}".format(new_image_path, latest_composite_path))
            new_stack_path = pyeo.stack_old_and_new_images(latest_composite_path, new_image_path, stacked_image_dir)

        # Classify with composite
        if args.do_classify or do_all:
            log.info("Classifying with composite")
            new_class_image = os.path.join(catagorised_image_dir, "class_{}".format(os.path.basename(new_stack_path)))
            new_prob_image = os.path.join(probability_image_dir, "prob_{}".format(os.path.basename(new_stack_path)))
            pyeo.classify_image(new_stack_path, model_path, new_class_image, new_prob_image, num_chunks=10)


        # Update composite
        if args.do_update or do_all:
            log.info("Updating composite")
            new_composite_path = os.path.join(composite_dir, "composite_"+os.path.basename(image))
            pyeo.composite_images_with_mask((latest_composite_path, new_image_path), new_composite_path)
            latest_composite_path = new_composite_path

    log.info("***PROCESSING END***")
