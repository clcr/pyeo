import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import configparser
import argparse
import os

if __name__ == "__main__":

    do_all = True

    # Reading in config file
    parser = argparse.ArgumentParser(description='Automatically detect and report on change')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'change_detection.ini',
                        help="Path to the .ini file specifying the job.")
    parser.add_argument('-d', '--download', dest='do_download', action='store_true', default=False)
    parser.add_argument('-p', '--preprocess', dest='do_preprocess', action='store_true',  default=False)
    parser.add_argument('-m', '--merge', dest='do_merge', action='store_true', default=False)
    parser.add_argument('-s', '--stack', dest='do_stack', action='store_true', default=False)
    parser.add_argument('-c', '--classify', dest='do_classify', action='store_true', default=False)
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
    composite_path = conf['forest_sentinel']['composite']
    sen2cor_path = conf['sen2cor']['path']

    pyeo.create_file_structure(project_root)
    log = pyeo.init_log(log_path)

    l1_image_path = os.path.join(project_root, r"images/L1")
    l2_image_path = os.path.join(project_root, r"images/L2")
    planet_image_path = os.path.join(project_root, r"images/planet")
    merged_image_path = os.path.join(project_root, r"images/merged")
    stacked_image_path = os.path.join(project_root, r"images/stacked")
    catagorised_image_path = os.path.join(project_root, r"output/categories")
    probability_image_path = os.path.join(project_root, r"output/probabilities")

    # Query and download
    if args.do_download or do_all:
        products = pyeo.check_for_s2_data_by_date(aoi_path, start_date, end_date, conf)
        log.info("Downloading")
        pyeo.download_new_s2_data(products, l1_image_path)

    # Atmospheric correction
    if args.do_preprocess or do_all:
        log.info("Applying sen2cor")
        pyeo.atmospheric_correction(l1_image_path, l2_image_path, sen2cor_path, delete_unprocessed_image=True)

    # Aggregating layers into single image
    if args.do_merge or do_all:
        log.info("Aggregating layers")
        pyeo.aggregate_and_mask_10m_bands(l2_image_path, merged_image_path, cloud_certainty_threshold)

    # Stack layers
    if args.do_stack or do_all:
        log.info("Stacking before and after images")
        pyeo.create_new_stacks(merged_image_path, stacked_image_path)