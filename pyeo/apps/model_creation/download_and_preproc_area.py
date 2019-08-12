import sys, os

import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import argparse
import configparser

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Downloads and preprocessses all images for an area")
    parser.add_argument("aoi_path", help="The path to a json of the area you wish to download")
    parser.add_argument("start_date", help="That start date for an image in yyyymmdd")
    parser.add_argument("end_date", help="The end date of the query, in yyyymmdd")
    parser.add_argument("l1_dir", help="The directory to store the level 1 images")
    parser.add_argument("l2_dir",  help="The directory to store the level 2 images")
    parser.add_argument("merge_dir", help="The directory to store the merged images")
    parser.add_argument("conf", help="The path to a config file containing your Copernicus login and password")
    parser.add_argument("--log_path", default="download_and_preproc.log")

    args=parser.parse_args()

    pyeo.filesystem_utilities.init_log(args.log_path)

    conf = configparser.ConfigParser()
    conf.read(args.conf)

    products = pyeo.queries_and_downloads.check_for_s2_data_by_date(args.aoi_path, args.start_date, args.end_date, conf)
    if args.download_l2_data:
        products = pyeo.queries_and_downloads.filter_non_matching_s2_data(products)
        pyeo.queries_and_downloads.download_s2_data(products, args.l1_dir, args.l2_dir, user=conf["sent_2"]["user"], passwd=conf["sent_2"]["pass"])
    else
    pyeo.raster_manipulation.preprocess_sen2_images(args.l2_dir, args.merge_dir, args.l1_dir, cloud_threshold=0, buffer_size=5)
