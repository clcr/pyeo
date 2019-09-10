import sys, os


sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))

import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities

import argparse
import configparser

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Downloads and preprocessses all images for an area. ")
    parser.add_argument("aoi_path", help="The path to a json of the area you wish to download")
    parser.add_argument("start_date", help="That start date for an image in yyyymmdd")
    parser.add_argument("end_date", help="The end date of the query, in yyyymmdd")
    parser.add_argument("l1_dir", help="The directory to store the level 1 images")
    parser.add_argument("l2_dir",  help="The directory to store the level 2 images")
    parser.add_argument("merge_dir", help="The directory to store the merged images")
    parser.add_argument("conf", help="The path to a config file containing your Copernicus login and password")
    parser.add_argument("--log_path", default="download_and_preproc.log")
    parser.add_argument("--download_l2_data", action="store_true", help="If present, download L2 data")
    parser.add_argument("--sen2cor_path", help="Path to the sen2cor folder")
    parser.add_argument("--stacked_dir",  help="If present, will create a set of stacked image pairs in stacked_dir")
    parser.add_argument("--bands", nargs="*", default=("B08", "B04", "B03", "B02"),
                        help="If present, specifies the bands to include in the preprocessed output. "
                             "Ex: --bands B01 B03 B8A")

    args = parser.parse_args()

    log = pyeo.filesystem_utilities.init_log(args.log_path)

    conf = configparser.ConfigParser()
    conf.read(args.conf)

    products = pyeo.queries_and_downloads.check_for_s2_data_by_date(args.aoi_path, args.start_date, args.end_date, conf)
    log.info("{} products found.".format(len(products)))
    if args.download_l2_data:
        products = pyeo.queries_and_downloads.filter_non_matching_s2_data(products)
        log.info("{} products left after non-matching".format(len(products)))
        pyeo.queries_and_downloads.download_s2_data(products, args.l1_dir, args.l2_dir, user=conf["sent_2"]["user"],
                                                    passwd=conf["sent_2"]["pass"])
    elif args.sen2cor_path:
        products = pyeo.queries_and_downloads.filter_to_l1_data(products)
        pyeo.queries_and_downloads.download_s2_data(products, args.l1_dir, args.l2_dir, user=conf["sent_2"]["user"],
                                                    passwd=conf["sent_2"]["pass"])
        pyeo.raster_manipulation.atmospheric_correction(args.l1_dir, args.l2_dir, args.sen2cor_path)
    else:
        log.critical("Please provide either the --download_l2_data flag or the arg --sen2cor"
                     "_path with a path to Sen2Cor")
        sys.exit(1)
    log.info("Extracting BGRI bands from L2 images and creating cloudmasks. Output in {}".format(args.merged_dir))
    pyeo.raster_manipulation.preprocess_sen2_images(args.l2_dir, args.merge_dir, args.l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=args.bands)
    if args.stacked_dir:
        log.info("Stacking images pairs from {} in {}".format(args.merged_dir, args.stacked_dir))
        pyeo.raster_manipulation.create_new_stacks(args.merge_dir, args.stacked_dir)