import sys, os


sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))

import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities

import argparse
import configparser


def main(aoi_path, start_date, end_date, l1_dir, l2_dir, merge_dir, conf_path,
         download_l2_data=False, sen2cor_path=None, stacked_dir=None, bands=("B08", "B04", "B03", "B02"),
         resolution=10, cloud_cover=100):

    conf = configparser.ConfigParser()
    conf.read(conf_path)

    products = pyeo.queries_and_downloads.check_for_s2_data_by_date(aoi_path, start_date, end_date, conf,
                                                                    cloud_cover=cloud_cover)
    log.info("{} products found.".format(len(products)))
    if download_l2_data:
        products = pyeo.queries_and_downloads.filter_non_matching_s2_data(products)
        log.info("{} products left after non-matching".format(len(products)))
        pyeo.queries_and_downloads.download_s2_data(products, l1_dir, l2_dir, user=conf["sent_2"]["user"],
                                                    passwd=conf["sent_2"]["pass"])
    elif sen2cor_path:
        products = pyeo.queries_and_downloads.filter_to_l1_data(products)
        pyeo.queries_and_downloads.download_s2_data(products, l1_dir, l2_dir, user=conf["sent_2"]["user"],
                                                    passwd=conf["sent_2"]["pass"])
        pyeo.raster_manipulation.atmospheric_correction(l1_dir, l2_dir, sen2cor_path)
    else:
        log.critical("Please provide either the --download_l2_data flag or the arg --sen2cor"
                     "_path with a path to Sen2Cor")
        sys.exit(1)
    log.info("Extracting bands {} from L2 images and creating cloudmasks. Output in {}".format(bands, merge_dir))
    pyeo.raster_manipulation.preprocess_sen2_images(l2_dir, merge_dir, l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=bands, out_resolution=resolution)
    if stacked_dir:
        log.info("Stacking images pairs from {} in {}".format(merge_dir, stacked_dir))
        pyeo.raster_manipulation.create_new_stacks(merge_dir, stacked_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Downloads and preprocessses all images for an area.")
    parser.add_argument("aoi_path", help="The path to a json of the area you wish to download")
    parser.add_argument("start_date", help="That start date for an image in yyyymmdd")
    parser.add_argument("end_date", help="The end date of the query, in yyyymmdd")
    parser.add_argument("l1_dir", help="The directory to store the level 1 images")
    parser.add_argument("l2_dir",  help="The directory to store the level 2 images")
    parser.add_argument("merge_dir", help="The directory to store the merged images")
    parser.add_argument("conf_path", help="The path to a config file containing your Copernicus login and password")
    parser.add_argument("--log_path", default="download_and_preproc.log")
    parser.add_argument("--download_l2_data", action="store_true", help="If present, download L2 data")
    parser.add_argument("--sen2cor_path", help="Path to the sen2cor folder")
    parser.add_argument("--stacked_dir",  help="If present, will create a set of stacked image pairs in stacked_dir")
    parser.add_argument("--bands", nargs="*", default=("B08", "B04", "B03", "B02"),
                        help="If present, specifies the bands to include in the preprocessed output. "
                             "Ex: --bands B01 B03 B8A")
    parser.add_argument("--resolution", default=10, help="Resolution of final merged image.")
    parser.add_argument("--cloud_cover", default=100, help="Maximum cloud cover of images to download")

    args = parser.parse_args()

    log = pyeo.filesystem_utilities.init_log(args.log_path)

    main(args.aoi_path, args.start_date, args.end_date, args.l1_dir, args.l2_dir, args.merge_dir, args.conf_path,
         args.download_l2_data, args.sen2cor_path, args.stacked_dir, args.bands, args.resolution, args.resolution,
         args.cloud_cover)


