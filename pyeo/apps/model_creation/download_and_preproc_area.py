import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Downloads and preprocessses all images for an area")
    parser.add_argument("aoi_path")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("l1_dir")
    parser.add_argument("l2_dir")
    parser.add_argument("merge_dir")
    parser.add_argument("stack_dir")
    parser.add_argument("conf")
    parser.add_argument("-f", "--filter", action="store_true")

    args=parser.parse_args()

    products = pyeo.check_for_s2_data_by_date(args.aoi_path, args.start_date, args.end_date, args.conf)
    if args.filter:
        products = pyeo.filter_non_matching_s2_data(products)
    pyeo.download_s2_data(products, args.l1_dir, args.l2_dir)
    pyeo.preprocess_sen2_images(args.l2_dir, args.merge_dir, args.l1_dir, cloud_threshold=0, buffer_size=5)
