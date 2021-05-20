from pyeo import filesystem_utilities, raster_manipulation
import argparse


def main(l1_dir, l2_dir, merge_dir, stacked_dir, bands, resolution):
    log = filesystem_utilities.init_log("preprocess_and_stack.log")
    raster_manipulation.preprocess_sen2_images(l2_dir, merge_dir, l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=bands, out_resolution=resolution)

    log.info("Stacking images pairs from {} in {}".format(merge_dir, stacked_dir))
    raster_manipulation.create_new_stacks(merge_dir, stacked_dir)

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("l1_dir")
    parser.add_argument("l2_dir")
    parser.add_argument("merge_dir")
    parser.add_argument("stacked_dir")
    parser.add_argument("--bands", nargs="*", default=("B08", "B04", "B03", "B02"),)
    parser.add_argument("--resolution", default=10)
    args = parser.parse_args()
    main(args.l1_dir, args.l2_dir, args.merge_dir, args.stacked_dir, args.bands, args.resolution)
