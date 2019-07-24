"""Creates a composite of all imagery in the target directory. Assumes that each image name contains an S2-like
 timestamp (yyyymmddThhmmss), and the each .tif has an associated multiplicative mask file (where masked pixels
 are 0 and unmasked pixels are 1)

 Usage: python composite_directory.py /path/to/image/dir path/to/output.tif

 This script will sort in_dir from the earliest to the latest image based on S2 timestamps. It will iterate through
 this list, saving any unmasked pixels in each image into the image at out_path. Any pixels that are masked all
 the way through will be 0 at present (nodata to be implemented).

 Masks can either be single band images or n-band image (same at the corresponding .tif)
"""

import os

import pyeo.raster_manipulation
import pyeo.filesystem_utilities
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Composites a directory of .tifs and .msks, creating an image of the '
                                                 'latest cloud-free pixels.')
    parser.add_argument('in_dir', action='store', help="Path to the directory to be composited")
    parser.add_argument('out_path', action='store', help="Output image path")
    parser.add_argument('-m', '--merge_bands', action="store", dest="merge_path",
                        help='Treats the contents if in_dir as a L2 S2 directroy of .SAFE files; merges 10m images'
                             'prior to compositing and stores them in the argument')
    parser.add_argument('-l', '--logpath', dest='logpath', action="store", default="composite.log",
                        help="Path to logfile (optional)")
    parser.add_argument('-r', '--remask', dest='mask_path', action="store",
                        help="If present, remask the files using an image at model_path")
    parser.add_argument('-d', '--dates_image', dest="generate_dates_image", action = "store_true",
                        help="If present, will build a single-layer .tif of dates of pixels in composite")
    args = parser.parse_args()

    comp_dir = args.in_dir

    pyeo.filesystem_utilities.init_log(args.logpath)

    if args.merge_path:
        comp_dir = args.merge_path
        for safe_file in [os.path.join(os.path.dirname(args.in_dir), file) for file in os.listdir(args.in_dir)]:
            pyeo.raster_manipulation.stack_sentinel_2_bands(safe_file, comp_dir)

    if args.mask_path:
        for image in [os.path.join(os.path.dirname(comp_dir), file) for file in os.listdir(comp_dir)]:
            pyeo.raster_manipulation.create_mask_from_model(image, args.mask_path)

    pyeo.raster_manipulation.composite_directory(comp_dir, args.out_path, generate_date_images=args.generate_dates_image)
