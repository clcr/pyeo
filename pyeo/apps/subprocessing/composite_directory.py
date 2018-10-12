"""Creates a composite of all imagery in the target directory. Assumes that each image name contains an S2-like
 timestamp (yyyymmddThhmmss), and the each .tif has an associated multiplicative mask file (where masked pixels
 are 0 and unmasked pixels are 1)

 Usage: python composite_directory.py /path/to/image/dir path/to/output.tif

 This script will sort in_dir from the earliest to the latest image based on S2 timestamps. It will iterate through
 this list, saving any unmasked pixels in each image into the image at out_path. Any pixels that are masked all
 the way through will be 0 at present (nodata to be implemented).

 Masks can either be single band images or n-band image (same at the corresponding .tif)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core
import argparse
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Composites a directory of .tifs and .msks, creating an image of the '
                                                 'latest cloud-free pixels.')
    parser.add_argument('in_dir', action='store', help="Path to the directory to be composited")
    parser.add_argument('out_path', action='store', help="Output image path")
    parser.add_argument('-l', '--logpath', dest='logpath', action="store", default="composite.log",
                        help="Path to logfile (optional)")
    args = parser.parse_args()

    pyeo.core.init_log(args.logpath)

    pyeo.core.composite_directory(args.in_dir, args.out_path)
