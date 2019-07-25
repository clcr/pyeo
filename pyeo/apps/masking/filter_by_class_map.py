"""
filter_by_class_map
-------------------
Filters every pixel in input_image that is not one of filter_classes.

Usage:

::

   $ filter_by_class_map my_image.tif my_class_map.tif my_output_image.tif useful_class_1 useful_class_2

This will create an image, my_output_image.tif, that contains only the pixels from my_image.tif
that are labelled as useful_class_1 and useful_class_2 in my_class_map.tif
"""
import pyeo.filesystem_utilities
import pyeo.raster_manipulation

import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Filter all of class out of input_image')
    parser.add_argument("input_image")
    parser.add_argument("class_image")
    parser.add_argument("output")
    parser.add_argument("filter_classes", nargs="*", type=int)
    parser.add_argument("-l", "--log_path", default=os.path.join(os.getcwd(), "comparison.log"))
    args = parser.parse_args()

    log = pyeo.filesystem_utilities.init_log(args.log_path)

    pyeo.raster_manipulation.filter_by_class_map(args.input_image, args.class_image, args.output, args.filter_classes)
