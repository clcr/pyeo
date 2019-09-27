"""
image_comparison
----------------
An application for applying a pickled scikit-learn model to two contiguous raster images.

Example call:

::

   $image_comparison image_1.tif image_2.tif model.pkl class_map.tif
"""

import pyeo.classification
import pyeo.raster_manipulation
import pyeo.filesystem_utilities

import argparse
import os
from tempfile import TemporaryDirectory

if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(description='Compare old_image with new_image using model')
    parser.add_argument("old_image", help="Path the the older image to be compared")
    parser.add_argument("new_image", help="Path to the newer image to be compard")
    parser.add_argument("model", help="")
    parser.add_argument("output")
    parser.add_argument("-l", "--log_path", default=os.path.join(os.getcwd(), "comparison.log"))
    parser.add_argument("-c", "--chunks", default=16)
    parser.add_argument("-m", "--mask", action="store_true")
    args = parser.parse_args()

    log = pyeo.filesystem_utilities.init_log(args.log_path)

    with TemporaryDirectory() as td:
        stacked_path = os.path.join(td, "stacked.tif")
        pyeo.raster_manipulation.stack_images([args.old_image, args.new_image], stacked_path, geometry_mode="intersect")
        pyeo.classification.classify_image(stacked_path, args.model, args.output, prob_out_path=None,
                                           num_chunks=args.chunks, apply_mask=args.mask)



