"""Quick script that stacks and classifies two images"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import argparse
import os
from tempfile import TemporaryDirectory

if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(description='Compare old_image with new_image using model')
    parser.add_argument("old_image")
    parser.add_argument("new_image")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("-l", "--log_path", default=os.path.join(os.getcwd(), "comparison.log"))
    parser.add_argument("-c", "--chunks", default=16)
    parser.add_argument("-m", "--mask", action="store_true")
    args = parser.parse_args()

    log = pyeo.init_log(args.log_path)

    with TemporaryDirectory() as td:
        stacked_path = os.path.join(td, "stacked.tif")
        pyeo.stack_images([args.old_image, args.new_image], stacked_path, geometry_mode="intersect")

        pyeo.classify_image(stacked_path, args.model, args.output, prob_out_path=None,
                            num_chunks=args.chunks, apply_mask=args.mask)



