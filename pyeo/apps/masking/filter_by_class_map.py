"""Quick script that stacks and classifies two images"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import argparse
import os
from tempfile import TemporaryDirectory

if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(description='Filter all of class out of input_image')
    parser.add_argument("input_image")
    parser.add_argument("class_image")
    parser.add_argument("output")
    parser.add_argument("filter_classes", nargs="*", type=int)
    parser.add_argument("-l", "--log_path", default=os.path.join(os.getcwd(), "comparison.log"))
    args = parser.parse_args()

    log = pyeo.init_log(args.log_path)

    pyeo.filter_by_class_map(args.input_image, args.class_image, args.output, args.filter_classes)
