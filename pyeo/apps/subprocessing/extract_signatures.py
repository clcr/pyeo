"""Command line application for extracting signatures from a .tif
 file and a folder containing a .shp of the same name. Example of use:
    python extract_signatures.py in_ras ras1.tif ras2.tif out sigs.csv
 """

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pc
import csv
import os
import glob
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts the signatures from a list of .tif files')
    parser.add_argument('in_ras', action='store', help="List of tif files to read", nargs="+")
    parser.add_argument("out", action='store', help="Path of the output .csv file")
    args = parser.parse_args()

    for training_image_file_path in args.in_ras:
        training_image_folder, training_image_name = os.path.split(training_image_file_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path = os.path.join(training_image_folder, training_image_name, training_image_name + '.shp')
        this_training_data, this_classes = pc.get_training_data(training_image_file_path, shape_path)

        sigs=np.vstack((this_classes, this_training_data.T))

        with open(args.out, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(sigs.T)
