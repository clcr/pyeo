"""Command line application for extracting signatures from a .tif
 file and a folder containing a .shp of the same name. Example of use:
    python extract_signatures.py in_ras.tif training_polys.shp sigs.csv
 """

import argparse

from pyeo.classification import extract_features_to_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts the signatures from a .tif file and a polygon')
    parser.add_argument('in_ras_path', action='store', help="Path to the image file")
    parser.add_argument('training_shape_path', help="Path to a shapefile containing training polygons")
    parser.add_argument("out_path", action='store', help="Path of the output .csv file")
    parser.add_argument("--apply_mask", action='store_true', help="If true, apply the images .msk file to before "
                                                                "extracting signatures")
    parser.add_argument("--attribute", default="CODE", help="The name of the field in the training shape containing"
                                                            " class labels")
    args = parser.parse_args()

    extract_features_to_csv(args.in_ras_path, args.training_shape_path, args.out_path, attribute=args.attribute)
