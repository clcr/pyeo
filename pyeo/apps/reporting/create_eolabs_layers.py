import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import csv
import gdal
import numpy as np
import boto3
import argparse

# Class, R, G, B, A, label
DEFAULT_KEY = [
    ["1", "12", "193", "76", "255", "Stable Forest"],
    ["2", "229", "232", "23", "255", "Forest -> Veg"],
    ["3", "249", "31", "45", "255", "Forest -> Non-Forest"],
    ["4", "116", "110", "22", "255", "Stable Veg"],
    ["5", "255", "237", "170", "255", "Veg -> Non-Forest"],
    ["6", "237", "248", "185", "255", "Veg -> Forest"],
    ["7", "205", "235", "238", "255", "Stable Non-Forest"],
    ["8", "157", "211", "167", "255", "Non-Forest -> Forest"],
    ["9", "100", "171", "176", "255", "Non-Forest -> Veg"],
    ["10", "43", "131", "186", "255", "Water"]
]


def create_report(class_path, certainty_path, out_dir, class_color_key=DEFAULT_KEY):
    if class_color_key != DEFAULT_KEY:
        class_color_key = load_color_pallet(class_color_key)
    pyeo.flatten_probability_image(certainty_path, os.path.join(out_dir, "prob.tif"))
    create_display_layer(class_path, os.path.join(out_dir, "display.tif"), class_color_key)


def create_display_layer(class_path, out_path, class_color_key):
    class_raster = gdal.Open(class_path)
    display_raster = pyeo.create_matching_dataset(class_raster, out_path, bands=3)
    display_array = display_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)

    class_array = class_raster.GetVirtualMemArray()
    for index, class_pixel in np.ndenumerate(class_array):
        display_array[:, index[0], index[1]] =\
            [class_row[1:4] for class_row in class_color_key if class_row[0] == str(class_pixel)]
    display_array = None
    class_array = None
    display_raster = None
    class_raster = None


def load_color_pallet(pallet_path):
    reader = csv.reader(pallet_path)
    out = list(reader)
    return out


def write_color_pallet(pallet, pallet_path):
    with open(pallet_path, "w") as f:
        writer = csv.writer(f)
        writer.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produces a zip of products to upload to EOLabs")
    parser.add_argument("class_path")
    parser.add_argument("certainty_path")
    parser.add_argument("output_folder")
    parser.add_argument("-p" "--pallet", dest="pallet")
    args = parser.parse_args()

    create_report(args.class_path, args.certainty_path, args.output_folder)


