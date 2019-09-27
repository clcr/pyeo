import os

import pyeo.raster_manipulation
import csv
import gdal
import numpy as np
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
    pyeo.raster_manipulation.flatten_probability_image(certainty_path, os.path.join(out_dir, "prob.tif"))
    create_display_layer(class_path, os.path.join(out_dir, "display.tif"), class_color_key)


def create_display_layer(class_path, out_path, class_color_key):
    srs = """PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]"""
    class_raster = gdal.Open(class_path)
    display_raster = pyeo.raster_manipulation.create_matching_dataset(class_raster, out_path, bands=3, datatype=gdal.GDT_Byte)
    display_array = display_raster.GetVirtualMemArray(eAccess=gdal.GF_Write)

    class_array = class_raster.GetVirtualMemArray()
    for index, class_pixel in np.ndenumerate(class_array):
        display_array[:, index[0], index[1]] =\
            [class_row[1:4] for class_row in class_color_key if class_row[0] == str(class_pixel)][0]
    display_array = None
    class_array = None
    gdal.ReprojectImage(display_raster, dst_wkt=srs)
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


