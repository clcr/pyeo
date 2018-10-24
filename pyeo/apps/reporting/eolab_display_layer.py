import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import csv


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


def create_report(class_path, certainty_path, out_dir, class_colour_key=DEFAULT_KEY):
    create_report_layer(raster_path) # whoops dont actually need to do anything here
    pyeo.flatten_probability_image(certainty_path, os.path.join(out_dir, ))
    create_display_layer_(raster_path)

def create_display_layer(raster_path):

    pyeo.create_matching_dataset(raster_path, out_path)
