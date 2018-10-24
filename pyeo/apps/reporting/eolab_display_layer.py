import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo


def create_report(raster_path, out_dir):
    create_report_layer(raster_path)
    create_certainty_layer(raster_path)
    create_display_layer_(raster_path)

def create_report_layer(raster_path):

    pyeo.create_matching_dataset(raster_path, out_path)
