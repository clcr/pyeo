"""
get raster stats and file sizes from all tiff files in a directory
"""

import pyeo.raster_manipulation
import pyeo.filesystem_utilities
from pyeo.filesystem_utilities import get_filenames
from pyeo.raster_manipulation import get_stats_from_raster_file

import argparse
import os
from osgeo import gdal
from tempfile import TemporaryDirectory

gdal.UseExceptions()


def reports(path, logfile):
    # get all image paths
    image_paths = [
        f.path for f in os.scandir(path) if f.is_file() and f.name.endswith(".tif")
    ]
    if len(image_paths) == 0:
        raise FileNotFoundError("No tiff images found in {}.".format(path))
    # sort class images by image acquisition date
    image_paths = list(
        filter(pyeo.filesystem_utilities.get_image_acquisition_time, image_paths)
    )
    image_paths.sort(
        key=lambda x: pyeo.filesystem_utilities.get_image_acquisition_time(x)
    )
    log = pyeo.filesystem_utilities.init_log(logfile)
    for index, image in enumerate(image_paths):
        log.info("{}: {}".format(index + 1, image))
        log.info(
            "   Size: {} MB".format(
                image, str(round(os.path.getsize(image) / 1024 / 2024, 3))
            )
        )
        get_stats_from_raster_file(image, format="GTiff", missing_data_value=0)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick raster stats reporting.")
    parser.add_argument(
        dest="path", action="store", default=r"~", help="A path to a directory."
    )
    parser.add_argument(
        dest="logfile",
        action="store",
        default=r"~/raster_stats_log.txt",
        help="A path to the log file.",
    )
    args = parser.parse_args()

    reports(**vars(args))
