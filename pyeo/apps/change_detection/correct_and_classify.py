"""
Downloads, preprocesses, applies terrain correction and classifies a set of images from Landsat or Sentinel-2
"""

from pyeo import terrain_correction
from pyeo import queries_and_downloads
from pyeo import raster_manipulation
from pyeo import filesystem_utilities
from pyeo import classification

import argparse
import configparser

if __name__ == "__main__":
    pass