#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a unique runtime ID for pyeo from the AOI name in the ini file, 
  the current time and date, and the computer hostname

Created on Thu Feb  1 11:15:18 2024

@author: hb91
"""
import argparse
import configparser
import datetime
import random
from pyeo import filesystem_utilities

def make_runtime_id(config_path):
    # get AOI name from the ini file
    configparser.ConfigParser(allow_no_value=True)
    config_dict = filesystem_utilities.config_path_to_config_dict(config_path)
    aoi_name = config_dict['aoi_name']
    # create current timestamp
    timestamp = datetime.datetime.now().strftime(r"%Y%m%dT%H%M%S")
    # create a random number
    r = str(int(random.random()*1000000))
    new_runtime_id = "_".join( (aoi_name, timestamp, r) )
    return new_runtime_id

if __name__ == "__main__":
    # Reading in config file
    parser = argparse.ArgumentParser(
        description="Creates a runtime ID for pyeo."
    )
    parser.add_argument(
        dest="config_path",
        action="store",
        default=r"pyeo_linux.ini",
        help="A path to a .ini file containing the specification for the job. See "
        "pyeo/pyeo_linux.ini for an example.",
    )

    args = parser.parse_args()

    print(make_runtime_id(**vars(args)))