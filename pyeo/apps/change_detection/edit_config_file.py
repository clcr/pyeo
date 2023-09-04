#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:41:43 2023

@author: hb91
"""

"""
edit_config_file
-------------------------------------
An app that allows the user to read the contents of the initial config file
and confirm or update all processing parameters. The output will be saved
in a new config file.
"""

import argparse
import datetime
import os
from osgeo import gdal
import sys
from pyeo import filesystem_utilities
from pyeo.acd_national import acd_initialisation

gdal.UseExceptions()


def edit_config(config_path_in, config_path_out):
    """
    The main function that creates the updated new config file with 
       user inputs.
       
    Args:

        config_path_in : string with the full path to the original config file 
                        containing the default processing parameters

        config_path_out : string with the full path to the new config file 
                        that will be created with the user inputs
                        
    Returns:
        
        config_path : string containing the path to the new config file. This
                        may differ from the original path if the file already
                        existed.

    """

    config_dict, log = acd_initialisation(config_path_in)

    log.info("------------------------------------------------")
    log.info("Interactive editing of the config file for pyeo.")
    log.info("------------------------------------------------")
    log.info("NOTE: This function does not check the validity of user inputs.")

    # check whether input file exists
    if not os.path.exists(config_path_in):
        log.error("ERROR: Input config file path does not exist: " + config_path_in)
        sys.exit(1)

    # check whether output file already exists
    if os.path.exists(config_path_out):
        # if the output file already exists, append today's date and time to the 
        #    output file name to create a new file name
        config_path_out = config_path_out + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log.warning("Output config file path already exists. Using new file name:" + \
                    config_path_out)

    config_path = filesystem_utilities.input_to_config_path(config_path_in, 
                                                            config_path_out)

    log.info(f"Finished saving edited config file: {config_path_out}")
        
    return config_path


##############################
# Main block of code
##############################

if __name__ == "__main__":

    # Reading in existing config file
    parser = argparse.ArgumentParser(
        description="User confirmation and editing of a config file for pyeo."
    )
    parser.add_argument(
        dest="config_path_in",
        action="store",
        default=r"pyeo_linux.ini",
        help="A path to an existing .ini file containing the specification"
        "for the job. See pyeo/pyeo_linux.ini for an example.",
    )
    parser.add_argument(
        dest="config_path_out",
        action="store",
        default=r"pyeo_linux_test.ini",
        help="A path where the new .ini file with the user inputs will be created.",
    )

    args = parser.parse_args()

    edit_config(**vars(args))