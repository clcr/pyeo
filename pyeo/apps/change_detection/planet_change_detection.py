"""
planet_change_detection
-----------------------
UNIFINISHED
"""

import argparse
import configparser
import os
import pyeo

if __name__ == "__main__":
    # Load config

    parser = argparse.ArgumentParser(description='Cla')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'change_detection.ini',
                        help="Path to the .ini file specifying the job.")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    conf.read(args.config_path)
    api_key_path = conf['planet']['api_key_location']
    project_root = conf['planet_change_detection']['root_dir']
    aoi_path = conf['planet_change_detection']['aoi_path']
    date_1 = conf['planet_change_detection']['image_1_date']
    date_2 = conf['planet_change_detection']['image_2_date']
    log_path = conf['planet_change_detection']['log_path']

    pyeo.filesystem_utilities.create_file_structure(project_root)
    log = pyeo.filesystem_utilities.init_log(log_path)

    planet_image_path = os.path.join(project_root, r"images/planet")
    stacked_image_path = os.path.join(project_root, r"images/stacked")
    catagorised_image_path = os.path.join(project_root, r"output/categories")
    probability_image_path = os.path.join(project_root, r"output/probabilities")

    api_key = pyeo.queries_and_downloads.load_api_key(api_key_path)

    # Download two images

    pyeo.queries_and_downloads.download_planet_image_on_day(aoi_path, date_1, planet_image_path, api_key)
    pyeo.queries_and_downloads.download_planet_image_on_day(aoi_path, date_2, planet_image_path, api_key)

    # Save RGB


    # Do classification

    # Do the imshow transitive closure trick

    # Polygonise

    # Burn

    # Resample

