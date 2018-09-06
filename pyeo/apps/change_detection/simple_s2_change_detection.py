"""A change detection script that downloads, stacks and classifies a set of 10m sentinel 2 images.

When run, this script will download every S2 image in the aoi (defined by the geojson at aoi_path) between the two
dates that meets the specified cloud cover range. It will use the sen2cor distribution specified in the .ini file
to atmospherically correct the data to L2A, merge each set of 10m bands into a single geotiff, stack the images
into pairs based on the algorithm in create_new_stacks and classify those images using a scikit-learn model

To use this script, fill out the [sent_2], [forest_sentinel] and [sen2cor] sections of the configuration file
change_detection.ini, then run

$ python pyeo/apps/change_detection/simple_s2_change_detection.py

Produces two directories of un-mosaiced imagery; one of classified images and one of class probabilites"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..','..', '..')))
import pyeo.core as pyeo
import configparser
import argparse
import os

if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(description='Automatically detect and report on change')
    parser.add_argument('--conf', dest='config_path', action='store', default=r'change_detection.ini',
                        help="Path to the .ini file specifying the job.")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    conf.read(args.config_path)
    sen_user = conf['sent_2']['user']
    sen_pass = conf['sent_2']['pass']
    project_root = conf['forest_sentinel']['root_dir']
    aoi_path = conf['forest_sentinel']['aoi_path']
    start_date = conf['forest_sentinel']['start_date']
    end_date = conf['forest_sentinel']['end_date']
    log_path = conf['forest_sentinel']['log_path']
    cloud_cover = conf['forest_sentinel']['cloud_cover']
    cloud_certainty_threshold = int(conf['forest_sentinel']['cloud_certainty_threshold'])
    model_path = conf['forest_sentinel']['model']
    sen2cor_path = conf['sen2cor']['path']

    pyeo.create_file_structure(project_root)
    log = pyeo.init_log(log_path)

    l1_image_path = os.path.join(project_root, r"images/L1")
    l2_image_path = os.path.join(project_root, r"images/L2")
    planet_image_path = os.path.join(project_root, r"images/planet")
    merged_image_path = os.path.join(project_root, r"images/merged")
    stacked_image_path = os.path.join(project_root, r"images/stacked")
    catagorised_image_path = os.path.join(project_root, r"output/categories")
    probability_image_path = os.path.join(project_root, r"output/probabilities")

    # Query and download
    log.info("Downloading")
    pyeo.sent2_query(sen_user, sen_pass, aoi_path, start_date, end_date, l1_image_path, )

    # Atmospheric correction
    log.info("Applying sen2cor")
    pyeo.atmospheric_correction(l1_image_path, l2_image_path, sen2cor_path)

    # Aggregating layers into single image
    log.info("Aggregating layers")
    pyeo.aggregate_and_mask_10m_bands(l2_image_path, merged_image_path, cloud_certainty_threshold)

    # Stack layers
    log.info("Stacking before and after images")
    pyeo.create_new_stacks(merged_image_path, stacked_image_path)

    # Classify stacks
    log.info("Classifying images")
    pyeo.classify_directory(stacked_image_path, model_path, catagorised_image_path, probability_image_path,
                            apply_mask=True)


# # ############################################################################################################
# # ###### 3. Post processing                                                                  #################
# # ######    3.1 cloud masking  = DONE                                                        #################
# # ######    3.2 image tools: resample/clip/re-project/masking/plot                           #################
# # ######    3.3 regional mosaics                                                             #################
# # ######    3.4 time series stacks                                                           #################
# # ######    3.5 fusion tools: co-registration/ stacking                                      #################
# # ############################################################################################################

# 3.1 cloud mask

# 3.1.1 default masking using generated mask from Sen2Cor
# 3.1.2 option for user-input mask (a advanced random-forest based cloud masking)

# 3.2 image tools:
#  resample
#  os.system('gdalwarp -overwrite -tr 25 25 -r cubic ' + i + ' ' + iout)
#  clip
#  re-project
#  masking with user input layer (e.g. sea)

#3.3 regional mosaics

#3.4 time series stacks

#3.5 fusion tools: co-registration

# # ############################################################################################################
# # ###### 4. Analysing                                                                        #################
# # ######    4.1 change maps                                                                  #################
# # ######    4.2 annual stacks                                                                #################
# # ######    4.3 time series stacks / temporal statistics                                     #################
# # ######    4.4 real-time detection / alerts                                                 #################
# # ######    4.5 validation                                                                   #################
# # ############################################################################################################

#4.1 change maps: random forest based change detection

#4.2 annual stacks

#4.3 time series and temporal statistics

#4.4 real-time detection / alerts


#4.5 validation