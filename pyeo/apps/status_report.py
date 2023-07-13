"""
status_report
-------------------------------------
An app for listing the number of files or subdirectories that reside in the file system of the change detection processing.
It is meant to provide a quick overview of the processing status of multiple jobs.
Directories are taken from a .ini file.
"""

import os
import configparser
import argparse
from datetime import datetime as dt

from pyeo.filesystem_utilities import get_filenames, init_log

parser = argparse.ArgumentParser(
    description="Status Reporting on a complex directory structure."
)
parser.add_argument(
    dest="config_path",
    action="store",
    default=r"change_detection.ini",
    help="A path to a .ini file containing the specification for the job. See "
    "pyeo/apps/change_detection/change_detection.ini for an example.",
)
args = parser.parse_args()
config_path = vars(args)["config_path"]
conf = configparser.ConfigParser(allow_no_value=True)
conf.read(config_path)
sen_user = conf["sent_2"]["user"]
sen_pass = conf["sent_2"]["pass"]
root_dir = conf["forest_sentinel"]["root_dir"]

now = dt.now()  # current date and time
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
time = now.strftime("%H:%M:%S")
timestamp = now.strftime("%Y%m%dT%H%M%S")

log = init_log(os.path.join(root_dir, "status_log_" + timestamp + ".txt"))
out = os.path.join(root_dir, "status_report_" + timestamp + ".txt")
files = sorted(os.listdir(root_dir))
run_dirs = [o for o in files if os.path.isdir(os.path.join(root_dir, o))]
log.info("Root directory: {}".format(root_dir))

with open(out, "w") as f_out:
    line = "run, cmpL1Cd, cmpL1Cf, cmpL2Ad, cmpL2Af, cmpstacksd, cmpstacksf, compsd, compsf, imL1Cd, imL1Cf, imL2Ad, imL2Af, imstacksd, imstacksf, classmapsd, classmapsf, probmapsd, probmapsf\n"
    f_out.write(line)
    for d in run_dirs:
        run_dir = os.path.join(root_dir, d)
        try:
            l1_image_dir = os.path.join(run_dir, r"images/L1C")
            l2_image_dir = os.path.join(run_dir, r"images/L2A")
            l2_masked_image_dir = os.path.join(run_dir, r"images/cloud_masked")
            categorised_image_dir = os.path.join(run_dir, r"output/classified")
            probability_image_dir = os.path.join(run_dir, r"output/probabilities")
            composite_dir = os.path.join(run_dir, r"composite")
            composite_l1_image_dir = os.path.join(run_dir, r"composite/L1C")
            composite_l2_image_dir = os.path.join(run_dir, r"composite/L2A")
            composite_l2_masked_image_dir = os.path.join(
                run_dir, r"composite/cloud_masked"
            )
        except:
            log.warning("Something went wrong with {}.".format(run_dir))

        all_sub_dirs = [
            composite_l1_image_dir,
            composite_l2_image_dir,
            composite_l2_masked_image_dir,
            composite_dir,
            l1_image_dir,
            l2_image_dir,
            l2_masked_image_dir,
            categorised_image_dir,
            probability_image_dir,
        ]
        line = d
        for sd in all_sub_dirs:
            try:
                dir_contents = [
                    o for o in os.listdir(sd) if os.path.isdir(os.path.join(sd, o))
                ]
                """
                log.info("  Directories in {}".format(os.path.basename(sd)))
                for dd in dir_contents:
                    log.info("    {}".format(dd))
                """

                file_contents = [
                    o for o in os.listdir(sd) if os.path.isfile(os.path.join(sd, o))
                ]
                # log.info("  {}; {}; {}".format(os.path.basename(sd), len(dir_contents), len(file_contents)))
                line = (
                    line
                    + ", "
                    + str(len(dir_contents))
                    + ", "
                    + str(len(file_contents))
                )

                """
                log.info("  Files in {}".format(os.path.basename(sd)))
                for ff in file_contents:
                    log.info("    {}".format(ff))
                """
            except:
                log.warning("  Incomplete subdirectory structure in {}".format(run_dir))
        line = line + "\n"
        f_out.write(line)
f_out.close()
