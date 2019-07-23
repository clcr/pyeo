import glob
import logging
import os
import shutil
import subprocess

from pyeo.filesystem_utilities import check_for_invalid_l2_data
from pyeo.exceptions import BadS2Exception


def apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=False):
    """Applies sen2cor to the SAFE file at image_path. Returns the path to the new product."""
    # Here be OS magic. Since sen2cor runs in its own process, Python has to spin around and wait
    # for it; since it's doing that, it may as well be logging the output from sen2cor. This
    # approach can be multithreaded in future to process multiple image (1 per core) but that
    # will take some work to make sure they all finish before the program moves on.
    log = logging.getLogger(__name__)
    # added sen2cor_path by hb91
    log.info("calling subprocess: {}".format([sen2cor_path, image_path]))
    sen2cor_proc = subprocess.Popen([sen2cor_path, image_path],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)

    while True:
        nextline = sen2cor_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and sen2cor_proc.poll() is not None:
            break
        if "CRITICAL" in nextline:
            #log.error(nextline)
            raise subprocess.CalledProcessError(-1, "L2A_Process")

    log.info("sen2cor processing finished for {}".format(image_path))
    log.info("Validating:")
    if not check_for_invalid_l2_data(image_path.replace("MSIL1C", "MSIL2A")):
        log.error("10m imagery not present in {}".format(image_path.replace("MSIL1C", "MSIL2A")))
        raise BadS2Exception
    if delete_unprocessed_image:
            log.info("removing {}".format(image_path))
            shutil.rmtree(image_path)
    return image_path.replace("MSIL1C", "MSIL2A")


def apply_fmask(in_safe_dir, out_file, fmask_command="fmask_sentinel2Stacked.py"):
    """Calls fmask to create a new mask for L1 data"""
    # For reasons known only to the spirits, calling subprocess.run from within this function on a HPC cause the PATH
    # to be prepended with a Windows "eoenv\Library\bin;" that breaks the environment. What follows is a large kludge.
    if "torque" in os.getenv("PATH"):  # Are we on a HPC? If so, give explicit path to fmask
        fmask_command = "/data/clcr/shared/miniconda3/envs/eoenv/bin/fmask_sentinel2Stacked.py"
    log = logging.getLogger(__name__)
    args = [
        fmask_command,
        "-o", out_file,
        "--safedir", in_safe_dir
    ]
    log.info("Creating fmask from {}, output at {}".format(in_safe_dir, out_file))
    # pdb.set_trace()
    fmask_proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    while True:
        nextline = fmask_proc.stdout.readline()
        if len(nextline) > 0:
            log.info(nextline)
        if nextline == '' and fmask_proc.poll() is not None:
            break


def atmospheric_correction(in_directory, out_directory, sen2cor_path, delete_unprocessed_image=False):
    """Applies Sen2cor cloud correction to level 1C images"""
    log = logging.getLogger(__name__)
    images = [image for image in os.listdir(in_directory)
              if image.startswith('MSIL1C', 4)]
    # Opportunity for multithreading here
    for image in images:
        log.info("Atmospheric correction of {}".format(image))
        image_path = os.path.join(in_directory, image)
        #image_timestamp = get_sen_2_image_timestamp(image)
        if glob.glob(os.path.join(out_directory, image.replace("MSIL1C", "MSIL2A"))):
            log.warning("{} exists. Skipping.".format(image.replace("MSIL1C", "MSIL2A")))
            continue
        try:
            l2_path = apply_sen2cor(image_path, sen2cor_path, delete_unprocessed_image=delete_unprocessed_image)
        except (subprocess.CalledProcessError, BadS2Exception):
            log.error("Atmospheric correction failed for {}. Moving on to next image.".format(image))
            pass
        else:
            l2_name = os.path.basename(l2_path)
            log.info("L2  path: {}".format(l2_path))
            log.info("New path: {}".format(os.path.join(out_directory, l2_name)))
            os.rename(l2_path, os.path.join(out_directory, l2_name))