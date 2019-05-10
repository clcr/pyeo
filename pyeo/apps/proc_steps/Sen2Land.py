import os
import logging
import platform
import subprocess
import glob
import re


def sen2cor_process(safe_filepath, conf):
    """Uses sen2cor to calculate surface reflectance."""
    if platform.system() == "Windows":
        sen2cor_process_windows(safe_filepath, conf)
    if platform.system() == "Linux":
        sen2cor_process_unix(safe_filepath)


def sen2cor_process_windows(safe_filepath, conf):
    sen2cor_path = conf["post_processing"]["sen2cor_path"]
    sen2cor_cmd = os.path.join(sen2cor_path, 'L2A_Process.bat {}'.format(safe_filepath))
    proc = subprocess.Popen(sen2cor_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # There's got to be a better way to get the live out/stderr than this.
    while not proc.poll:
        print(proc.stdout.read())
        print(proc.stderr.read())
    print(proc.stdout.read())
    print(proc.stderr.read())
    return proc.returncode


def sen2cor_process_unix(image_directory, conf):
    """Applies Sen2cor cloud correction to level 1C images."""
    log = logging.getLogger(__name__)
    L2A_path = conf["post_processing"]["sen2cor_path"]
def sen2cor_process_unix(image_directory: str, out_directory: str, L2A_path: str,
                           delete_unprocessed_image: bool=False):
    """Applies Sen2cor cloud correction to level 1C images."""
    log = logging.getLogger(__name__)
    images = [image for image in os.listdir(image_directory)
              if image.startswith('MSIL1C', 4)]
    log.info(images)
    for image in images:
        image_path = os.path.join(image_directory, image)
        image_timestamp = get_sen_2_image_timestamp(image)
        if glob.glob(os.path.join(out_directory, r"*_{}_*".format(image_timestamp))):
            log.warning("{} exists. Skipping.".format(image))
            continue
        # Here be OS magic. Since sen2cor runs in its own process, Python has to spin around and wait
        # for it; since it's doing that, it may as well be logging the output from sen2cor. This
        # approatch can be multithreaded in future to process multiple image (1 per core) but that
        # will take some work to make sure they all finish before the program moves on.
        sen2cor_proc = subprocess.Popen([L2A_path, image_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True)
        try:
            while True:
                nextline = sen2cor_proc.stdout.readline()
                if nextline == '' and sen2cor_proc.poll() is not None:
                    break
                if "CRITICAL" in nextline:
                    log.error(nextline)
                    break
                log.info(nextline)
        except subprocess.CalledProcessError:
            log.error("Sen2Cor failed")
            break
        log.info("sen2cor processing finished for {}".format(image_path))
        if delete_unprocessed_image:
            os.rmdir(image_path)
    l2_glob = os.path.join(image_directory, r"*_MSIL2A_*")
    for l2_path in glob.glob(l2_glob):
        l2_name = os.path.basename(l2_path)
        os.rename(l2_path, os.path.join(out_directory, l2_name))


def get_sen_2_image_timestamp(image_name: str):
    """Returns the timestamps part of a Sentinel 2 image"""
    timestamp_re = r"\d{8}T\d{6}"
    ts_result = re.search(timestamp_re, image_name)
    return ts_result.group(0)


def move_l2_images(from_dir, to_dir):
    """Moves all L2 images in from_dir to to_dir"""
    l2_glob = os.path.join(from_dir, r"*_MSIL2A_*")
    for l2_path in glob.glob(l2_glob):
        l2_name = os.path.basename(l2_path)
        os.rename(l2_path, os.path.join(to_dir, l2_name))