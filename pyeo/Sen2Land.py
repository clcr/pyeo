import os
import logging
import platform
import subprocess

def sen2cor_process(safe_filepath, conf):
    """Uses sen2cor to calculate surface reflectance."""
    if platform.system() == "Windows":
        sen2cor_process_windows(safe_filepath, conf)




def sen2cor_process_windows(safe_filepath, conf):
    sen2cor_path = conf["post_processing"]["sen2cor_path"]
    sen2cor_cmd = 'L2A_Process.bat {}'.format(safe_filepath)
    logging.DEBUG(sen2cor_cmd)
    sen2cor_out = subprocess.run([sen2cor_path, safe_filepath], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)


def sen2cor_process_unix(safe_filepath, conf):
    # TODO Implement
    pass


def atmospheric_correction(image_directory, L2A_path, delete_unprocessed_image=False):
    """Applies Sen2cor cloud correction to level 1C images"""
    images = [image for image in os.listdir(image_directory)
              if image.startswith('MSIL1C', 4)]
    print(images)
    for image in images:
        image_path = os.path.join(image_directory, image)
        sen2cor_out = subprocess.run([L2A_path, '--resolution=10', image_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(sen2cor_out.stdout)  # TODO: Change to log
    pass