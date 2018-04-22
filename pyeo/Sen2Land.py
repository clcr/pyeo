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
    sen2cor_out = subprocess.run([sen2cor_path, safe_filepath], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
    return sen2cor_out

def sen2cor_process_unix(safe_filepath, conf):
    # TODO Implement
    pass
