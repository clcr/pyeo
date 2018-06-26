import os
import logging
import platform
import subprocess


def sen2cor_process(safe_filepath, conf):
    """Uses sen2cor to calculate surface reflectance."""
    if platform.system() == "Windows":
        sen2cor_process_windows(safe_filepath, conf)
    if platform.system() == "Linux":
        sen2cor_process_unix()


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


def sen2cor_process_unix(safe_filepath, conf):
    # TODO Implement
    pass
