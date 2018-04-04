import os
import logging


def sen2cor_process(safe_filepath, conf):
    #On windows
    sen2cor_path = conf["post_processing"]["sen2cor_path"]
    sen2cor_cmd = 'L2A_Process.bat --resolution 10 {}'.format(safe_filepath)
    logging.DEBUG(sen2cor_cmd)
    os.system()