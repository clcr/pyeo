import os
import logging


def sen2cor_process(safe_filepath, conf):
    sen2cor_cmd = 'L2A_Process --resolution 10 {}'.format(safe_filepath)
    logging.DEBUG(sen2cor_cmd)
    os.system()