import sys
from pyeo import acd_national

if __name__ == "__main__":
    # if run from terminal, __name__ becomes "__main__"
    # sys.argv[0] is the name of the python script, e.g. acd_national.py

    # I.R. Changed as a list was getting passed but a single config_path string is expected by the receiving functions
    acd_national.automatic_change_detection_national(sys.argv[1])
