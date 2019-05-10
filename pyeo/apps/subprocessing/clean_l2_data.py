import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import argparse
import pyeo.core as pyeo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Removes any SAFEfile in directory without the 2,3,4 or 8m imagery"
                                                 "in the 10m folder")
    parser.add_argument("l2_dir", help="Path to the directory that needs cleaning")
    parser.add_argument("-d", "--disable_warning", dest="do_warning", action='store_false', default=True,
                        help="If present, do not prompt before removing files")
    parser.add_argument("-r", "--resolution", dest="resolution", action="store", default="10m",
                        help="Resolution to check (10m, 20m or 60m")
    args = parser.parse_args()

    pyeo.init_log("clean_log.log")
    pyeo.clean_l2_dir(args.l2_dir, args.resolution, args.do_warning)
