"""
sample_allocation
-----------------
Creates a shapefile of randomly sampled sample points. See `validation`_ for more details, including setup of
configuration file.

Usage:

::

    sample_allocation config_path.ini

"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
from pyeo import validation
import configparser
import argparse


def main():
    parser = argparse.ArgumentParser("Sample allocation: given a specification in a .ini file,"
                                     " creates a set of points to pass to CollectEarth. MORE HERE.")
    parser.add_argument("conf_path")
    args = parser.parse_args()
    args.conf_path = os.path.abspath(args.conf_path)
    if not os.path.exists(args.conf_path):
        raise FileNotFoundError(
            "No configuration found at {}, please check path to .ini file".format(args.conf_path))
    conf = configparser.ConfigParser()
    conf.read(args.conf_path)


    user_accuracies = conf["user_accuracy"]
    user_accuracies = {int(map_class): float(number) for map_class, number in user_accuracies.items()}
    if "pinned_samples" in conf:
        pinned_samples = conf["pinned_samples"]
        pinned_samples = {int(map_class): int(number) for map_class, number in pinned_samples.items()}
    else:
        pinned_samples = None


    validation.create_validation_scenario(
        in_map_path=conf["paths"]["input_path"],
        out_shapefile_path=conf["paths"]["output_path"],
        target_standard_error=float(conf["augments"]["target_standard_error"]),
        user_accuracies=user_accuracies,
        pinned_samples=pinned_samples,
        no_data_class=conf["augments"]["no_data_class"],
        produce_csv=True
    )


if __name__ == "__main__":
    main()
