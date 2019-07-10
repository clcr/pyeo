from pyeo import validation
from pyeo.core import init_log
import configparser
import argparse


def main():
    parser = argparse.ArgumentParser("Sample allocation: given a specification in a .ini file,"
                                     " creates a set of points to pass to CollectEarth. MORE HERE.")
    parser.add_argument("conf_path")
    args = parser.parse_args()
    conf = configparser.ConfigParser()
    conf.read(args.conf_path)

    user_accuracies = conf["user_accuracy"]
    pinned_samples = conf["pinned_samples"]
    for map_class in user_accuracies:
        if map_class not in pinned_samples:
            pinned_samples.update({map_class: None})

    validation.create_validation_scenario(
        in_map=conf["paths"]["input_path"],
        out_shapefile=conf["paths"]["output_path"],
        target_standard_error=conf["augments"]["target_standard_error"],
        user_accuracies=user_accuracies,
        pinned_samples=pinned_samples
    )


if __name__ == "__main__":
    main()
