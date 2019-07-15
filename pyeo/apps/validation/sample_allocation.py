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
    conf = configparser.ConfigParser()
    conf.read(args.conf_path)

    user_accuracies = conf._sections["user_accuracy"]
    pinned_samples = conf._sections["pinned_samples"]
    pinned_samples = {map_class: int(number) for map_class, number in pinned_samples.items()}
    user_accuracies = {map_class: float(number) for map_class, number in user_accuracies.items()}
    for map_class in user_accuracies:
        if map_class not in pinned_samples:
            pinned_samples.update({map_class: None})

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
