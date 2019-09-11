"""App to generate a set of proportional stratified sample points from a class map"""

import argparse
from pyeo.validation import produce_stratified_validation_points
from pyeo.filesystem_utilities import init_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates a shapefile containing n stratified random points"
                                     "Note This is ONLY STRATIFIED. It does not impose a minimum sample size"
                                     "as recommended yet.")
    parser.add_argument("map_path", help="Path to a classidfied map to validate")
    parser.add_argument("out_path", help="Where to store the output .shpfile")
    parser.add_argument("n_points", help="The number of points to generate")
    parser.add_argument("--no_data", help="If given, a nodata class to ignore.")
    parser.add_argument("--random_seed", help="If given, override the random seed. For testing.")
    parser.add_argument("--log", help="Path to log.")

    args = parser.parse_args()

    if args.log:
        init_log(args.log)

    produce_stratified_validation_points(
        map_path=args.map_path,
        out_path=args.out_path,
        n_points=args.n_points,
        no_data=args.no_data,
        seed=args.random_seed
    )
