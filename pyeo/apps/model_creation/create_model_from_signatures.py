from pyeo.classification import create_model_from_signatures
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a pixel classification model from a signature file produces from "
                                     "extract_signatues.")
    parser.add_argument("sig_file", help="Path to the .csv file containing the signatures. The first column should"
                                         "be a set of class labels, with the rest of the columns being the pixel"
                                         "values for each band connected to that class.")
    parser.add_argument("out_path", help="File to save the pickeled ML model to.")
    args = parser.parse_args()

    create_model_from_signatures(args.sig_file, args.out_path)
