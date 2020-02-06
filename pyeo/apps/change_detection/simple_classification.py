from pyeo.classification import classify_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Applies a model to an image")
    parser.add_argument("image", help="The image to classify")
    parser.add_argument("model", help="The model to use to classify the image.")
    parser.add_argument("output", help="Where to store the classified .tif")
    parser.add_argument("--use_mask", action="store_true", help="If present, apply mask.")
    args = parser.parse_args()

    classify_image(args.image, args.model, args.output, apply_mask=args.use_mask)
