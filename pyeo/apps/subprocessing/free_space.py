"""Removes all but the most recent n images from the filetree"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import shutil
import argparse


def free_space(aoi_dir, images_to_keep, with_warning=True):
    directory_list = [
        "images/L1/",
        "images/L2/",
        "images/merged/",
        "images/stacked/",
        "images/planet/",
        "composite/",
        "composite/L1",
        "composite/L2",
        "composite/merged"
    ]
    for directory in directory_list:
        to_be_cleaned = os.path.join(aoi_dir, directory)
        remove_old_images(to_be_cleaned, images_to_keep, with_warning)


def remove_old_images(image_dir, images_to_keep, with_warning=True):
    """Removes all but the latest images from image_dir."""
    images = pyeo.sort_by_timestamp(os.listdir(image_dir))
    if with_warning:
        if not input(
                "About to delete {} files from {}: Y/N?".format(len(images[images_to_keep:]), image_dir)).upper().\
                startswith("Y"):
            return
    for image_name in images[images_to_keep:]:
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            os.remove(image_path)
        else:
            shutil.rmtree(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Removes all but the latest n interim images from a given aoi dir.")
    parser.add_argument("aoi_dir", help="The root of the directory containing the composite, images and "
                                        "output folders")
    parser.add_argument("images_to_keep", type=int, help="The number of images to keep")
    parser.add_argument("--do_warning", type=bool, default=True, help="If false, skips the warning. Use with care")
    # and if you're reading this, don't come crying to me if it goes wrong.

    args = parser.parse_args()
    free_space(args.aoi_dir, args.images_to_keep, args.do_warning)
