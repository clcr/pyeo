"""Removes all but the most recent n images from the filetree"""

import pyeo.core as pyeo
import os
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
        remove_latest_images(to_be_cleaned, images_to_keep, with_warning)


def remove_latest_images(dir, images_to_keep, with_warning=True):
    images = pyeo.sort_by_timestamp(os.listdir(dir))
    if with_warning:
        if not input(
                "About to delete {} files from {}: Y/N?".format(len(images[images_to_keep:]), dir)).upper().startswith(
                "Y"):
            return
    for image in images[images_to_keep:]:
        if os.path.isfile(image):
            os.remove(image)
        else:
            shutil.rmtree(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Removes all but the latest n interim images from a given aoi dir.")
    parser.add_argument("aoi_dir", help="The root of the directory containing the composite, images and "
                                        "output folders")
    parser.add_argument("images_to_keep", type=int, help="The number of images to keep")
    parser.add_argument("do_warning", type=bool, default=True, help="If false, skips the warning. Use with care")
    # and if you're reading this, don't come crying to me if it goes wrong.

    args = parser.parse_args()
    free_space(args.aoi_dir, args.images_to_keep, args.do_warning)
