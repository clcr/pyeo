"""A very small plotting library."""

import gdal
from matplotlib import pyplot as plt


def show_satellite_image(image_path):
    """Uses matplotlib.imshow() to preview a satellite image at image_path. Assumes image is bgr (ie quicklook)"""
    img = gdal.Open(image_path)
    array = img.GetVirtualMemArray()
    if len(array.shape) >= 3:
        img_view = array.transpose([1,2,0])
    else:
        img_view = array
    plt.imshow(img_view)
    img_view = None
    array = None
    img = None