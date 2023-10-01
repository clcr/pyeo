"""
Run ACD National
----------------
An application that runs the raster and vector pipeline for all tiles intersecting with a Region of Interest.

The application runs using an initialisation file, which provides all the parameters PyEO needs to make decisions. See SEPAL pipeline training notebook within the notebooks folder on the GitHub Repository for an explanation of the initialisation file.

The raster pipeline, depending on True/False parameters provided in the initialisation file, performs the following:

- Takes a Region of Interest (ROI) and calculates which Sentinel-2 (S2) tiles overlap with the ROI.
- Builds a Baseline Composite to compare land cover changes against, by downloading S2 images and calculating the median of these images.
- Downloads images over the Change Period
- Classifies the Composite and the Change images using a classifier in ./models/
- Calculates the change between the from classes and the to classes, for each classified image. This could be changes from forest to bare soil.
- Creates a Change Report describing the consistency of the class changes, highlighting the changes that PyEO is confident.

The vector pipline, depending on the True/False parameters provided in the initialisation file, performs the following:

- Vectorises the Change Report, removing any changes observed outside of the ROI.
- Performs Zonal Statistics on the change polygons.
- Filters the change polygons based on Counties of Interest.


Example call:

::

    $python pyeo/run_acd_national.py path/to/pyeo_linux.ini

:code:`path/to/pyeo_linux.ini` needs to be an absolute path, and works for all the OS .ini files e.g. :code:`pyeo_windows.ini` , :code:`pyeo_sepal.ini` .
"""

import sys
from pyeo import acd_national

if __name__ == "__main__":
    # if run from terminal, __name__ becomes "__main__"
    # sys.argv[0] is the name of the python script, e.g. acd_national.py

    acd_national.automatic_change_detection_national(sys.argv[1])
