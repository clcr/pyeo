.. Pyeo documentation master file, created by
   sphinx-quickstart on Wed Aug 29 11:27:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pyeo's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Introduction
============

Python For Earth Observation is a collection of functions for downloading, manipulating, combining and classifying
geospatial raster and vector data. It is intended to require a minimum of dependencies - most functions only require
the basic GDAL/OGR/OSR stack and Numpy.


Installation
============

With Git and Miniconda or Anaconda installed, :code:`cd` to an install location then run the following lines

.. code-block:: bash

   git clone https://github.com/clcr/pyeo.git
   cd pyeo
   conda env create --file environment.yml --name pyeo_env
   conda activate pyeo_env
   python -m pip install . -vv

In a Python prompt, try  :code:`import pyeo` - you should see no errors.

Quick start
===========
 Before you start, you will need:

-A linux (or maybe mac, untested) machine
-A raster of your window area
-A shapefile of polygons over your training areas with a field containing class labels
-A raster to classify. This can be the same as your original raster.
All rasters and shapefiles should be in the same projection; ideally in the local projection of your satellite data.
-Git
-Anaconda/Miniconda

INSTALLATION

git clone https://github.com/clcr/pyeo
cd pyeo
git checkout restructure      # This gets you the present working branch
conda env create --file environment.yml --name pyeo_env
echo export PYEO=$(pwd) >> ~/.bashrc    #Or just append to the end of your .bashrc
echo export PATH=$PATH:$PYEO/bin
chmod 755 bin/*

USE

conda activate pyeo_env
cd (whereever your data is)
extract_signatures your_raster.tif your_shapefile.shp output.csv
# You can now explore your signatures as a .csv file, and perform any interim processing
create_model_from_signatures output.csv model.pkl
classify_image your_raster model.pkl output_image.tif

Assumptions and design decisions
=====================================

###Rasters ###

- Can read from any gdal-readable format
- Stores internally as geotiff
- Named with the .tif suffix
- Most raster processing is done using numpy and getVirtualMemArray
- Unless otherwise stated, **all rastesrs are assumed to be in a projected coordinate system** - i.e. in meters.
  Functions may fail if passed a raster in lat-long projection

### Masks ###

- Some Pyeo functions include options for applying masks
- A raster may have an associated mask
- A mask is a geotif with an identical name as the raster it's masking with a .msk extension
   - For example, the mask for my_sat_image.tif is my_sat_image.msk
- A mask is
   - a single band raster
   - of identical height, width and resolution of the related image
   - contains values 0 or 1
- A mask is applied by multiplying it with each band of its raster
   - So any pixel with a 0 in its mask will be removed, and a 1 will be kept

### Timestamps ###

- Pyeo uses the same timestamp convention as ESA: yyyymmddThhmmss
   - For example, 1PM on 27th December 2020 would be 20201227T130000
- All timestamps are in UTC

### Models ###
- All models are serialised and deserialised using joblib.dump or joblib.load




Function reference
==================

.. automodule:: pyeo.classification
   :members:

.. automodule:: pyeo.array_utilities
   :members:

.. automodule:: pyeo.coordinate_manipulation
   :members:

.. automodule:: pyeo.filesystem_utilities
   :members:

.. automodule:: pyeo.queries_and_downloads
   :members:

.. automodule:: pyeo.raster_manipulation
   :members:

.. automodule:: pyeo.validation
   :members:


A small test suite is located in pyeo/tests/pyeo_tests.py; this is designed for use with py.test.
Some example applications and demos are in pyeo/apps; for an illustration of the use of the library,
pyeo/apps/change_detection/simple_s2_change_detection.py is recommended.

Applications
============

.. automodule:: pyeo.apps.change_detection.image_comparison

.. automodule:: pyeo.apps.change_detection.rolling_composite_s2_change_detection

.. automodule:: pyeo.apps.change_detection.simple_s2_change_detection

.. automodule:: pyeo.apps.masking.filter_by_class_map

.. automodule:: pyeo.apps.model_creation.create_model_from_region


