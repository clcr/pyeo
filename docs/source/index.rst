****
Pyeo
****

.. toctree::
   :caption: Contents:

   classification
   coordinate_manipulation
   queries_and_downloads
   raster_manipulation
   filesystem_utilities
   validation
   scripts

Introduction
############

Python For Earth Observation is a collection of functions for downloading, manipulating, combining and classifying
geospatial raster and vector data.


Installation
############

With Git and Miniconda or Anaconda installed, :code:`cd` to an install location then run the following lines

.. code-block:: bash

   git clone https://github.com/clcr/pyeo.git
   cd pyeo
   conda env create --file environment.yml --name pyeo_env
   conda activate pyeo_env
   python -m pip install . -vv

In a Python prompt, try  :code:`import pyeo` - you should see no errors.

Quick start
###########
Before you start, you will need:

* Git
* Anaconda/Miniconda
* A raster of your window area
* A shapefile of polygons over your training areas with a field containing class labels
* A raster to classify. This can be the same as your original raster.

  * All rasters and shapefiles should be in the same projection; ideally in the local projection of your satellite data.


Use
***
You can use Pyeo's command-line functions to create and apply a pixel classification model from a set of polygons
and a raster. The below example:

* saves the training data defined in :code:`your_raster.tif` and :code:`your_shapefile.tif` into :code:`signatures.csv`
* creates a model from :code:`signatures.csv` named :code:`model.pkl`
* Classifies the whole of :code:`your_raster.tif` using :code:`model.pkl`, and saves the result into :code:`output_image.tif`

.. code-block:: bash

   conda activate pyeo_env
   extract_signatures your_raster.tif your_shapefile.shp signatures.csv
   create_model_from_signatures signatures.csv model.pkl
   classify_image your_raster model.pkl output_image.tif

A small test suite is located in pyeo/tests/pyeo_tests.py; this is designed for use with py.test.
Some example applications and demos are in pyeo/apps; for an illustration of the use of the library,
pyeo/apps/change_detection/simple_s2_change_detection.py is recommended.


