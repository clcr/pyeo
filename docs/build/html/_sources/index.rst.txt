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
the basic GDAL/OGR/OSR stack.


Installation
============

With Git and Miniconda or Anaconda installed, :code:`cd` to an install location then run the following lines

.. code-block:: bash

   git clone https://github.com/clcr/pyeo.git
   cd pyeo
   conda env create --file environment.yml --name pyeo_env
   conda activate pyeo_env


Including Pyeo in your own code
===============================

Include the following lines at the start of your Python scripts:

.. code-block:: python

   import sys
   sys.path.append("/path/to/pyeo")
   import pyeo.core as pyeo

You may see a warning about scikit versions; this is normal.

Filenaming, assumptions and structure
=====================================

Pyeo is divided into high-level and low-level fucntions.

Function reference
==================

.. automodule:: pyeo.classification
   :members:

.. automodule:: pyeo.array_utilities
   :members:

.. automodule:: pyeo.coordinate_manipulation
   :members:

.. automodule:: pyeo.filesystems_utilities
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


