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

With Git installed, :code:`cd` to an install location then run the following lines

.. code-block:: bash

   git clone https://github.com/clcr/pyeo.git
   cd pyeo
   python setup.py install

If you want to edit this code, run

.. code-block:: bash

   python setup.py devel

instead of :code:`install`

To verify the installation, open a Python prompt and type

>>> import pyeo

You should get no errors.

Function reference
==================

.. module:: pyeo.core

At present, all processing code is located in pyeo/core.py.
A small test suite is located in pyeo/tests/pyeo_tests.py; this is designed for use with py.test.
Some example applications and demos are in pyeo/apps; for an illustration of the use of the library,
pyeo/apps/change_detection/simple_s2_change_detection.py is recommended.


File structure and logging
--------------------------

 .. autofunction:: init_log

 .. autofunction:: create_file_structure

 .. autofunction:: read_aoi


Sentinel 2 data acquisition
---------------------------

 .. autofunction:: check_for_new_s2_data

 .. autofunction:: download_new_s2_data

 .. autofunction:: sent2_query (from geospatial_learn)


Planet data acquisition
-----------------------

 .. autofunction:: planet_query

 .. autofunction:: build_search_request

 .. autofunction:: do_quick_search

 .. autofunction:: do_saved_search

 .. autofunction:: activate_and_dl_planet_item


Sentinel 2 preprocessing
------------------------

 .. autofunction:: apply_sen2cor

 .. autofunction:: atmospheric_correction

 .. autofunction:: sort_by_timestamp

 .. autofunction:: get_image_acquisition_time

 .. autofunction:: open_dataset_from_safe

 .. autofunction:: aggregate_and_mask_10m_bands

 .. autofunction:: stack_sentinel_2_bands

 .. autofunction:: get_sen_2_image_timestamp


Raster processing
-----------------

 .. autofunction:: create_matching_dataset

 .. autofunction:: create_new_stacks

 .. autofunction:: stack_old_and_new_images

 .. autofunction:: stack_images

 .. autofunction:: get_raster_bounds

 .. autofunction:: get_raster_size

 .. autofunction:: resample_image_in_place


Geometry processing
-------------------

 .. autofunction:: get_combined_polygon

 .. autofunction:: multiple_union

 .. autofunction:: multiple_intersection

 .. autofunction:: write_polygon

 .. autofunction:: check_overlap

 .. autofunction:: get_aoi_bounds

 .. autofunction:: get_aoi_size

 .. autofunction:: get_poly_size


Raster/Geometry interactions
----------------------------

 .. autofunction:: pixel_bounds_from_polygon

 .. autofunction:: point_to_pixel_coordinates

 .. autofunction:: stack_and_trim_images

 .. autofunction:: clip_raster

 .. autofunction:: get_aoi_intersection

 .. autofunction:: get_raster_intersection

 .. autofunction:: get_poly_intersection

 .. autofunction:: create_new_image_from_polygon

 .. autofunction:: get_local_top_left


Masking functions
-----------------

 .. autofunction:: create_cloud_mask

 .. autofunction:: create_mask_from_confidence_layer

 .. autofunction:: get_mask_path

 .. autofunction:: combine_masks

 .. autofunction:: apply_array_image_mask


Machine learning functions
--------------------------

 .. autofunction:: classify_image

 .. autofunction:: reshape_raster_for_ml

 .. autofunction:: reshape_ml_out_to_raster

 .. autofunction:: reshape_prob_out_to_raster

 .. autofunction:: create_trained_model

 .. autofunction:: create_model_for_region

 .. autofunction:: get_training_data


Exception objects
-----------------

 .. autoclass:: ForestSentinelException

 .. autoclass:: StackImageException

 .. autoclass:: CreateNewStacksException

 .. autoclass:: TooManyRequests


Example scripts
===============

simple_s2_change_detection.py
-----------------------------
 .. automodule:: pyeo.apps.change_detection.simple_s2_change_detection

create_model_from_shapefile_and_raster.py
------------------------------
 .. automodule:: pyeo.apps.model_creation.create_model_from_shapefile_and_raster