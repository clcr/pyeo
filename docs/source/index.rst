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

Function reference
==================

.. module:: pyeo.core

At present, all processing code is located in pyeo/core.py.
A small test suite is located in pyeo/tests/pyeo_tests.py; this is designed for use with py.test.
Some example applications and demos are in pyeo/apps; for an illustration of the use of the library,
pyeo/apps/change_detection/simple_s2_change_detection.py is recommended.

Conven


File structure and logging
--------------------------

 .. autofunction:: init_log

 .. autofunction:: create_file_structure

 .. autofunction:: read_aoi
 
 .. autofunction:: clean_aoi
 
 .. autofunction:: get_preceding_image_path
 
 .. autofunction:: get_pyeo_timestamp
 
 .. autofunction:: is_tif
 
 .. autofunction:: load_api_key
 
 .. autofunction:: read_geojson


Sentinel 2 data acquisition
---------------------------

 .. autofunction:: check_for_new_s2_data

 .. autofunction:: download_new_s2_data

 .. autofunction:: sent2_query (from geospatial_learn)
 
 .. autofunction:: check_for_s2_data_by_date


Planet data acquisition
-----------------------

 .. autofunction:: download_planet_image_on_day

 .. autofunction:: planet_query

 .. autofunction:: build_search_request

 .. autofunction:: do_quick_search

 .. autofunction:: do_saved_search

 .. autofunction:: activate_and_dl_planet_item
 
 .. autofunction:: get_planet_product_path

 


Data source download functions
------------------------------

 .. autofunction:: download_blob_from_google
 
 .. autofunction:: download_from_google_cloud
 
 .. autofunction:: download_from_scihub


Sentinel 2 preprocessing
------------------------

 .. autofunction:: apply_sen2cor
 
 .. autofunction:: apply_fmask

 .. autofunction:: atmospheric_correction

 .. autofunction:: sort_by_timestamp

 .. autofunction:: get_image_acquisition_time

 .. autofunction:: open_dataset_from_safe

 .. autofunction:: preprocess_sen2_images

 .. autofunction:: stack_sentinel_2_bands

 .. autofunction:: get_sen_2_image_timestamp
 
 .. autofunction:: check_for_invalid_l1_data
 
 .. autofunction:: check_for_invalid_l2_data
 
 .. autofunction:: clean_l2_data
 
 .. autofunction:: clean_l2_dir
 
 .. autofunction:: get_l1_safe_file
 
 .. autofunction:: get_l2_safe_file
 
 .. autofunction:: get_sen_2_granule_id
 
 .. autofunction:: get_sen_2_image_orbit
 
 .. autofunction:: get_sen_2_image_tile
 
 .. autofunction:: get_sen_2_tiles


Raster processing
-----------------

 .. autofunction:: create_matching_dataset

 .. autofunction:: create_new_stacks

 .. autofunction:: stack_old_and_new_images

 .. autofunction:: stack_images
 
 .. autofunction:: stack_image_with_composite

 .. autofunction:: get_raster_bounds
 
 .. autofunction:: get_raster_size

 .. autofunction:: resample_image_in_place

 .. autofunction:: composite_images_with_mask

 .. autofunction:: composite_directory
 
 .. autofunction:: filter_by_class_map
 
 .. autofunction:: mosaic_images

 .. autofunction:: raster_reclass_binary
 
 .. autofunction:: raster_sum
 
 .. autofunction:: raster_to_array
 
 .. autofunction:: reproject_directory
 
 .. autofunction:: reproject_image
 
 .. autofunction:: reproject_geotransform 
 
 

Geometry processing
-------------------

 .. autofunction:: get_combined_polygon

 .. autofunction:: multiple_union

 .. autofunction:: multiple_intersection

 .. autofunction:: write_geometry

 .. autofunction:: check_overlap

 .. autofunction:: get_aoi_bounds

 .. autofunction:: get_aoi_size

 .. autofunction:: get_poly_size
 
 .. autofunction:: get_poly_bounding_rect


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
 
 .. autofunction:: align_bounds_to_whole_number
 
 .. autofunction:: floor_to_resolution


Masking functions
-----------------

 .. autofunction:: create_mask_from_model

 .. autofunction:: create_mask_from_confidence_layer
 
 .. autofunction:: create_mask_from_class_map
 
 .. autofunction:: create_mask_from_fmask
 
 .. autofunction:: create_mask_from_sen2cor_and_fmask

 .. autofunction:: get_mask_path

 .. autofunction:: combine_masks
 
 .. autofunction:: get_masked_array

 .. autofunction:: apply_array_image_mask
 
 .. autofunction:: buffer_mask_in_place
 
 .. autofunction:: project_array


Machine learning functions
--------------------------

 .. autofunction:: classify_image
 
 .. autofunction:: classify_directory
 
 .. autofunction:: change_from_composite

 .. autofunction:: reshape_raster_for_ml

 .. autofunction:: reshape_ml_out_to_raster

 .. autofunction:: reshape_prob_out_to_raster

 .. autofunction:: create_trained_model

 .. autofunction:: create_model_for_region
 
 .. autofunction:: create_model_from_signatures

 .. autofunction:: get_training_data
 
 .. autofunction:: flatten_probability_image
 
 .. autofunction:: autochunk


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

composite_directory.py
----------------------
 .. automodule:: pyeo.apps.subprocessing.composite_directory

extract_signatures.py
---------------------
 .. automodule:: pyeo.apps.subprocessing.extract_signatures
