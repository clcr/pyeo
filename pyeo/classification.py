"""
Contains every function to do with map classification. This includes model creation, map classification and processes for array manipulation into scikit-learn compatible forms.

For details on how to build a class shapefile, see the notebook :code:`PyEO_sepal_model_training.ipynb` within the notebooks directory in the PyEO GitHub.

All models are serialised and deserialised using :code:`joblib.dump` or :code:`joblib.load`, and saved with the .pkl
extension.

Key functions
-------------

:py:func:`classify_model_for_region` Creates a model from a directory of class shapefile and .tif pairs. The benefit of this function is that a model can be produced from multiple regions, increasing the generalisation ability of the model.

:py:func:`create_trained_model` Creates a model from a class shapefile and a .tif

Alternatively, these two functions are suitable for those wishing to create a simpler model:

1. :py:func:`extract_features_to_csv` Extracts class signatures from a class shapefile and a .tif

2. :py:func:`create_model_from_signatures` Creates a model from a .csv of classes and band signatures

Finally, a raster can be classified using:

:py:func:`classify_image` Produces a classification map from an image using a model.

Function reference
------------------
"""
import csv
import glob
import logging
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from osgeo import gdalconst
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import sparse as sp
import shutil
from sklearn import ensemble as ens

# I.R.
# from sklearn.externals import joblib as sklearn_joblib
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn import metrics
import sys

from pyeo.coordinate_manipulation import get_local_top_left
from pyeo.filesystem_utilities import get_mask_path
from pyeo.raster_manipulation import (
    stack_images,
    create_matching_dataset,
    apply_array_image_mask,
    get_masked_array,
)
import pyeo.windows_compatability

gdal.UseExceptions()

log = logging.getLogger(__name__)


def change_from_composite(
    image_path: str,
    composite_path: str,
    model_path: str,
    class_out_path: str,
    prob_out_path: str = None,
    skip_existing: bool = False,
    apply_mask: bool = False,
) -> None:
    """
    Stacks an image with a composite and classifies each pixel change with a scikit-learn model.

    The image that is classified has the following bands:

    1.  1. composite blue
    2.  2. composite green
    3.  3. composite red
    4.  4. composite IR
    5.  5. image blue
    6.  6. image green
    7.  7. image red
    8.  8. image IR

    Parameters
    ----------
    image_path : str
        The path to the image
    composite_path : str
        The path to the composite
    model_path : str
        The path to a .pkl of a scikit-learn classifier that takes 8 features
    class_out_path : str
        A location to save the resulting classification .tif
    prob_out_path : str, optional
        A location to save the probability raster of each pixel.
    skip_existing : bool, optional
        If true, do not run if class_out_path already exists. Defaults to False.
    apply_mask : bool, optional
        If True, uses the .msk file corresponding to the image at image_path to skip any invalid pixels. Default False.
    
    Returns
    -------
    None
    """

    if skip_existing:
        if os.path.exists(class_out_path):
            log.info(" Classified image exists. Skipping. {}".format(class_out_path))
            return
    if os.path.exists(composite_path):
        if os.path.exists(image_path):
            with TemporaryDirectory(dir=os.getcwd()) as td:
                stacked_path = os.path.join(td, "comp_stack.tif")
                log.info("stacked path: {}".format(stacked_path))
                stack_images([composite_path, image_path], stacked_path)
                log.info(
                    " stacked path exists? {}".format(os.path.exists(stacked_path))
                )
                classify_image(
                    stacked_path,
                    model_path,
                    class_out_path,
                    prob_out_path,
                    apply_mask,
                    skip_existing,
                )
                log.info(
                    " class out path exists? {}".format(os.path.exists(class_out_path))
                )
                return
        else:
            log.error("File not found: {}".format(image_path))
    else:
        log.error("File not found: {}".format(composite_path))
    return


def classify_image(
    image_path: str,
    model_path: str,
    class_out_path: str,
    prob_out_path: str = None,
    apply_mask: bool = False,
    out_format: str = "GTiff",
    chunks: int = 4,
    nodata: int = 0,
    skip_existing: bool = False,
) -> str:
    """

    Produces a class map from a raster and a model.

    This applies the model's fit() function to each pixel in the input raster, and saves the result into an output
    raster. The model is presumed to be a scikit-learn fitted model created using one of the other functions in this
    library (:py:func:`create_model_from_signatures` or :py:func:`create_trained_model`).

    Parameters
    ----------
    image_path : str
        The path to the raster image to be classified.

    model_path : str
        The path to the .pkl file containing the model.

    class_out_path : str
        The path that the classified map will be saved at.

    prob_out_path : str, optional
        If present, the path that the class probability map will be stored at. Default None
        
    apply_mask : bool, optional
        If True, uses the .msk file corresponding to the image at image_path to skip any invalid pixels. Default False.

    out_type : str, optional
        The raster format of the class image. Defaults to "GTiff" (geotif). See gdal docs for valid types.

    chunks : int, optional
        The number of chunks the image is broken into prior to classification. The smaller this number, the faster
        classification will run - but the more likely you are to get a outofmemory error. Default 10.

    nodata : int, optional
        The value to write to masked pixels. Defaults to 0.

    skip_existing : bool, optional
        If true, do not run if class_out_path already exists. Defaults to False.

    Returns
    -------
    class_out_path : str
        The output path for the classified image.

    Notes
    -----
    If you want to create a custom model, the object is presumed to have the following methods and attributes:

    -   - model.n_classes_ : the number of classes the model will produce
    -   - model.n_cores : The number of CPU cores used to run the model
    -   - model.predict() : A function that will take a set of band inputs from a pixel and produce a class.
    -   - model.predict_proba() : If called with prob_out_path, a function that takes a set of n band inputs from a pixel and produces :code:`n_classes_` outputs corresponding to the probabilties of a given pixel being that class

    """

    if skip_existing:
        log.info("Checking for existing classification {}".format(class_out_path))
        if os.path.isfile(class_out_path):
            try:
                f = gdal.Open(class_out_path)
                f = None
                log.info("Class image exists and is readable - skipping.")
                return class_out_path
            except:
                log.info(
                    "Class image exists but is not readable - deleting and redoing: {}".format(
                        class_out_path
                    )
                )
                os.remove(class_out_path)
    if not os.path.exists(image_path):
        log.error("File not found: {}".format(image_path))
    if not os.path.exists(model_path):
        log.error("File not found: {}".format(model_path))
    log.info("Classifying file: {}".format(image_path))
    log.info("Saved model     : {}".format(model_path))
    try:
        image = gdal.Open(image_path)
    except RuntimeError as e:
        log.info("Exception: {}".format(e))
        exit(1)
    if chunks == None:
        log.info("No chunk size given, attempting autochunk.")
        chunks = autochunk(image)
        log.info("Autochunk to {} chunks".format(chunks))
    try:
        # I.R.
        # model = sklearn_joblib.load(model_path)
        model = joblib.load(model_path)
    except KeyError as e:
        # log.warning("Sklearn joblib import failed,trying generic joblib")
        log.warning("KeyError: joblib import failed: {}".format(e))
        # model = joblib.load(model_path)
    except TypeError as e:
        log.warning("TypeError: joblib import failed: {}".format(e))
        # log.warning("Sklearn joblib import failed,trying generic joblib: {}".format(e))
        # model = joblib.load(model_path)
    with TemporaryDirectory(dir=os.getcwd()) as td:
        class_out_temp = os.path.join(td, os.path.basename(class_out_path))
        class_out_image = create_matching_dataset(
            image, class_out_temp, format=str(out_format), datatype=gdal.GDT_Byte
        )
        if prob_out_path:
            try:
                log.info("n classes in the model: {}".format(model.n_classes_))
            except AttributeError as e:
                log.warning(
                    "Model has no n_classes_ attribute (known issue with GridSearch): {}".format(
                        e
                    )
                )
            prob_out_temp = os.path.join(td, os.path.basename(prob_out_path))
            prob_out_image = create_matching_dataset(
                image, prob_out_temp, bands=model.n_classes_, datatype=gdal.GDT_Float32
            )
        model.n_cores = -1
        image_array = image.GetVirtualMemArray()
        if apply_mask:
            mask_path = get_mask_path(image_path)
            # log.info("Applying mask at {}".format(mask_path))
            mask = gdal.Open(mask_path)
            mask_array = mask.GetVirtualMemArray()
            image_array = apply_array_image_mask(image_array, mask_array)
            mask_array = None
            mask = None
        # Mask out missing values from the classification
        # at this point, image_array has dimensions [band, y, x]
        image_array = reshape_raster_for_ml(image_array)
        # Now it has dimensions [x * y, band] as needed for Scikit-Learn

        # Determine where in the image array there are no missing values in any of the bands (axis 1)
        n_samples = image_array.shape[0]  # gives x * y dimension of the whole image
        nbands = image_array.shape[1]  # gives number of bands
        boo = np.where(image_array[:, 0] != nodata, True, False)
        if nbands > 1:
            for band in range(1, nbands, 1):
                boo1 = np.where(image_array[:, band] != nodata, True, False)
                boo = np.logical_and(boo, boo1)
        good_indices = np.where(boo)[0]  # get indices where all bands contain data
        good_sample_count = np.count_nonzero(boo)
        log.info(
            "Proportion of non-missing values: {:3.2f}%".format(
                good_sample_count / n_samples * 100
            )
        )
        good_samples = np.take(image_array, good_indices, axis=0).squeeze()
        n_good_samples = len(good_samples)
        classes = np.full(n_good_samples, nodata, dtype=np.ubyte)
        if prob_out_path:
            probs = np.full(
                (n_good_samples, model.n_classes_), nodata, dtype=np.float32
            )
        chunk_size = int(n_good_samples / chunks)
        chunk_resid = n_good_samples - (chunk_size * chunks)
        log.info(
            "   Number of chunks {} Chunk size {} Chunk residual {}".format(
                chunks, chunk_size, chunk_resid
            )
        )
        # The chunks iterate over all values in the array [x * y, bands] always with all bands per chunk
        for chunk_id in range(chunks):
            offset = chunk_id * chunk_size
            if chunk_id == chunks - 1:
                chunk_size = chunk_size + chunk_resid
            log.info(
                "   Classifying chunk {} of size {}".format(chunk_id + 1, chunk_size)
            )
            chunk_view = good_samples[offset : offset + chunk_size]
            indices_view = good_indices[offset : offset + chunk_size]
            out_view = classes[offset : offset + chunk_size]
            chunk_view = (
                chunk_view.copy()
            )  # bug fix for Pandas bug: https://stackoverflow.com/questions/53985535/pandas-valueerror-buffer-source-array-is-read-only
            indices_view = (
                indices_view.copy()
            )  # bug fix for Pandas bug: https://stackoverflow.com/questions/53985535/pandas-valueerror-buffer-source-array-is-read-only
            out_view[:] = model.predict(chunk_view)
            if prob_out_path:
                log.info("   Calculating probabilities")
                prob_view = probs[offset : offset + chunk_size, :]
                prob_view[:, :] = model.predict_proba(chunk_view)
        class_out_array = np.full((n_samples), nodata)
        for i, class_val in zip(good_indices, classes):
            class_out_array[i] = class_val
        class_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[
            :, :
        ] = reshape_ml_out_to_raster(
            class_out_array, image.RasterXSize, image.RasterYSize
        )
        if prob_out_path:
            prob_out_array = np.full((n_samples, model.n_classes_), nodata)
            for i, prob_val in zip(good_indices, probs):
                prob_out_array[i] = prob_val
            prob_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[
                :, :, :
            ] = reshape_prob_out_to_raster(
                prob_out_array, image.RasterXSize, image.RasterYSize
            )
            prob_out_image = None
            prob_out_array = None
            shutil.move(prob_out_temp, prob_out_path)
        class_out_image = None
        class_out_array = None
        shutil.move(class_out_temp, class_out_path)
    # verify that the output file(s) have been created
    if not os.path.exists(class_out_path):
        log.error("Classification output file not found: {}".format(class_out_path))
    else:
        log.info("Created classification image file: {}".format(class_out_path))
    if prob_out_path:
        if not os.path.exists(prob_out_path):
            log.error("Probability output file not found: {}".format(prob_out_path))
        else:
            log.info("Created probability image file: {}".format(prob_out_path))
        return class_out_path, prob_out_path
    else:
        return class_out_path

def classify_image_and_composite(
    image_path: str,
    composite_path,
    model_path,
    class_out_path,
    prob_out_path=None,
    apply_mask=False,
    out_type="GTiff",
    num_chunks=10,
    nodata=0,
    skip_existing=False,
):
    """
    :meta private:

    !!! WARNING - currently does nothing. Not successfully tested yet. Use change_from_composite() function instead.

    Produces a class map from a raster file, a composite raster file and a model.
    This applies the model's fit() function to each pixel in the input raster, and saves the result into an output
    raster. The model is presumed to be a scikit-learn fitted model created using one of the other functions in this
    library (:py:func:`create_model_from_signatures` or :py:func:`create_trained_model`).

    Parameters
    ----------
    image_path : str
        The path to the raster image to be classified.
    composite_path : str
        The path to the raster image composite to be used as a baseline.
    model_path : str
        The path to the .pkl file containing the model
    class_out_path : str
        The path that the classified map will be saved at.
    prob_out_path : str, optional
        If present, the path that the class probability map will be stored at. Default None
    apply_mask : bool, optional
        If True, uses the .msk file corresponding to the image at image_path to skip any invalid pixels. Default False.
    out_type : str, optional
        The raster format of the class image. Defaults to "GTiff" (geotif). See gdal docs for valid types.
    num_chunks : int, optional
        The number of chunks the image is broken into prior to classification. The smaller this number, the faster
        classification will run - but the more likely you are to get a outofmemory error. Default 10.
    nodata : int, optional
        The value to write to masked pixels. Defaults to 0.
    skip_existing : bool, optional
        If true, do not run if class_out_path already exists. Defaults to False.


    Notes
    -----
    If you want to create a custom model, the object is presumed to have the following methods and attributes:

       - model.n_classes_ : the number of classes the model will produce
       - model.n_cores : The number of CPU cores used to run the model
       - model.predict() : A function that will take a set of band inputs from a pixel and produce a class.
       - model.predict_proba() : If called with prob_out_path, a function that takes a set of n band inputs from a pixel
                                and produces n_classes_ outputs corresponding to the probabilties of a given pixel being
                                that class
    

    if skip_existing:
        log.info("Checking for existing classification {}".format(class_out_path))
        if os.path.isfile(class_out_path):
            log.info("Class image exists, skipping.")
            return class_out_path
    log.info("Classifying file: {}".format(image_path))
    log.info("Saved model     : {}".format(model_path))
    image = gdal.Open(image_path)
    composite = gdal.Open(composite_path)
    if num_chunks == None:
        log.info("No chunk size given, attempting autochunk.")
        num_chunks = autochunk(image)
        log.info("Autochunk to {} chunks".format(num_chunks))
    try:
        # I.R.
        # model = sklearn_joblib.load(model_path)
        model = joblib.load(model_path)
    except KeyError as e:
        # log.warning("Sklearn joblib import failed,trying generic joblib")
        log.warning("KeyError: joblib import failed: {}".format(e))
        # model = joblib.load(model_path)
    except TypeError as e:
        log.warning("TypeError: joblib import failed: {}".format(e))
        # log.warning("Sklearn joblib import failed,trying generic joblib: {}".format(e))
        # model = joblib.load(model_path)
    class_out_image = create_matching_dataset(image, class_out_path, format=out_type, datatype=gdal.GDT_Byte)
    log.info("Created classification image file: {}".format(class_out_path))
    if prob_out_path:
        try:
            log.info("n classes in the model: {}".format(model.n_classes_))
        except AttributeError:
            log.warning("Model has no n_classes_ attribute (known issue with GridSearch)")
        prob_out_image = create_matching_dataset(image, prob_out_path, bands=model.n_classes_, datatype=gdal.GDT_Float32)
        log.info("Created probability image file: {}".format(prob_out_path))
    model.n_cores = -1
    image_array = image.GetVirtualMemArray()
    composite_array = composite.GetVirtualMemArray()

    if apply_mask:
        mask_path = get_mask_path(image_path)
        log.info("Applying mask at {}".format(mask_path))
        mask = gdal.Open(mask_path)
        mask_array = mask.GetVirtualMemArray()
        image_array = apply_array_image_mask(image_array, mask_array)
        mask_array = None
        mask = None

    # Mask out missing values from the classification
    # at this point, image_array has dimensions [band, y, x]
    image_array = reshape_raster_for_ml(image_array)
    composite_array = reshape_raster_for_ml(composite_array)
    # Now it has dimensions [x * y, band] as needed for Scikit-Learn
    log.info("Shape of composite array = {}".format(composite_array.shape))
    log.info("Shape of image array = {}".format(image_array.shape))

    # Determine where in the image array there are no missing values in any of the bands (axis 1)
    n_samples = image_array.shape[0]  # gives x * y dimension of the whole image
    good_mask = np.all(image_array != nodata, axis=1)
    good_sample_count = np.count_nonzero(good_mask)
    log.info("Number of good pixel values: {}".format(good_sample_count))
    if good_sample_count > 0:
        #TODO: if good_sample_count <= 0.5*len(good_mask):  # If the images is less than 50% good pixels, do filtering
        if 1 == 0:  # Removing the filter until we fix the classification issue with it
            log.info("Filtering nodata values")
            good_indices = np.nonzero(good_mask)
            good_samples = np.take(image_array, good_indices, axis=0).squeeze()
            n_good_samples = len(good_samples)
            log.info("Number of pixel values to be classified: {}".format(n_good_samples))
        else:
            #log.info("Not worth filtering nodata, skipping.")
            good_samples = np.concatenate((composite_array, image_array), axis=1)
            good_indices = range(0, n_samples)
            n_good_samples = n_samples
            log.info("Number of pixel values to be classified: {}".format(n_good_samples))
        log.info("Shape of good samples array = {}".format(good_samples.shape))
        classes = np.full(n_good_samples, nodata, dtype=np.ubyte)
        if prob_out_path:
            probs = np.full((n_good_samples, model.n_classes_), nodata, dtype=np.float32)

        chunk_size = int(n_good_samples / num_chunks)
        chunk_resid = n_good_samples - (chunk_size * num_chunks)
        log.info("   Number of chunks {}. Chunk size {}. Chunk residual {}.".format(num_chunks, chunk_size, chunk_resid))
        # The chunks iterate over all values in the array [x * y, bands] always with 8 bands per chunk
        for chunk_id in range(num_chunks):
            offset = chunk_id * chunk_size
            # process the residual pixels with the last chunk
            if chunk_id == num_chunks - 1:
                chunk_size = chunk_size + chunk_resid
            log.info("   Classifying chunk {} of size {}".format(chunk_id, chunk_size))
            chunk_view = good_samples[offset : offset + chunk_size]
            #indices_view = good_indices[offset : offset + chunk_size]
            #log.info("   Creating out_view")
            out_view = classes[offset : offset + chunk_size]  # dimensions [chunk_size]
            #log.info("   Calling model.predict")
            chunk_view = chunk_view.copy() # bug fix for Pandas bug: https://stackoverflow.com/questions/53985535/pandas-valueerror-buffer-source-array-is-read-only
            out_view[:] = model.predict(chunk_view)

            if prob_out_path:
                log.info("   Calculating probabilities")
                prob_view = probs[offset : offset + chunk_size, :]
                prob_view[:, :] = model.predict_proba(chunk_view)

        #log.info("   Creating class array of size {}".format(n_samples))
        class_out_array = np.full((n_samples), nodata)
        for i, class_val in zip(good_indices, classes):
            class_out_array[i] = class_val

        #log.info("   Creating GDAL class image")
        class_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :] = \
            reshape_ml_out_to_raster(class_out_array, image.RasterXSize, image.RasterYSize)

        if prob_out_path:
            #log.info("   Creating probability array of size {}".format(n_samples * model.n_classes_))
            prob_out_array = np.full((n_samples, model.n_classes_), nodata)
            for i, prob_val in zip(good_indices, probs):
                prob_out_array[i] = prob_val
            #log.info("   Creating GDAL probability image")
            #log.info("   N Classes = {}".format(prob_out_array.shape[1]))
            #log.info("   Image X size = {}".format(image.RasterXSize))
            #log.info("   Image Y size = {}".format(image.RasterYSize))
            prob_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :, :] = \
                reshape_prob_out_to_raster(prob_out_array, image.RasterXSize, image.RasterYSize)

        class_out_image = None
        prob_out_image = None
        if prob_out_path:
            return class_out_path, prob_out_path
        else:
            return class_out_path
    else:
        log.warning("No good pixels found - no classification image was created.")
        return ""
    """
    log.error(
        "This function currently does nothing. Call pyeo.classification.change_from_composite instead."
    )
    return ""


def autochunk(dataset, mem_limit=None):
    """
    :meta private:
    EXPERIMENTAL Calculates the number of chunks to break a dataset into without a memory error. Presumes that 80% of the
    memory on the host machine is available for use by pyeo.
    We want to break the dataset into as few chunks as possible without going over mem_limit.
    mem_limit defaults to total amount of RAM available on machine if not specified

    Parameters
    ----------
    dataset
        The dataset to chunk
    mem_limit
        The maximum amount of memory available to the process. Will be automatically populated from os.sysconf if missing.

    Returns
    -------
    num_chunks : int
        The number of chunks to most efficiently break the image into.

    """
    pixels = dataset.RasterXSize * dataset.RasterYSize
    bytes_per_pixel = dataset.GetVirtualMemArray().dtype.itemsize * dataset.RasterCount
    image_bytes = bytes_per_pixel * pixels
    if not mem_limit:
        mem_limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        # Lets assume that 20% of memory is being used for non-map bits
        mem_limit = int(mem_limit * 0.8)
    # if I went back now, I would fail basic programming here.
    for num_chunks in range(1, pixels):
        if pixels % num_chunks != 0:
            continue
        chunk_size_bytes = (pixels / num_chunks) * bytes_per_pixel
        if chunk_size_bytes < mem_limit:
            return num_chunks


def classify_directory(
    in_dir: str,
    model_path: str,
    class_out_dir: str,
    prob_out_dir: str = None,
    apply_mask: bool = False,
    out_type: str = "GTiff",
    chunks: int = 4,
    skip_existing: bool = False,
) -> None:
    """
    Classifies every file ending in .tif in in_dir using model at model_path. Outputs are saved
    in class_out_dir and prob_out_dir, named [input_name]_class and _prob, respectively.

    See the documentation for :py:func:`classify_image` for more details.

    Parameters
    ----------
    in_dir : str
        The path to the directory containing the rasters to be classified.
    model_path : str
        The path to the .pkl file containing the model.
    class_out_dir : str
        The directory that will store the classified maps
    prob_out_dir : str, optional
        If present, the directory that will store the probability maps of the classified maps. If not provided, will not generate probability maps.
    apply_mask : bool, optional
        If present, uses the corresponding .msk files to mask the directories. Defaults to True.
    out_type : str, optional
        The raster format of the class image. Defaults to "GTiff" (geotif). See gdal docs for valid datatypes.
    chunks : int, optional
        The number of chunks to break each image into for processing. See :py:func:`classify_image`
    skip_existing : boolean, optional
        If True, skips the classification if the output file already exists.

    Returns
    -------
    None
    """

    log = logging.getLogger(__name__)
    log.info("Classifying files in {}".format(in_dir))
    log.info("Class files saved in {}".format(class_out_dir))
    if prob_out_dir is not None:
        log.info("Prob. files saved in {}".format(prob_out_dir))
    if skip_existing:
        log.info("Skipping existing files.")
    for image_path in glob.glob(in_dir + r"/*.tif"):
        image_name = os.path.basename(image_path)[:-4]
        class_out_path = os.path.join(class_out_dir, image_name + "_class.tif")
        if prob_out_dir:
            prob_out_path = os.path.join(prob_out_dir, image_name + "_prob.tif")
        else:
            prob_out_path = None
        classify_image(
            image_path=image_path,
            model_path=model_path,
            class_out_path=class_out_path,
            prob_out_path=prob_out_path,
            apply_mask=apply_mask,
            out_format=out_type,
            chunks=chunks,
            skip_existing=skip_existing,
        )


def reshape_raster_for_ml(image_array: np.ndarray) -> np.ndarray:
    """
    A low-level function that reshapes an array from gdal order `[band, y, x]` to scikit features order `[x*y, band]`

    For classification, scikit-learn functions take a 2-dimensional array of features of the shape (samples, features).
    For pixel classification, features correspond to bands and samples correspond to specific pixels.

    Parameters
    ----------
    image_array : np.ndarray
        A 3-dimensional Numpy array of shape (bands, y, x) containing raster data.

    Returns
    -------
    image_array : np.ndarray
        A 2-dimensional Numpy array of shape (samples, features)

    """
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def reshape_ml_out_to_raster(classes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Takes the output of a pixel classifier and reshapes to a single band image.

    Parameters
    ----------
    classes : array_like of int
        A 1-d numpy array of classes from a pixel classifier
    width : int
        The width in pixels of the image the produced the classification
    height : int
        The height in pixels of the image that produced the classification

    Returns
    -------
    image_array: np.ndarray
        A 2-dimensional Numpy array of shape(width, height)

    """
    image_array = np.reshape(classes, (height, width))
    return image_array


def reshape_prob_out_to_raster(probs: np.ndarray, width: int, height: int):
    """
    Takes the probability output of a pixel classifier and reshapes it to a raster.

    Parameters
    ----------
    probs : np.ndarray
        A numpy array of shape(n_pixels, n_classes)
    width : int
        The width in pixels of the image that produced the probability classification
    height : int
        The height in pixels of the image that produced the probability classification

    Returns
    -------
    image_array : np.ndarray
        The reshaped image array

    """
    classes = probs.shape[1]
    image_array = np.transpose(probs, (1, 0))
    image_array = np.reshape(image_array, (classes, height, width))
    return image_array


def extract_features_to_csv(
    in_ras_path: str, training_shape_path: str, out_path: str, attribute: str = "CODE"
):
    """
    Given a raster and a shapefile containing training polygons, extracts all pixels into a CSV file for further
    analysis.

    This produces a CSV file where each row corresponds to a pixel. The columns are as follows:

    -   Column 1: Class labels from the shapefile field labelled as 'attribute'.
    -   Column 2+ : Band values from the raster at in_ras_path.

    Parameters
    ----------
    in_ras_path : str
        The path to the raster used for creating the training dataset
    training_shape_path : str
        The path to the shapefile containing classification polygons
    out_path : str
        The path for the new .csv file
    attribute : str, optional.
        The label of the field in the training shapefile that contains the classification labels. Defaults to "CODE"
    
    Returns
    -------
    None
    """
    this_training_data, this_classes = get_training_data(
        in_ras_path, training_shape_path, attribute=attribute
    )
    sigs = np.vstack((this_classes, this_training_data.T))
    with open(out_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(sigs.T)
    return


def create_trained_model(
    training_image_file_paths: list, cross_val_repeats: int = 10, attribute: str = "CODE"
):
    """
    Creates a trained model from a set of training images with associated shapefiles.

    This assumes that each image in training_image_file_paths has in the same directory a folder of the same
    name containing a shapefile of the same name. For example, in the folder training_data:

    :code:`training_data`

      - :code:`area1.tif`

      - :code:`area1`

        - :code:`area1.shp`
        - :code:`area1.dbf`
        - :code:`area1.cpg`
        - :code:`area1.shx`
        - :code:`area1.prj`

      - :code:`area2.tif`

      - :code:`area2`

        - :code:`area2.shp`
        - :code:`area2.dbf`
        - :code:`area2.cpg`
        - :code:`area2.shx`
        - :code:`area2.prj`

    Parameters
    ----------
    training_image_file_paths : list[str]
        A list of filepaths to training images.
    cross_val_repeats : int, optional
        The number of cross-validation repeats to use. Defaults to 10.
    attribute : str, optional.
        The label of the field in the training shapefiles that contains the classification labels. Defaults to CODE.

    Returns
    -------
    model : sklearn.classifier
        A fitted scikit-learn model. See notes.
    scores : tuple(float, float, float, float)
        The cross-validation scores for model

    Notes
    ----
    For full details of how to create an appropriate shapefile, see [here](../docs/build/html/index.html#training_data).
    At present, the model is an ExtraTreesClassifier arrived at by tpot:

    .. code:: python

        model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55,
            min_samples_leaf=2, min_samples_split=16, n_estimators=100, n_jobs=-1, class_weight='balanced')

    """
    # TODO: This could be optimised by pre-allocating the training array.
    learning_data = None
    classes = None
    log.info("Collecting training data from all tif/shp file pairs.")
    for training_image_file_path in training_image_file_paths:
        # check whether both the tiff file and the shapefile exist
        training_image_folder, training_image_name = os.path.split(
            training_image_file_path
        )
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path_name = training_image_name + ".shp"
        # find the full path to the shapefile, this can be in a subdirectory
        shape_paths = [
            f.path
            for f in os.scandir(training_image_folder)
            if f.is_file() and os.path.basename(f) == shape_path_name
        ]
        if len(shape_paths) == 0:
            log.warning("{} not found. Skipping.".format(shape_path_name))
        else:
            if len(shape_paths) > 1:
                log.warning(
                    "Several versions of {} exist. Using the first of these files.".format(
                        shape_path_name
                    )
                )
                for f in shape_paths:
                    log.info("  {}".format(f))
            shape_path = shape_paths[0]
            more_training_data, more_classes = get_training_data(
                training_image_file_path, shape_path, attribute
            )
            log.info("  Found class labels: {}".format(np.unique(more_classes)))
            if learning_data is None:
                learning_data = more_training_data
                classes = more_classes
            else:
                learning_data = np.append(learning_data, more_training_data, 0)
                classes = np.append(classes, more_classes)
    log.info("Training the model.")
    log.info("  Class labels: {}".format(np.unique(classes)))
    log.info("  Class data labels   : {}".format(classes.shape))
    log.info("  Learning data labels: {}".format(learning_data.shape))

    """ this saves the training data to a text file, currently disabled
    train_out_path = os.path.join(os.path.dirname(training_image_file_paths[0]), 'training_data.txt')
    log.info("  Writing out training data to: {}".format(train_out_path))
    with open(train_out_path, 'w') as f:
        for line in range(len(classes)):
            textline = str(classes[line])+", "+str(learning_data[line])
            f.writelines(textline)
    """

    model = ens.ExtraTreesClassifier(
        bootstrap=False,
        criterion="gini",
        max_features=0.55,
        min_samples_leaf=2,
        min_samples_split=16,
        n_estimators=100,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(learning_data, classes)
    scores = cross_val_score(
        model, learning_data, classes, scoring="accuracy", cv=cross_val_repeats
    )
    log.info("Accuracy: {.3f} ({.3f})".format(np.mean(scores), np.std(scores)))
    return model, scores


def create_model_for_region(path_to_region: str, model_out: str, scores_out: str, attribute: str = "CODE") -> None:
    """
    Takes all .tif files in a given folder and creates a pickled scikit-learn model for classifying them.
    Wraps :py:func:`create_trained_model`; see docs for that for the details.

    Parameters
    ----------
    path_to_region : str
        Path to the folder containing the tifs.
    model_out : str
        Path to location to save the .pkl file
    scores_out : str
        Path to save the cross-validation scores
    attribute : str, optional
        The label of the field in the training shapefiles that contains the classification labels. Defaults to "CODE".
    
    Returns
    -------
    None
    """
    log.info(
        "Create model for region based on tif/shp file pairs: {}".format(path_to_region)
    )
    image_glob = os.path.join(path_to_region, r"*.tif")
    image_list = glob.glob(image_glob)
    model, scores = create_trained_model(image_list, attribute=attribute)
    joblib.dump(model, model_out)
    log.info("Making file: {}".format(scores_out))
    with open(scores_out, "w") as score_file:
        score_file.write(str(scores))
        score_file = None


def create_rf_model_for_region(
    path_to_region: str, model_out: str, attribute: str = "CODE", band_names: list = [], gridsearch: int = 1, k_fold: int = 5
) -> None:
    """
    Takes all .tif files in a given folder and creates a pickled scikit-learn random forest model.

    Parameters
    ----------
    path_to_region : str
        Path to the folder containing the tifs.
    model_out : str
        Path to location to save the .pkl file
    scores_out : str
        Path to save the cross-validation scores
    attribute : str
        The label of the field in the training shapefiles that contains the classification labels. Defaults to "CODE".
    band_names : list[str]
        List of band names using in labelling the signatures in the signature file. Can be left as an empty list [].
    gridsearch : int, optional
        Number of randomized random forests for gridsearch. Defaults to 1.
    k_fold : int, optional
        Number of groups for k-fold validation during gridsearch. Defaults to 5.

    Returns
    -------
    None
    """
    log.info(
        "Create a random forest classification model for region based on tif/shp file pairs: {}".format(
            path_to_region
        )
    )
    image_glob = os.path.join(path_to_region, r"*.tif")
    image_list = glob.glob(image_glob)
    model = train_rf_model(
        image_list,
        model_out,
        ntrees=101,
        attribute=attribute,
        band_names=band_names,
        weights=None,
        gridsearch=gridsearch,
    )
    return


def create_model_from_signatures(sig_csv_path: str,
                                 model_out: str,
                                 sig_datatype=np.int32):
    """
    Takes a .csv file containing class signatures - produced by extract_features_to_csv - and uses it to train
    and pickle a scikit-learn model.

    Parameters
    ----------
    sig_csv_path : str
        The path to the signatures file
    model_out : str
        The location to save the pickled model to.
    sig_datatype : dtype, optional
        The datatype to read the csv as. Defaults to int32.

    Returns
    -------
    None

    Notes
    -----
    At present, the model is an ExtraTreesClassifier arrived at by tpot:

    .. code:: python

        model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
              min_samples_split=16, n_estimators=100, n_jobs=-1, class_weight='balanced')


    """
    model = ens.ExtraTreesClassifier(
        bootstrap=False,
        criterion="gini",
        max_features=0.55,
        min_samples_leaf=2,
        min_samples_split=16,
        n_estimators=100,
        n_jobs=-1,
        class_weight="balanced",
    )
    features, labels = load_signatures(sig_csv_path, sig_datatype)

    model.fit(features, labels)
    joblib.dump(model, model_out)


def load_signatures(sig_csv_path: str, sig_datatype=np.int32):
    """
    Extracts features and class labels from a signature CSV.

    Parameters
    ----------
    sig_csv_path : str
        The path to the csv
    sig_datatype : dtype, optional
        The type of pixel data in the signature CSV. Defaults to np.int32

    Returns
    -------
    features : np.ndarray
        a numpy array of the shape (feature_count, sample_count)
    class_labels : np.ndarray
        a 1d numpy array of class labels (int) corresponding to the samples in features.

    """
    data = np.genfromtxt(sig_csv_path, delimiter=",", dtype=sig_datatype).T
    return (data[1:, :].T, data[0, :])


def get_shp_extent(shapefile: str):
    """
    Get the extent of the first layer, the CRS and the EPSG code from a shapefile.

    Parameters
    ----------
    
    shapefile : str
        path of the shapefile (.shp)

    Returns
    -------
    tuple:

      - extent : tuple(float, float, float, float) - 
        Extent of the shapefile represented as (min_x, min_y, max_x, max_y).
      - SpatialRef : osgeo.osr.SpatialReference - 
        Coordinate referencing system of the shapefile.
      - EPSG : int - 
        EPSG code of the shapefile.

    """

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shapefile, 0)

    # get extent
    lyr = ds.GetLayer()
    extent = lyr.GetExtent()

    # get projection information
    SpatialRef = lyr.GetSpatialRef()

    # get EPSG code of the CRS
    EPSG = SpatialRef.GetAttrValue("AUTHORITY", 1)

    # close file
    ds = None

    return (extent, SpatialRef, EPSG)


def get_training_data(image_path: str, shape_path: str, attribute: str = "CODE"):
    """
    Given an image and a shapefile with categories, returns training data and features suitable
    for fitting a scikit-learn classifier.Image and shapefile must be in the same map projection /
    coordinate referencing system.

    For full details of how to create an appropriate shapefile, see [here](../index.html#training_data).

    Parameters
    ----------
    image_path : str
        The path to the raster image to extract signatures from
    shape_path : str
        The path to the shapefile containing labelled class polygons
    attribute : str, optional
        The shapefile field containing the class labels. Defaults to "CODE".

    Returns
    -------
    training_data : np.ndarray
        A numpy array of shape (n_pixels, bands), where n_pixels is the number of pixels covered by the training polygons
    training_pixels : np.ndarray
        A 1-d numpy array of length (n_pixels) containing the class labels for the corresponding pixel in training_data

    Notes
    -----
    For performance, this uses scikit's sparse.nonzero() function to get the location of each training data pixel.
    This means that this will ignore any classes with a label of '0'.

    """
    # TODO: WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong
    # quietly and in a way that'll cause the most issues further on down the line
    log.info("Get training data from {}".format(image_path))
    log.info("                   and {}".format(shape_path))
    if not os.path.exists(image_path):
        log.error("{} not found.".format(image_path))
        sys.exit(1)
    if not os.path.exists(shape_path):
        log.error("{} not found.".format(shape_path))
        sys.exit(1)
    # check that the two map projections have the same EPSG codes
    image = gdal.Open(image_path)
    epsg1 = osr.SpatialReference(wkt=image.GetProjection()).GetAttrValue("AUTHORITY", 1)
    shp_extent, shp_crs, epsg2 = get_shp_extent(shape_path)
    if not epsg1 == epsg2:
        log.error("EPSG codes of the image and shapefile are different. Aborting.")
        log.error("   Image has EPSG: {}".format(epsg1))
        log.error("   Image has EPSG: {}".format(epsg2))
        image = None
        return [], []
    with TemporaryDirectory(dir=os.getcwd()) as td:
        shape_raster_path = os.path.join(
            td, os.path.basename(shape_path)[:-4] + "_rasterised"
        )
        log.info("Shape raster path {}".format(shape_raster_path))
        shape_raster_path = shapefile_to_raster(
            shape_path,
            image_path,
            shape_raster_path,
            verbose=False,
            attribute=attribute,
            nodata=0,
        )
        rasterised_shapefile = gdal.Open(shape_raster_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        shape_sparse = sp.coo_matrix(np.asarray(shape_array).squeeze())
        y, x, training_pixels = sp.find(shape_sparse)
        log.info("{} bands in image file".format(image.RasterCount))
        log.info("{} training pixels in shapefile".format(len(training_pixels)))
        # log.info("Image raster x size: {}".format(image.RasterXSize))
        # log.info("Image raster y size: {}".format(image.RasterYSize))
        # log.info("Shape raster x size: {}".format(rasterised_shapefile.RasterXSize))
        # log.info("Shape raster y size: {}".format(rasterised_shapefile.RasterYSize))
        training_data = np.empty((len(training_pixels), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        for index in range(len(training_pixels)):
            training_data[index, :] = image_array[:, y[index], x[index]]
        image_array = None
        shape_array = None
        image = None
        rasterised_shapefile = None
        return training_data, training_pixels


def raster_reclass_binary(img_path: str, rcl_value: int, outFn: str, outFmt: str = "GTiff", write_out: bool = True) -> np.ndarray:
    """
    Takes a raster and reclassifies rcl_value to 1, with all others becoming 0. In-place operation if write_out is True.

    Parameters
    ----------
    img_path : str
        Path to 1 band input raster.
    rcl_value : int
        Integer indication the value that should be reclassified to 1. All other values will be 0.
    outFn : str
        Output file name.
    outFmt : str, optional
        Output format. Set to GTiff by default. Other GDAL options available.
    write_out : bool, optional.
        Set to True by default. Will write raster to disk. If False, only an array is returned

    Returns
    -------
    in_array : np.ndarray
        Reclassifies numpy array
    """
    log = logging.getLogger(__name__)
    log.info("Starting raster reclassification.")
    # load in classification raster
    in_ds = gdal.Open(img_path)
    in_band = in_ds.GetRasterBand(1)
    in_array = in_band.ReadAsArray()

    # reclassify
    in_array[in_array != rcl_value] = 0
    in_array[in_array == rcl_value] = 1

    if write_out:
        driver = gdal.GetDriverByName(str(outFmt))
        out_ds = driver.Create(outFn, in_band.XSize, in_band.YSize, 1, in_band.DataType)
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        # Todo: Check for existing files. Skip if exists or make overwrite optional.
        out_ds.GetRasterBand(1).WriteArray(in_array)

        # write the data to disk
        out_ds.FlushCache()

        # Compute statistics on each output band
        # setting ComputeStatistics to false calculates stats on all pixels not estimates
        out_ds.GetRasterBand(1).ComputeStatistics(False)

        out_ds.BuildOverviews("average", [2, 4, 8, 16, 32])

        out_ds = None

    return in_array


def shapefile_to_raster(
    shapefilename: str,
    inraster_filename: str,
    outraster_filename: str,
    verbose=False,
    nodata=0,
    attribute: str="CODE",
):
    """
    This function:

    Reads in a shapefile with polygons and produces a raster file that
    aligns with an input rasterfile (same corner coordinates, resolution, coordinate
    reference system and geotransform). Each pixel value in the output raster will
    indicate the number from the shapefile based on the selected attribute column.

    Parameters
    ----------
    shapefilename : str
        String pointing to the input shapefile in ESRI format.

    inraster_filename : str
        String pointing to the input raster file that we want to align the output raster to.

    outraster_filename : str
        String pointing to the output raster file.

    verbose : boolean
        True or False. If True, additional text output will be printed to the log file.

    nodata : int
        No data value.

    attribute : str
        Name of the column of the attribute table of the shapefile that will be burned into the raster. If None, use the first attribute.

    Returns
    -------
    outraster_filename : str

    Notes
    -----
    Based on https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

    """

    with TemporaryDirectory(dir=os.getcwd()) as td:
        image = gdal.Open(inraster_filename)
        image_gt = image.GetGeoTransform()
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if drv is None:
            log.error("  {} driver not available.".format("ESRI Shapefile"))
            sys.exit(1)
        inshape = drv.Open(shapefilename)
        inlayer = inshape.GetLayer()
        out_path = os.path.join(td, os.path.basename(outraster_filename))
        x_min = image_gt[0]
        y_max = image_gt[3]
        x_max = x_min + image_gt[1] * image.RasterXSize
        y_min = y_max + image_gt[5] * image.RasterYSize
        x_res = image.RasterXSize
        y_res = image.RasterYSize
        pixel_width = image_gt[1]
        target_ds = gdal.GetDriverByName("GTiff").Create(
            out_path, x_res, y_res, 1, gdal.GDT_Int16
        )
        target_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_width))
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        if attribute is not None:
            gdal.RasterizeLayer(
                target_ds, [1], inlayer, options=["ATTRIBUTE={}".format(attribute)]
            )  # , 'ALL_TOUCHED=TRUE'])
        else:
            gdal.RasterizeLayer(
                target_ds, [1], inlayer
            )  # , options=['ALL_TOUCHED=TRUE'])
        shutil.move(out_path, outraster_filename)
        target_ds = None
        image = None
        inshape = None
        inlayer = None
    return outraster_filename


def train_rf_model(
    raster_paths: list,
    modelfile: str,
    ntrees: int=101,
    attribute: str="CODE",
    band_names: list=[],
    weights: list=None,
    balanced: bool=True,
    gridsearch: int=1,
    k_fold: int=5,
):
    """
    This function:

    Trains a random forest classifier model based on a raster file with bands as features and a second raster file with training data, in which pixel values indicate the class.

    Parameters
    ----------
    raster_paths: list[str]
        list of filenames and paths to the raster files to be classified in tiff format. It is a condition that shapefiles of matching name exist in the same directory.

    modelfile: str
        filename and path to a pickle file to save the trained model to

    ntrees: int
        number of trees in the random forest, default = 101

    attribute: str
        string naming the attribute column to be rasterised in the shapefile

    band_names: list[str]
        list of strings indicating the names of the bands (used for text output and labelling the learning data output file). If [], a sequence of numbers is assigned.
    
    weights: list[int], optional
        a list of integers giving weights for all classes. If not specified, all weights will be equal.

    balanced: bool, optional
        if True, use a balanced number of training pixels per class

    gridsearch : int, optional
        Number of randomized random forests for gridsearch. Defaults to 1.

    k_fold : int, optional
        Number of groups for k-fold validation during gridsearch. Defaults to 5.

    Returns
    -------
    random forest model object

    Notes
    -----
    Adapted from pygge.py

    """

    log.info("Collecting training data from all tif/shp file pairs.")
    learning_data = []
    labels = []
    for raster_path in raster_paths:
        # check whether both the tiff file and the shapefile exist
        if not os.path.exists(raster_path):
            log.warning("{} not found. Skipping.".format(raster_path))
            continue
        training_image_folder, training_image_name = os.path.split(raster_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path_name = training_image_name + ".shp"
        # find the full path to the shapefile, this can be in a subdirectory
        shape_paths = [
            f.path
            for f in os.scandir(training_image_folder)
            if f.is_file() and os.path.basename(f) == shape_path_name
        ]
        if len(shape_paths) == 0:
            log.warning("{} not found. Skipping.".format(shape_path_name))
        else:
            if len(shape_paths) > 1:
                log.warning(
                    "Several versions of {} exist. Using the first of these files.".format(
                        shape_path_name
                    )
                )
                for f in shape_paths:
                    log.info("  {}".format(f))
            shapefile_path = shape_paths[0]
            log.info("Analysing shapefile: {}".format(shapefile_path))
            image = gdal.Open(raster_path)
            image_array = image.GetVirtualMemArray(eAccess=gdal.GA_ReadOnly)
            # check that the two map projections have the same EPSG codes
            epsg1 = osr.SpatialReference(wkt=image.GetProjection()).GetAttrValue(
                "AUTHORITY", 1
            )
            img_prj = image.GetProjection()
            img_crs = osr.SpatialReference(wkt=img_prj)
            shp_extent, shp_crs, epsg2 = get_shp_extent(shapefile_path)
            log.info(
                "EPSG codes of the image and shapefile: {}, {}".format(epsg1, epsg2)
            )
            if not epsg1 == epsg2:
                log.warning(
                    "Reprojecting shapefile. The EPSG codes of the image and shapefile are different."
                )
                log.warning("   Image has EPSG    : {}".format(epsg1))
                log.warning("   Shapefile has EPSG: {}".format(epsg2))
                driver = ogr.GetDriverByName("ESRI Shapefile")
                dataSource = driver.Open(shapefile_path, 1)
                layer = dataSource.GetLayer()
                sourceprj = layer.GetSpatialRef()
                targetprj = osr.SpatialReference(wkt=img_prj)
                transform = osr.CoordinateTransformation(sourceprj, targetprj)
                to_fill = ogr.GetDriverByName("Esri Shapefile")
                new_shapefile_path = shapefile_path[:-4] + "_" + str(epsg1) + ".shp"
                # Remove reprojected output shapefile if it already exists
                if os.path.exists(new_shapefile_path):
                    log.warning(
                        "Reprojected shapefile already exists. Deleting previous version: {}".format(
                            new_shapefile_path
                        )
                    )
                    to_fill.DeleteDataSource(new_shapefile_path)
                ds = to_fill.CreateDataSource(new_shapefile_path)
                outlayer = ds.CreateLayer("", targetprj, ogr.wkbPolygon)
                outlayer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
                outlayer.CreateField(ogr.FieldDefn(attribute, ogr.OFTInteger))
                i = 0
                for feature in layer:
                    transformed = feature.GetGeometryRef()
                    transformed.Transform(transform)
                    geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
                    defn = outlayer.GetLayerDefn()
                    feat = ogr.Feature(defn)
                    feat.SetField("id", i)
                    feat.SetField(attribute, feature.GetField(attribute))
                    feat.SetGeometry(geom)
                    outlayer.CreateFeature(feat)
                    i += 1
                    feat = None
                dataSource = None
                ds = None
                shapefile_path = new_shapefile_path
            with TemporaryDirectory(dir=os.getcwd()) as td:
                # rasterise the shapefile
                shape_raster_path = os.path.join(
                    td, os.path.basename(shapefile_path)[:-4] + "_rasterised.tif"
                )
                # log.info("Shape raster path {}".format(shape_raster_path))
                shape_raster_path = shapefile_to_raster(
                    shapefile_path,
                    raster_path,
                    shape_raster_path,
                    verbose=False,
                    nodata=0,
                    attribute=attribute,
                )
                rasterised_shapefile = gdal.Open(shape_raster_path)
                shape_array = rasterised_shapefile.GetVirtualMemArray(
                    eAccess=gdal.GA_ReadOnly
                )
                # read in the class labels
                these_labels = list(np.unique(shape_array[shape_array > 0]))
                nclasses = len(these_labels)
                labels = labels + these_labels
                log.info(
                    "This training data shapefile includes {} classes: {}".format(
                        nclasses, these_labels
                    )
                )
                nbands = image.RasterCount
                if nbands == 1:
                    log.info("{} band in image file".format(nbands))
                else:
                    log.info("{} bands in image file".format(nbands))
                if len(band_names) == nbands:
                    log.info("  {}".format(band_names))
                # compose the X,Y pixel positions (feature dataset and training dataset)
                # assumed that 0 = 'no class' value
                log.info(image_array.shape)
                log.info(shape_array.shape)
                for j in range(image_array.shape[0] - 1):
                    log.info("\n{}".format(shape_array[j, :]))
                X = image_array[:, shape_array > 0]
                Y = shape_array[shape_array > 0].flatten()
                log.info(X.shape)
                log.info(Y.shape)
                log.info("{} training pixels in shapefile".format(len(Y)))
                shape_array = None
                rasterised_shapefile = None
            image_array = None
            image = None

            if learning_data == []:
                learning_data = X.transpose()
                classes = Y
            else:
                # log.info(learning_data.shape)
                # log.info(X.shape)
                learning_data = np.append(learning_data, X.transpose(), 0)
                classes = np.append(classes, Y)
                # log.info(learning_data.shape)
                # log.info(classes.shape)

    # get unique list of all class labels
    labels = list(dict.fromkeys(labels))
    log.info("Training data collection complete.")
    log.info("Training pixels by class:")
    smallest = 0
    for c in labels:
        found = len(classes[classes == c])
        log.info("  Class {} has {} training pixels".format(c, found))
        if found < smallest or smallest == 0:
            smallest = found

    if balanced:
        indices = []
        log.info(
            "Drawing a more balanced random sample of {} max. training pixels for each class.".format(
                10 * smallest
            )
        )
        for c in labels:
            # draw a random sample of the list indices where that class is found
            if (
                len([pos for pos, value in enumerate(classes) if value == c])
                > 10 * smallest
            ):
                indices = indices + random.sample(
                    [pos for pos, value in enumerate(classes) if value == c],
                    10 * smallest,
                )
            else:
                indices = indices + [
                    pos for pos, value in enumerate(classes) if value == c
                ]
        classes = classes[indices]
        learning_data = learning_data[indices]

        log.info("After class balancing, training pixels by class:")
        for c in labels:
            found = len(classes[classes == c])
            log.info("  Class {} has {} training pixels".format(c, found))

    # TODO: Test this bit: save training data to file and produce basic signature statistics by class
    if len(band_names) != nbands:
        band_names = [str(i + 1) for i in range(nbands)]
    learning_df = pd.DataFrame(learning_data, columns=band_names)
    learning_df["label"] = classes
    learningfile = modelfile[:-4] + "_learning.pkl"
    log.info("Saving learning data file: {}".format(learningfile))
    joblib.dump(learning_df, learningfile)
    sigfile = modelfile[:-4] + "_signatures.txt"
    log.info("Saving tab-delimited signature file: {}".format(sigfile))
    col_names = learning_df.columns.values.tolist()
    with open(sigfile, "w") as f:
        log.info("Signatures:")
        log.info("class, band, min, max, mean, stdev")
        f.write("class\tband\tmin\tmax\tmean\tstdev\n")
        for c in range(np.max(classes)):
            if c + 1 in classes:
                for b in band_names:
                    log.info(
                        "{}, {}, {}, {}, {}, {}".format(
                            c + 1,
                            b,
                            learning_df.groupby("label").min()[b][c + 1],
                            learning_df.groupby("label").max()[b][c + 1],
                            learning_df.groupby("label").mean()[b][c + 1],
                            learning_df.groupby("label").std()[b][c + 1],
                        )
                    )
                    f.write(
                        "{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            c + 1,
                            b,
                            learning_df.groupby("label").min()[b][c + 1],
                            learning_df.groupby("label").max()[b][c + 1],
                            learning_df.groupby("label").mean()[b][c + 1],
                            learning_df.groupby("label").std()[b][c + 1],
                        )
                    )
            else:
                log.warning(
                    "Class {} does not exist in the learning data.".format(c + 1)
                )
        f = None

    # create a dictionary of class weights (class 1 has the weight 1, etc.)
    w = dict()  # create an empty dictionary
    for i in range(nclasses):  # iterate over all classes from 0 to nclasses-1
        if weights == None:
            w[i + 1] = "1"  # if not specified, set all weights to 1
        else:
            if weights.size >= nclasses:  # if enough weights are given, assign them
                w[i + 1] = weights[i]  # assign the weights if specified by the user
            else:  # if fewer weights are defined than the number of classes, then set the remaining weights to 1
                if i > weights.size:
                    w[i + 1] = "1"  # set weight to 1
                else:
                    w[i + 1] = weights[i]  # assign the weights if specified by the user

    # Split the data into training and testing datasets
    training, testing, training_classes, testing_classes = train_test_split(
        learning_data, classes, test_size=0.25, random_state=101
    )
    # This function does not normalize the data. If this was added, then the new data for classification would
    # have to be normalized in the same way.
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(training), columns = X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns = X.columns)

    if gridsearch > 1:
        log.info(
            "Grid search: Finding optimal parameter space for the random forest from {} forests.".format(
                gridsearch
            )
        )
        # create a dictionary of values to choose from
        # inspired by https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        n_estimators = np.linspace(100, 500, int((500 - 100) / 200) + 1, dtype=int)
        max_features = ["auto", "sqrt"]
        max_depth = [2, 10, 50, 100, 150, 200]
        min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=9)]
        min_samples_leaf = [1, 2, 3, 4]
        bootstrap = [True, False]
        criterion = ["gini", "entropy"]
        grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "criterion": criterion,
        }
        # Begin the grid search and fit a new random forest classifier on the parameters found from the random search
        rf_base = ens.RandomForestClassifier()
        rf = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=grid,
            n_iter=gridsearch,
            cv=k_fold,
            verbose=3,
            random_state=42,
            n_jobs=1,
        )

        """
        rf = GridSearchCV(estimator = rf_base,
                          param_grid = grid,
                          scoring = 'balanced_accuracy',
                          n_jobs = -1,
                          refit = True,
                          cv = 10, # for k-fold cross-validation
                          verbose = 3,
                          pre_dispatch = 16,
                          return_train_score = True)
        """
        rf.fit(training, training_classes)
        # View the parameter values the random search found
        log.info("Best parameters from the grid search: {}".format(rf.best_params_))
        # TODO: At this point, we could do a targeted grid search on a smaller subset of the parameter space based on what we know now, but that cannot easily be automated
    else:
        log.info(
            "Fitting a single random forest model to the training dataset without gridsearch."
        )
        # for more information: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        rf = ens.RandomForestClassifier()
        rf.fit(training, training_classes)

    # Accuracy assessment / validation
    preds = rf.predict(testing)
    log.info("Accuracy assessment at model training stage")
    log.info(
        "Training   classification accuracy (75% of data): {}".format(
            rf.score(training, training_classes)
        )
    )
    log.info(
        "Validation classification accuracy (25% of data): {}".format(
            rf.score(testing, testing_classes)
        )
    )
    # Confusion matrix: Rows represent predicted classes, columns are the true classes
    confusion = metrics.confusion_matrix(testing_classes, preds, labels=labels)
    log.info("Confusion matrix:")
    log.info("            Reference class ->")
    log.info("                |")
    log.info("Predicted class v")
    log.info("\n{}".format(confusion))
    # Matt: 13/03/2023, added classification report functionality
    # Classification Report: useful metrics, recall, precision, accuracy, f1-score.
    log.info("--" * 20)
    report = metrics.classification_report(y_true=testing_classes, y_pred=preds)
    # target_names argument would provide class names in the classification report, but we would need to allow users to input their own labels (in order of digits 1 to n)
    log.info("Classification Report")
    log.info("\n{}".format(report))

    # save report
    out_folder = Path(modelfile).parent.absolute()
    fname = f"{out_folder}{os.sep}class_scores.txt"
    log.info(f"Saving classification report to {fname}")
    with open(fname, "w") as txt:
        print(report, file=txt)

    # build the Random Forest Classifier
    # for more information: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # OLD: rf = ens.RandomForestClassifier(class_weight = weights, n_estimators = ntrees, criterion = 'gini', max_depth = 4,
    #                            min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto',
    #                            bootstrap = True, oob_score = True, n_jobs = -1, random_state = None, verbose = True)
    # fit the model to the training data and the feature dataset
    # rf = rf.fit(learning_data, classes)

    # export the Random Forest model to a file
    joblib.dump(rf, modelfile)

    # calculate the feature importances
    if gridsearch == 1:
        importances = rf.feature_importances_
    else:
        log.info(
            "Best random forest estimator from gridsearch: {}".format(
                rf.best_estimator_
            )
        )
        log.info(
            "Best random forest parameters from gridsearch: {}".format(rf.best_params_)
        )
        log.info(
            "Best random forest accuracy score from gridsearch: {}".format(
                rf.best_score_
            )
        )
        # TODO: Test this - Add feature importances here. rf.feature_importances_ does not work after randomized grid search
        importances = rf.best_estimator_.feature_importances_
    # TODO: Does not work on RandomizedSearchCV objects, only for gridsearch objects
    # std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    log.info("Feature ranking:")
    for f in range(learning_data.shape[1]):
        log.info("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    return rf  # returns the random forest model object


def classify_rf(raster_path: str,
                modelfile: str,
                outfile: str,
                verbose: bool=False):
    """
    This function:
    
    Reads in a pickle file of a random forest model and a raster file with feature layers, and classifies the raster file using the model.

    Parameters
    ----------
    raster_path: str
        filename and path to the raster file to be classified (in tiff uint16 format)
    modelfile: str
        filename and path to the pickled file with the random forest model in uint8 format
    outfile: str
        filename and path to the output file with the classified map in uint8 format
    verbose: bool, optional
        Defaults to False. If True, provides additional printed output.

    Returns
    -------
    None
    """

    # Read Data
    image = gdal.Open(raster_path, "r")
    image_array = image.GetVirtualMemArray(eAccess=gdal.GA_ReadOnly)
    if verbose:
        log.info("image shape = {}".format(image.shape))
    n = img.shape[0]
    if verbose:
        log.info(" {} bands".format(n))
    # load your random forest model from the pickle file
    clf = joblib.load(modelfile)
    # to work with SciKitLearn, we have to reshape the raster as an image
    # this will change the shape from (bands, rows, columns) to (rows, columns, bands)
    img = reshape_as_image(image_array)
    # next, we have to reshape the image again into (rows * columns, bands)
    # because that is what SciKitLearn asks for
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    if verbose:
        log.info("img[:, :, :n].shape = {}".format(img[:, :, :n].shape))
        log.info("new_shape = {}".format(new_shape))
    img_as_array = img[:, :, :n].reshape(new_shape)
    if verbose:
        log.info("img_as_array.shape = {}".format(img_as_array.shape))
    # classify it
    class_prediction = clf.predict(img_as_array)
    # and reshape the flattened array back to its original dimensions
    if verbose:
        log.info("class_prediction.shape = {}".format(class_prediction.shape))
        log.info("img[:, :, 0].shape = {}".format(img[:, :, 0].shape))
    class_prediction = np.uint8(class_prediction.reshape(img[:, :, 0].shape))
    if verbose:
        log.info(class_prediction.dtype)
    # save the image as a uint8 Geotiff file
    tmpfile = rasterio.open(
        outfile,
        "w",
        driver="Gtiff",
        width=src.width,
        height=src.height,
        count=1,
        crs=src.crs,
        transform=src.transform,
        dtype=np.uint8,
    )

    tmpfile.write(class_prediction, 1)

    tmpfile.close()
    return


def plot_signatures(learning_path: str,
                    out_path: str,
                    format: str="PNG"):
    """
    This function:
    
    Creates a graphics image file of the signature scatterplots.

    Parameters
    ----------
    learning_path : str
        The string containing the full directory path to the learning data input file from the model training stage saved by joblib.dump()
    out_path : str
        The string containing the full directory path to the output file for graphical plots
    format : str, optional
        GDAL format for the quicklook raster file, defaults to PNG

    Returns
    -------
    None
    """
    if format != "PNG" and format != "GTiff":
        log.warning("Invalid plot format specified. Changing to PNG.")
        format = "PNG"
        out_path = out_path[:-4] + "png"
    with TemporaryDirectory(dir=os.getcwd()) as td:
        try:
            learning = joblib.load(learning_path)
        except RuntimeError as e:
            log.error(
                "Error loading pickled learning data file: {}".format(learning_path)
            )
            log.error("  {}".format(e))
            return
        groups = learning.groupby("label")
        for name, group in groups:
            log.info("Length of group {} is {}".format(name, len(group)))
        bands = learning.columns[:-1]
        nplots = (len(bands) - 1) * (len(bands) - 2)
        w = 5  # width of one plot in inches
        h = w * nplots  # height of the figure in inches
        fig, ax = plt.subplots(nplots, 1, figsize=(w, h), dpi=72)
        this = -1
        for bx, bandx in enumerate(bands):
            for by, bandy in enumerate(bands[bx + 1 :]):
                this = this + 1  # number of the plot
                log.info("Making plot {} of {}".format(this + 1, nplots))
                ax[this].margins(0.05)
                for name, group in groups:
                    if len(group[bandx]) > 10000:
                        x = random.sample(list(group[bandx]), 10000)
                    else:
                        x = list(group[bandx])
                    if len(group[bandy]) > 10000:
                        y = random.sample(list(group[bandy]), 10000)
                    else:
                        y = list(group[bandy])
                    ax[this].plot(x, y, marker="o", linestyle="", ms=2, label=name)
                # ax[this].set_aspect('equal', adjustable='box')
                ax[this].legend()
                ax[this].set_xlabel(bandx)
                ax[this].set_ylabel(bandy)
        log.info("Saving figure to {}".format(out_path))
        plt.savefig(out_path, dpi=72, format=format, pad_inches=0.1, facecolor="white")
        plt.close(fig)
    return
