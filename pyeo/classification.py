"""
pyeo.classification
===================
Contains every function to do with map classification. This includes model creation, map classification and processes
for array manipulation into scikit-learn compatible forms.
"""
import csv
import glob
import logging
import os
from tempfile import TemporaryDirectory

import gdal
import joblib
import numpy as np
from osgeo import osr
from scipy import sparse as sp
from sklearn import ensemble as ens
from sklearn.externals import joblib as sklearn_joblib
from sklearn.model_selection import cross_val_score

from pyeo.coordinate_manipulation import get_local_top_left
from pyeo.filesystem_utilities import get_mask_path

from pyeo.raster_manipulation import stack_images, create_matching_dataset, apply_array_image_mask, get_masked_array

log = logging.getLogger(__name__)

def change_from_composite(image_path, composite_path, model_path, class_out_path, prob_out_path=None):
    """
    Stacks an image with a composite and classifies each pixel change with a scikit-learn model
    The image that is classified is has the following bands

    1. composite blue
    2. composite green
    3. composite red
    4. composite IR
    5. image blue
    6. image green
    7. image red
    8. image IR

    Parameters
    ----------
    image_path
        The path to the image
    composite_path
        The path to the composite
    model_path
        The path to a .pkl of a scikit-learn classifier that takes 8 features
    class_out_path
        A location to save the resulting classification .tif
    prob_out_path
        A location to save the probability raster of each pixel


    """
    with TemporaryDirectory() as td:
        stacked_path = os.path.join(td, "comp_stack.tif")
        stack_images((composite_path, image_path), stacked_path)
        classify_image(stacked_path, model_path, class_out_path, prob_out_path)


def classify_image(image_path, model_path, class_out_path, prob_out_path=None,
                   apply_mask=False, out_type="GTiff", num_chunks=10, nodata=0, skip_existing = False):
    """
    Produces a class map from a raster and a model.
    This applies the model's fit() function to each pixel in the input raster, and saves the result into an output
    raster. The model is presumed to be a scikit-learn fitted model created using one of the other functions in this
    library (create_model_from_rasters, create_model_from_signatures).

    To fit into a

    Parameters
    ----------
    image_path
        The path to the raster image to be classified.
    model_path
        The path to the .pkl file containing the model
    class_out_path
        The path that the classified map will be saved at.
    prob_out_path
        If present, the path that the class probability map will be stored at.
    apply_mask
        If True, uses the .msk file corresponding to the image at image_path to skip any invalid pixels.
    out_type
        The raster format of the class image. Defaults to GTiff (geotif)
    num_chunks
        The number of chunks the image is broken into prior to classification. The smaller this number, the faster
        classification will run - but the more likely you are to get a outofmemory error.
    nodata
        The value to write to masked pixels
    skip_existing
        If true, do not run if class_out_path already exists


    Notes
    -----
    If you want to create a custom model, the object is presumed to have the following methods and attributes:
       - model.n_classes_ : the number of classes the model will produce
       - model.n_cores : The number of CPU cores used to run the model
       - model.predict() : A function that will take a set of band inputs from a pixel and produce a class.
       - model.predict_proba() : If called with prob_out_path, a function that takes a set of n band inputs from a pixel
                                and produces n_classes_ outputs corresponding to the probabilties of a given pixel being
                                that class

    """
    if skip_existing:
        log.info("Checking for existing classification {}".format(class_out_path))
        if os.path.isfile(class_out_path):
            log.info("Class image exists, skipping.")
            return class_out_path
    log.info("Classifying file: {}".format(image_path))
    log.info("Saved model     : {}".format(model_path))
    image = gdal.Open(image_path)
    if num_chunks == None:
        log.info("No chunk size given, attempting autochunk.")
        num_chunks = autochunk(image)
        log.info("Autochunk to {} chunks".format(num_chunks))
    try:
        model = sklearn_joblib.load(model_path)
    except KeyError:
        log.warning("Sklearn joblib import failed,trying generic joblib")
        model = joblib.load(model_path)
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
    log.info("Reshaping image from GDAL to Scikit-Learn dimensions")
    image_array = reshape_raster_for_ml(image_array)
    # Now it has dimensions [x * y, band] as needed for Scikit-Learn

    # Determine where in the image array there are no missing values in any of the bands (axis 1)
    log.info("Finding good pixels without missing values")
    log.info("image_array.shape = {}".format(image_array.shape))
    n_samples = image_array.shape[0]  # gives x * y dimension of the whole image
    good_mask = np.all(image_array != nodata, axis=1)
    good_sample_count = np.count_nonzero(good_mask)
    log.info("No. good values: {}".format(good_sample_count))
    if good_sample_count <= 0.5*len(good_mask):  # If the images is less than 50% good pixels, do filtering
        log.info("Filtering nodata values")
        good_indices = np.nonzero(good_mask)
        good_samples = np.take(image_array, good_indices, axis=0).squeeze()
        n_good_samples = len(good_samples)
    else:
        log.info("Not worth filtering nodata, skipping.")
        good_samples = image_array
        good_indices = range(0, n_samples)
        n_good_samples = n_samples
    log.info("   All  samples: {}".format(n_samples))
    log.info("   Good samples: {}".format(n_good_samples))
    classes = np.full(n_good_samples, nodata, dtype=np.ubyte)
    if prob_out_path:
        probs = np.full((n_good_samples, model.n_classes_), nodata, dtype=np.float32)

    chunk_size = int(n_good_samples / num_chunks)
    chunk_resid = n_good_samples - (chunk_size * num_chunks)
    log.info("   Number of chunks {} Chunk size {} Chunk residual {}".format(num_chunks, chunk_size, chunk_resid))
    # The chunks iterate over all values in the array [x * y, bands] always with 8 bands per chunk
    for chunk_id in range(num_chunks):
        offset = chunk_id * chunk_size
        # process the residual pixels with the last chunk
        if chunk_id == num_chunks - 1:
            chunk_size = chunk_size + chunk_resid
        log.info("   Classifying chunk {} of size {}".format(chunk_id, chunk_size))
        chunk_view = good_samples[offset : offset + chunk_size]
        #indices_view = good_indices[offset : offset + chunk_size]
        out_view = classes[offset : offset + chunk_size]  # dimensions [chunk_size]
        out_view[:] = model.predict(chunk_view)

        if prob_out_path:
            log.info("   Calculating probabilities")
            prob_view = probs[offset : offset + chunk_size, :]
            prob_view[:, :] = model.predict_proba(chunk_view)

    log.info("   Creating class array of size {}".format(n_samples))
    class_out_array = np.full((n_samples), nodata)
    for i, class_val in zip(good_indices, classes):
        class_out_array[i] = class_val

    log.info("   Creating GDAL class image")
    class_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :] = \
        reshape_ml_out_to_raster(class_out_array, image.RasterXSize, image.RasterYSize)

    if prob_out_path:
        log.info("   Creating probability array of size {}".format(n_samples * model.n_classes_))
        prob_out_array = np.full((n_samples, model.n_classes_), nodata)
        for i, prob_val in zip(good_indices, probs):
            prob_out_array[i] = prob_val
        log.info("   Creating GDAL probability image")
        log.info("   N Classes = {}".format(prob_out_array.shape[1]))
        log.info("   Image X size = {}".format(image.RasterXSize))
        log.info("   Image Y size = {}".format(image.RasterYSize))
        prob_out_image.GetVirtualMemArray(eAccess=gdal.GF_Write)[:, :, :] = \
            reshape_prob_out_to_raster(prob_out_array, image.RasterXSize, image.RasterYSize)

    class_out_image = None
    prob_out_image = None
    if prob_out_path:
        return class_out_path, prob_out_path
    else:
        return class_out_path


def autochunk(dataset, mem_limit=None):
    """
    EXPERIMENTAL Calculates the number of chunks to break a dataset into without a memory error. Presumes that 80% of the
    memory on the host machine is available for use by Pyeo.
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
    The number of chunks to most efficiently break the image into.
    """
    pixels = dataset.RasterXSize * dataset.RasterYSize
    bytes_per_pixel = dataset.GetVirtualMemArray().dtype.itemsize*dataset.RasterCount
    image_bytes = bytes_per_pixel*pixels
    if not mem_limit:
        mem_limit = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
        # Lets assume that 20% of memory is being used for non-map bits
        mem_limit = int(mem_limit*0.8)
    # if I went back now, I would fail basic programming here.
    for num_chunks in range(1, pixels):
        if pixels % num_chunks != 0:
            continue
        chunk_size_bytes = (pixels/num_chunks)*bytes_per_pixel
        if chunk_size_bytes < mem_limit:
            return num_chunks


def classify_directory(in_dir, model_path, class_out_dir, prob_out_dir = None,
                       apply_mask=False, out_type="GTiff", num_chunks=10):
    """
    Classifies every file ending in .tif in in_dir using model at model_path. Outputs are saved
    in class_out_dir and prob_out_dir, named [input_name]_class and _prob, respectively.

    See the documentation for classification.classify_image() for more details.


    Parameters
    ----------
    in_dir
        The path to the directory containing the rasters to be classified.
    model_path
        The path to the .pkl file containing the model.
    class_out_dir
        The directory that will store the classified maps
    prob_out_dir
        The directory that will store the probability maps of the classified maps
    apply_mask
        If present, uses the corresponding .msk files to mask the directories
    out_type
        The raster format of the class image. Defaults to GTiff (geotif)
    num_chunks
        The number of chunks to break an image into.

    """
    log = logging.getLogger(__name__)
    log.info("Classifying files in {}".format(in_dir))
    log.info("Class files saved in {}".format(class_out_dir))
    log.info("Prob. files saved in {}".format(prob_out_dir))
    for image_path in glob.glob(in_dir+r"/*.tif"):
        image_name = os.path.basename(image_path).split('.')[0]
        class_out_path = os.path.join(class_out_dir, image_name+"_class.tif")
        if prob_out_dir:
            prob_out_path = os.path.join(prob_out_dir, image_name+"_prob.tif")
        else:
            prob_out_path = None
        classify_image(image_path, model_path, class_out_path, prob_out_path,
                       apply_mask, out_type, num_chunks)


def reshape_raster_for_ml(image_array):
    """
    A low-level function that reshapes an array from gdal order [band, y, x] to scikit features order [x*y, band]

    For classification, scikit-learn functions take a 2-dimensional array of features of the shape (samples, features).
    For pixel classification, features correspond to bands and samples correspond to specific pixels.

    Parameters
    ----------
    image_array
        A 3-dimensional Numpy array of shape (bands, y, x)

    Returns
    -------
        A 2-dimensional Numpy array of shape (samples, features)

    """
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def reshape_ml_out_to_raster(classes, width, height):
    """
    Takes the output of a pixel classifier and reshapes to a single band image.

    Parameters
    ----------
    classes
        A 1-d numpy array of classes from a pixel classifier
    width
        The width in pixels of the image the produced the classification
    height
        The height in pixels of the image that produced the classification

    Returns
    -------
        A 2-dimensional Numpy array of shape(width, height)

    """
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array


def reshape_prob_out_to_raster(probs, width, height):
    """
    Takes the probability output of a pixel classifier and reshapes it to a raster.

    Parameters
    ----------
    probs
        A numpy array of shape(n_pixels, n_classes)
    width
        The width in pixels of the image that produced the probability classification
    height
        The height in pixels of the image that produced the probability classification

    Returns
    -------
    The reshaped image array

    """
    classes = probs.shape[1]
    image_array = np.transpose(probs, (1, 0))
    image_array = np.reshape(image_array, (classes, height, width))
    return image_array

def extract_features_to_csv(in_ras_path, training_shape_path, out_path, attribute="CODE"):
    """
    Given a raster and a shapefile containing training polygons, extracts all pixels into a CSV file for further
    analysis.

    This produces a CSV file where each row corresponds to a pixel. The columns are as follows:
        Column 1: Class labels from the shapefile field labelled as 'attribute'.
        Column 2... : Band values from the raster at in_ras_path.

    Parameters
    ----------
    in_ras_path
        The path to the raster used for creating the training dataset
    training_shape_path
        The path to the shapefile containing classification polygons
    out_path
        The path for the new .csv file
    attribute
        The label of the field in the training shapefile that contains the classification labels.

    """
    this_training_data, this_classes = get_training_data(in_ras_path, training_shape_path, attribute=attribute)
    sigs = np.vstack((this_classes, this_training_data.T))
    with open(out_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(sigs.T)

def create_trained_model(training_image_file_paths, cross_val_repeats = 5, attribute="CODE"):
    """
    Creates a trained model from a set of training images with associated shapefiles.

    This assumes that each image in training_image_file_paths has in the same directory a folder of the same
    name containing a shapefile of the same name. For example, in the folder training_data:

    training_data
    ---area1.tif
    ---area1
       ---area1.shp
       ---area1.dbx
       ...rest of shapefile for area 1 ...
    ---area2.tif
    ---area2
       ---area2.shp
       ---area2.dbx
       ...rest of shapefile for area 2...


    Parameters
    ----------
    training_image_file_paths
        A list of filepaths to training images.
    cross_val_repeats
        The number of cross-validation repeats to use
    attribute
        The label of the field in the training shapefiles that contains the classification labels.

    Returns
    -------
    model
        A fitted scikit-learn model. See notes.
    scores
        The cross-validation scores for model

    Notes
    ----
    For full details of how to create an appropriate shapefile, see [here](../index.html#training_data).
    At present, the model is an ExtraTreesClassifier arrived at by tpot:
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                 min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')

    """
    # This could be optimised by pre-allocating the training array. but not now.
    learning_data = None
    classes = None
    for training_image_file_path in training_image_file_paths:
        training_image_folder, training_image_name = os.path.split(training_image_file_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path = os.path.join(training_image_folder, training_image_name, training_image_name + '.shp')
        this_training_data, this_classes = get_training_data(training_image_file_path, shape_path, attribute)
        if learning_data is None:
            learning_data = this_training_data
            classes = this_classes
        else:
            learning_data = np.append(learning_data, this_training_data, 0)
            classes = np.append(classes, this_classes)
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    model.fit(learning_data, classes)
    scores = cross_val_score(model, learning_data, classes, cv=cross_val_repeats)
    return model, scores


def create_model_for_region(path_to_region, model_out, scores_out, attribute="CODE"):
    """
    Takes all .tif files in a given folder and creates a pickled scikit-learn model for classifying them.
    Wraps classification.create_trained_model() ; see docs for that for the details.

    Parameters
    ----------
    path_to_region
        Path to the folder containing the tifs.
    model_out
        Path to location to save the .pkl file
    scores_out
        Path to save the cross-validation scores
    attribute
        The label of the field in the training shapefiles that contains the classification labels.

    """
    image_glob = os.path.join(path_to_region, r"*.tif")
    image_list = glob.glob(image_glob)
    model, scores = create_trained_model(image_list, attribute=attribute)
    joblib.dump(model, model_out)
    with open(scores_out, 'w') as score_file:
        score_file.write(str(scores))


def create_model_from_signatures(sig_csv_path, model_out, sig_datatype=np.int32):
    """
    Takes a .csv file containing class signatures - produced by extract_features_to_csv - and uses it to train
    and pickle a scikit-learn model.

    Parameters
    ----------
    sig_csv_path
        The path to the signatures file
    model_out
        The location to save the pickled model to.
    sig_datatype
        The datatype to read the csv as. Defaults to int32.

    Notes
    -----
    At present, the model is an ExtraTreesClassifier arrived at by tpot:
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                 min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    """
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    features, labels = load_signatures(sig_csv_path, sig_datatype)
    model.fit(features, labels)
    joblib.dump(model, model_out)


def load_signatures(sig_csv_path, sig_datatype=np.int32):
    """
    Extracts features and class labels from a signature CSV
    Parameters
    ----------
    sig_csv_path
    sig_datatype

    Returns
    -------
    features
        a numpy array of the shape (feature_count, sample_count)
    class_labels
        a 1d numpy array of class labels corresponding to the samples in features.

    """
    data = np.genfromtxt(sig_csv_path, delimiter=",", dtype=sig_datatype).T
    return (data[1:, :].T, data[0, :])


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """
    Given an image and a shapefile with categories, returns training data and features suitable
    for fitting a scikit-learn classifier.

    This extracts every pixel in image_path touched by the polygons in shape_path

    For full details of how to create an appropriate shapefile, see [here](../index.html#training_data).

    Parameters
    ----------
    image_path
        The path to the raster image to extract signatures from
    shape_path
        The path to the shapefile containing labelled class polygons
    attribute
        The field containing the class labels
    shape_projection_id
        The projection of the shapefile

    Returns
    -------
    training_data
        A numpy array of shape (n_pixels, bands), where n_pixels is the number of pixels covered by the training polygons
    features
        A 1-d numpy array of length (n_pixels) containing the class labels for the corresponding pixel in training_data

    Notes
    -----
    For performance, this uses scikit's sparse.nonzero() function to get the location of each training data pixel.
    This means that this will ignore any classes with a label of '0'.

    """
    # TODO: WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong
    # quietly and in a way that'll cause the most issues further on down the line
    FILL_VALUE = -9999
    with TemporaryDirectory() as td:
        # Step 1; rasterise shapefile into .tif of class values
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromEPSG(shape_projection_id)
        image = gdal.Open(image_path)
        image_gt = image.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5]
        ras_path = os.path.join(td, "poly_ras")
        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute=attribute,
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_Int16,
            outputSRS=shape_projection
        )
        # This produces a rasterised geotiff that's right, but not perfectly aligned to pixels.
        # This can probably be fixed.
        gdal.Rasterize(ras_path, shape_path, options=ras_params)
        rasterised_shapefile = gdal.Open(ras_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        local_x, local_y = get_local_top_left(image, rasterised_shapefile)
        shape_sparse = sp.coo_matrix(shape_array)
        y, x, features = sp.find(shape_sparse)
        training_data = np.empty((len(features), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        image_view = image_array[:,
                    local_y: local_y + rasterised_shapefile.RasterYSize,
                    local_x: local_x + rasterised_shapefile.RasterXSize
                    ]
        for index in range(len(features)):
            training_data[index, :] = image_view[:, y[index], x[index]]
        return training_data, features


def raster_reclass_binary(img_path, rcl_value, outFn, outFmt='GTiff', write_out=True):
    """
    Takes a raster and reclassifies rcl_value to 1, with all others becoming 0. In-place operation if write_out is True.

    Parameters
    ----------
    img_path
        Path to 1 band input  raster.
    rcl_value
        Integer indication the value that should be reclassified to 1. All other values will be 0.
    outFn
        Output file name.
    outFmt
        Output format. Set to GTiff by default. Other GDAL options available.
    write_out
        Boolean. Set to True by default. Will write raster to disk. If False, only an array is returned

    Returns
    -------
    Reclassifies numpy array
    """
    log = logging.getLogger(__name__)
    log.info('Starting raster reclassification.')
    # load in classification raster
    in_ds = gdal.Open(img_path)
    in_band = in_ds.GetRasterBand(1)
    in_array = in_band.ReadAsArray()

    # reclassify
    in_array[in_array != rcl_value] = 0
    in_array[in_array == rcl_value] = 1

    if write_out:
        driver = gdal.GetDriverByName(outFmt)
        out_ds = driver.Create(outFn, in_band.XSize, in_band.YSize, 1,
                               in_band.DataType)
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


