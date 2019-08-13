"""
pyeo.classification
-------------------
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

def change_from_composite(image_path, composite_path, model_path, class_out_path, prob_out_path):
    """Generates a change map comparing an image with a composite"""
    with TemporaryDirectory() as td:
        stacked_path = os.path.join(td, "comp_stack.tif")
        stack_images((composite_path, image_path), stacked_path)
        classify_image(stacked_path, model_path, class_out_path, prob_out_path)


def classify_image(image_path, model_path, class_out_path, prob_out_path=None,
                   apply_mask=False, out_type="GTiff", num_chunks=10, nodata=0, skip_existing = False):
    """
    Classifies change between two stacked images.
    Images need to be chunked, otherwise they cause a memory error (~16GB of data with a ~15GB machine)
    TODO: This has gotten very hairy; rewrite when you update this to take generic models
    """
    print("Hi, develsetup works")
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
    """Calculates the number of chunks to break a dataset into without a memory error.
    We want to break the dataset into as few chunks as possible without going over mem_limit.
    mem_limit defaults to total amount of RAM available on machine if not specified"""
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


def classify_directory(in_dir, model_path, class_out_dir, prob_out_dir,
                       apply_mask=False, out_type="GTiff", num_chunks=None):
    """
    Classifies every .tif in in_dir using model at model_path. Outputs are saved
    in class_out_dir and prob_out_dir, named [input_name]_class and _prob, respectively.
    """
    log = logging.getLogger(__name__)
    log.info("Classifying files in {}".format(in_dir))
    log.info("Class files saved in {}".format(class_out_dir))
    log.info("Prob. files saved in {}".format(prob_out_dir))
    for image_path in glob.glob(in_dir+r"/*.tif"):
        image_name = os.path.basename(image_path).split('.')[0]
        class_out_path = os.path.join(class_out_dir, image_name+"_class.tif")
        prob_out_path = os.path.join(prob_out_dir, image_name+"_prob.tif")
        classify_image(image_path, model_path, class_out_path, prob_out_path,
                       apply_mask, out_type, num_chunks)


def reshape_raster_for_ml(image_array):
    """Reshapes an array from gdal order [band, y, x] to scikit order [x*y, band]"""
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def reshape_ml_out_to_raster(classes, width, height):
    """Reshapes an output [x*y] to gdal order [y, x]"""
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array


def reshape_prob_out_to_raster(probs, width, height):
    """reshapes an output of shape [x*y, classes] to gdal order [classes, y, x]"""
    classes = probs.shape[1]
    image_array = np.transpose(probs, (1, 0))
    image_array = np.reshape(image_array, (classes, height, width))
    return image_array

def extract_features_to_csv(in_ras_path, training_shape_path, out_path):
    this_training_data, this_classes = get_training_data(in_ras_path, training_shape_path)
    sigs = np.vstack((this_classes, this_training_data.T))
    with open(out_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(sigs.T)

def create_trained_model(training_image_file_paths, cross_val_repeats = 5, attribute="CODE"):
    """Returns a trained random forest model from the training data. This
    assumes that image and model are in the same directory, with a shapefile.
    Give training_image_path a path to a list of .tif files. See spec in the R drive for data structure.
    At present, the model is an ExtraTreesClassifier arrived at by tpot; see tpot_classifier_kenya -> tpot 1)"""
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
    """Creates a model based on training data for files in a given region"""
    image_glob = os.path.join(path_to_region, r"*.tif")
    image_list = glob.glob(image_glob)
    model, scores = create_trained_model(image_list, attribute=attribute)
    joblib.dump(model, model_out)
    with open(scores_out, 'w') as score_file:
        score_file.write(str(scores))


def create_model_from_signatures(sig_csv_path, model_out):
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
                                     min_samples_split=16, n_estimators=100, n_jobs=4, class_weight='balanced')
    data = np.loadtxt(sig_csv_path, delimiter=",").T
    model.fit(data[1:, :].T, data[0, :])
    joblib.dump(model, model_out)


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'."""
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
    """Takes a raster and reclassifies the values


    :param str img_path: Path to 1 band input  raster.
    :param int rcl_value: Integer indication the value that should be reclassified to 1. All other values will be 0.
    :param str outFn: Output file name.
    :param str outFmt: Output format. Set to GTiff by default. Other GDAL options available.
    :param write_out: Boolean. Set to True by default. Will write raster to disk. If False, only an array is returned
    :return: Reclassifies numpy array
    """
    log = logging.getLogger(__name__)
    log.info('Starting raster reclassifcation.')
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


