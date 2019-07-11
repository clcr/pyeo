"""A small set of functions for producing validation points from maps"""

import numpy as np
import gdal
import random
import ogr, osr
from pyeo import core
import logging
import datetime
import json

gdal.UseExceptions()


def create_validation_scenario(in_map_path, out_shapefile_path, target_standard_error, user_accuracies,
                               no_data_class=None, pinned_samples=None):
    log = core.init_log("validation_log.log")
    class_counts = count_pixel_classes(in_map_path, no_data_class)
    log.info("Class counts: {}".format(class_counts))
    sample_size = cal_total_sample_size(target_standard_error, user_accuracies, class_counts)
    log.info("Sample sizes: {}".format(sample_size))
    class_sample_counts = part_fixed_value_sampling(pinned_samples, class_counts, sample_size)
    log.info("Sample counts per class: {}".format(class_sample_counts))

    sample_weights = cal_w_all(class_counts)
    overall_accuracy = cal_sd_for_overall_accuracy(
        weight_dict=sample_weights,
        u_dict=user_accuracies,
        sample_size_dict=class_sample_counts
    )
    for map_class, accuracy in user_accuracies.items():
        ua = cal_sd_for_user_accuracy(accuracy, class_sample_counts[map_class])
        log.info("Accuracy for class {}: {}".format(map_class, ua))
    log.info("Overall accuracy: {}".format(overall_accuracy))
    produce_stratifed_validation_points(in_map_path, out_shapefile_path, class_sample_counts, no_data_class)

    log.info("Validation points at out: {}".format(out_shapefile_path))
    # manifest_path = out_shapefile_path.rsplit(".")[0] + "_manifest.json"
    # save_validation_maifest(manifest_path, class_counts, sample_size, class_sample_counts, target_standard_error,
    #                        user_accuracies)


def count_pixel_classes(map_path, no_data=None):
    """
    Counts pixels in a map. Returns a dictionary of pixels.
    Parameters
    ----------
    map_path: Path to the map to count
    no_data: A value to ignore

    Returns
    -------
    A dictionary of class:count
    """
    map = gdal.Open(map_path)
    map_array = map.GetVirtualMemArray()
    unique, counts = np.unique(map_array, return_counts=True)
    out = dict(zip([str(val) for val in unique], counts))
    out.pop(no_data, "_")   # pop the no data value, but don't worry if there's nothing there.
    map_array=None
    map=None
    return out


def produce_stratifed_validation_points(map_path, out_path, class_sample_counts,
                                        no_data=None, seed=None):
    """Produces a set of stratified validation points from map_path"""
    log = logging.getLogger(__name__)
    log.info("Producing random sampling of {}.".format(map_path))
    log.info("Class sample count: {}".format(class_sample_counts))
    map = gdal.Open(map_path)
    gt = map.GetGeoTransform()
    proj = map.GetProjection()
    map = None
    point_dict = stratified_random_sample(map_path, class_sample_counts, int(no_data), seed)
    save_point_list_to_shapefile(point_dict, out_path, gt, proj)
    log.info("Complete. Output saved at {}.".format(out_path))


def save_point_list_to_shapefile(class_sample_point_dict, out_path, geotransform, projection_wkt):
    """Saves a list of points to a shapefile at out_path. Need the gt and projection of the raster.
    GT is needed to move each point to the centre of the pixel."""
    log = logging.getLogger(__name__)
    log.info("Saving point list to shapefile")
    log.debug("GT: {}\nProjection: {}".format(geotransform, projection_wkt))
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_wkt)
    layer = data_source.CreateLayer("validation_points", srs, ogr.wkbPoint)
    class_field = ogr.FieldDefn("class", ogr.OFTString)
    class_field.SetWidth(24)
    layer.CreateField(class_field)
    for map_class, point_list in class_sample_point_dict.items():
        for point in point_list:
            feature = ogr.Feature(layer.GetLayerDefn())
            coord = core.pixel_to_point_coordinates(point, geotransform)
            offset = geotransform[1]/2   # Adds half a pixel offset so points end up in the center of pixels
            wkt = "POINT({} {})".format(coord[0]+offset, coord[1]-offset) # Never forget about negative y values in gts.
            new_point = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(new_point)
            feature.SetField("class", map_class)
            layer.CreateFeature(feature)
            feature = None
    layer = None
    data_source = None


def stratified_random_sample(map_path, class_sample_count, no_data=None, seed = None):
    """Produces a stratified list of pixel coordinates. WARNING: high mem!"""
    log = logging.getLogger(__name__)
    if not seed:
        seed = datetime.datetime.now().timestamp()
    map = gdal.Open(map_path)
    map_array = map.GetVirtualMemArray()
    class_dict = build_class_dict(map_array, no_data)
    map_array = None
    out_coord_dict = {}
    for map_class, sample_count in class_sample_count.items():
        map_class_samples = random.sample(class_dict[int(map_class)], class_sample_count[map_class])
        out_coord_dict.update({map_class: map_class_samples})
    return out_coord_dict


def build_class_dict(class_array, no_data=None):
    """Returns a dict of coordinates of the following shape:
    [class, coord].
    WARNING: This will take up a LOT of memory!"""
    out_dict = {}
    it = np.nditer(class_array, flags=['multi_index'])
    while not it.finished:
        this_class = int(it.value)
        if this_class == no_data:
            it.iternext()
            continue
        if this_class in out_dict.keys():
            out_dict[this_class].append(it.multi_index)
        else:
            out_dict.update({this_class: [it.multi_index]})
        it.iternext()
    return out_dict



def cal_si(ui):
    si = np.sqrt(ui * (1 - ui))
    return si


def cal_wi(n,total_n):
    wi = float(n/total_n)
    return wi


def cal_w_all(dict_pixel_numbers):
    w_dict = {}
    total_pixel = (sum(dict_pixel_numbers.values()))
    for key in dict_pixel_numbers:
        w_dict[key] = cal_wi(n=dict_pixel_numbers[key], total_n= total_pixel)
    return w_dict


def cal_n_by_prop(weight, sample_size):
    n = round(weight * sample_size)
    return n


def val_to_sd(val):
    sd = val**0.5
    return sd


def cal_val_for_overall_accruacy(weight_dict,u_dict,sample_size_dict):
    sum_val = 0
    for key in u_dict:
        val_i = (weight_dict[key] **2) * u_dict[key] * (1-u_dict[key])/(sample_size_dict[key]-1)
        sum_val += val_i
    return sum_val


def cal_val_for_user_accuracy(u_i,sample_size_i):
    val_user = (u_i*(1-u_i))/(sample_size_i-1)
    return val_user


def cal_sd_for_overall_accuracy(weight_dict, u_dict, sample_size_dict):
    val_overall = cal_val_for_overall_accruacy(weight_dict=weight_dict,u_dict=u_dict,sample_size_dict=sample_size_dict)
    sd_overall = val_to_sd(val_overall)
    return sd_overall


def cal_sd_for_user_accuracy(u_i,sample_size_i):
    val_user = cal_val_for_user_accuracy(u_i=u_i,sample_size_i=sample_size_i)
    sd_user = val_to_sd(val_user)
    return sd_user


def cal_total_sample_size(desired_standard_error, user_accuracy, total_class_sizes, type ='simple'):
    """
    Calculates the number of sample points for a map to get a specified standard error.
    Parameters
    ----------
    desired_standard_error: The desired standard error (between 0 and 1)
    user_accuracy: A dictionary of user accuracies from apriori knowledge
    total_class_sizes: The total number of pixels for each class
    type: whether to use the simple approximation or the full expession from Olofsson eq 13

    Returns
    -------
    The total number of sample points to achieve the specified error
    """
    total_pixel = (sum(total_class_sizes.values()))
    if type == 'simple':
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in user_accuracy:
            S_i = cal_si(user_accuracy[key])
            Wi = cal_wi(n= total_class_sizes[key], total_n=total_pixel)  # proportion of each class
            weighted_U_sum += S_i*Wi
        n = (weighted_U_sum / desired_standard_error) ** 2
    elif type == 'full':
        weighted_U_sum2 = 0
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in user_accuracy:
            S_i = cal_si(user_accuracy[key])
            Wi = cal_wi(n= total_class_sizes[key], total_n=total_pixel)  # proportion of each class
            weighted_U_sum2 += S_i * Wi
            weighted_U_sum += (S_i ** 2) * Wi
        up = (weighted_U_sum2) ** 2
        bottom_right = (1 / total_pixel) * weighted_U_sum

        n = (up / (desired_standard_error ** 2 + bottom_right))
    print('suggested total sample size are:' + str(n))
    return int(np.round(n))


def calc_minimum_n(expected_accuracy, variance_tolerance):
    """
    Calculates the rminimum number of points required to achieve the specified accuracy
    Parameters
    ----------
    expected_accuracy: Between 0 and 1
    variance_tolerance:

    Returns
    -------

    """
    n = expected_accuracy * (1-expected_accuracy) / variance_tolerance
    return n


def allocate_category_sample_sizes(total_sample_size, user_accuracy, class_total_sizes, variance_tolerance,
                                   allocate_type='olofsson'):
    """
    Allocates a number of pixels to sample per class that will fulfil the parameters given

    Parameters
    ----------
    total_sample_size: The total number of validation points requested (from cal_total_sample_size)
    user_accuracy: Dictionary of estimated user accuracies for classes in map (between 0 and 1)
    class_total_sizes: Dictionary of total pixels for each class in user_accuracy
    variance_tolerance: Acceptable vairance between the sample accuary and the data accuracy with a certain sample size
    allocate_type: The allocation strategy to be used. Can be 'equal', 'prop' or 'olofsson'.

    Returns
    -------
    A dictionary of classes and no. pixels per class.

    """
    log = logging.getLogger(__name__)
    minimum_n = {}
    allocated_n = {}
    weight = cal_w_all(class_total_sizes)
    log.info('the weight for each class is: ')
    log.info(weight)
    log.info('-----------------')
    log.info('the minimum sampling number for : ')
    for key in user_accuracy:
        minimum_n_i = calc_minimum_n(expected_accuracy=user_accuracy[key], variance_tolerance=variance_tolerance)
        log.info('      ' + key + ' is: ' + str(round(minimum_n_i)))
        minimum_n[key] = minimum_n_i

    if allocate_type == 'equal' or allocate_type == 'prop':
        for key in user_accuracy:
            if allocate_type == 'equal':
                n = total_sample_size/float(len(user_accuracy.keys()))
            elif allocate_type == 'prop':
                n = cal_n_by_prop(weight=weight[key], sample_size=total_sample_size)
            else:
                continue
            allocated_n[key] = n
            log.info('allocated sampling number for ' + key + ' is: ' + str(allocated_n[key]))

    elif allocate_type == 'olofsson':
        allocated_n = part_fixed_value_sampling(allocated_n,  total_sample_size, weight)
        log.info('allocated sample number under different scenario is: ')
        log.info(allocated_n)
    else:
        raise core.ForestSentinelException("Invalid allocation type: valid values are 'equal', 'prop' or 'olofsson")
    return allocated_n


def part_fixed_value_sampling(pinned_sample_numbers, class_total_sizes, total_sample_size):
    """

    Parameters
    ----------
    pinned_sample_numbers
    class_total_sizes
    total_sample_size

    Returns
    -------

    """
    pinned_sample_total = sum(sample_size for sample_size in pinned_sample_numbers.values() if sample_size is not None)
    pinned_map_total = sum(map_sample_size for map_class, map_sample_size
                           in class_total_sizes.items() if pinned_sample_numbers[map_class] is not None)
    total_map_size = sum(class_total_sizes.values())
    remaining_map_size = total_map_size - pinned_map_total
    remaining_samples = total_sample_size - pinned_sample_total
    out_values = {}
    weights = {}
    for map_class, sample_points in pinned_sample_numbers.items():
        if sample_points is not None:
            out_values.update({map_class: sample_points})
        else:
            class_proportion = class_total_sizes[map_class]/remaining_map_size
            sample_points = int(np.round(class_proportion*remaining_samples))
            out_values.update({map_class: sample_points})

    return out_values


def save_validation_maifest(out_path, class_counts, sample_size, class_sample_counts, target_standard_error,
                            user_accuracies):
    """Creates a json file containing the parameters used to produce this validation set"""
    out_dict = {
        "class_counts": class_counts,
        "sample_size": sample_size,
        "samples_per_class": class_sample_counts,
        "target_error": target_standard_error,
        "user_accuracies": user_accuracies
    }
    with open(out_path, "w") as fp:
        json.dump(out_dict, fp)

