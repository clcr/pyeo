"""A small set of functions for producing validation points from maps"""

import numpy as np
import gdal
import random
import ogr, osr
from pyeo import core
import logging
import datetime

from pyeo.apps.validation.sample_allocation import U

gdal.UseExceptions()


def produce_stratifed_validation_points(map_path, n_points, out_path, no_data=None, seed=None):
    """Produces a set of stratified validation points from map_path"""
    log = logging.getLogger(__name__)
    log.info("Producing random sampling of {} with {} points.".format(map_path, n_points))
    map = gdal.Open(map_path)
    gt = map.GetGeoTransform()
    proj = map.GetProjection()
    map = None
    point_list = stratified_random_sample(map_path, n_points, no_data, seed)
    save_point_list_to_shapefile(point_list, out_path, gt, proj)
    log.info("Complete. Output saved at {}.".format(out_path))


def save_point_list_to_shapefile(point_list, out_path, geotransform, projection_wkt):
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
    for point in point_list:
        feature = ogr.Feature(layer.GetLayerDefn())
        coord = core.pixel_to_point_coordinates(point, geotransform)
        offset = geotransform[1]/2   # Adds half a pixel offset so points end up in the center of pixels
        wkt = "POINT({} {})".format(coord[0]+offset, coord[1]+offset)
        new_point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(new_point)
        layer.CreateFeature(feature)
        feature = None
    layer = None
    data_source = None


def stratified_random_sample(map_path, n_points, no_data=None, seed = None):
    """Produces a stratified list of pixel coordinates. WARNING:"""
    if not seed:
        seed = datetime.datetime.now().timestamp()
    map = gdal.Open(map_path)
    map_array = map.GetVirtualMemArray()
    class_dict = build_class_dict(map_array, no_data)
    map_array = None
    n_pixels = sum(len(coord_list) for coord_list in class_dict.values())
    out_coord_list = []
    for pixel_class, coord_list in class_dict.items():
        proportion = len(coord_list)/n_pixels
        n_sample_pixels = int(np.round(proportion*n_points))
        out_coord_list.extend(random.sample(coord_list, n_sample_pixels))
    return out_coord_list


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


def cal_sd_for_overall_accruacy(weight_dict,u_dict,sample_size_dict):
    val_overall = cal_val_for_overall_accruacy(weight_dict=weight_dict,u_dict=u_dict,sample_size_dict=sample_size_dict)
    sd_overall = val_to_sd(val_overall)
    return sd_overall


def cal_sd_for_user_accuracy(u_i,sample_size_i):
    val_user = cal_val_for_user_accuracy(u_i=u_i,sample_size_i=sample_size_i)
    sd_user = val_to_sd(val_user)
    return sd_user


def cal_total_sample_size(se_expected_overall, U, pixel_numbers, type = 'simple'):
    total_pixel = (sum(pixel_numbers.values()))
    if type == 'simple':
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in U:
            S_i = cal_si(U[key])
            Wi = cal_wi(n= pixel_numbers[key], total_n=total_pixel)# proportion of each class
            weighted_U_sum += S_i*Wi
        n = (weighted_U_sum/se_expected_overall)**2
    elif type == 'full':
        weighted_U_sum2 = 0
        weighted_U_sum = 0
        # weight are equal between different classes
        for key in U:
            S_i = cal_si(U[key])
            Wi = cal_wi(n= pixel_numbers[key], total_n=total_pixel)  # proportion of each class
            weighted_U_sum2 += S_i * Wi
            weighted_U_sum += (S_i ** 2) * Wi
        up = (weighted_U_sum2) ** 2
        bottom_right = (1 / total_pixel) * weighted_U_sum

        n = (up / (se_expected_overall ** 2 + bottom_right))
    print('suggested total sample size are:' + str(n))
    return n


def cal_minum_n(expected_accuracy,required_val):
    n = expected_accuracy*(1-expected_accuracy)/required_val
    return n


def allocate(total_sample_size, user_accuracy, pixel_numbers, required_val, allocate_type= 'olofsson'):
    """
    Allocates a number of pixels to sample per class that will fulfil the parameters given

    Parameters
    ----------
    total_sample_size: The total number of validation points requested
    user_accuracy: Dictionary of estimated user accuracies for classes in map (between 0 and 1)
    pixel_numbers: Dictionary of total pixels for each class in user_accuracy
    required_val: ???? Qing, please help here. What does this mean?
    allocate_type: The allocation strategy to be used. Can be 'equal', 'prop' or 'olofsson'.

    Returns
    -------
    A dictionary of classes and no. pixels per class.

    """
    minimum_n = {}
    allocated_n = {}
    weight = cal_w_all(pixel_numbers)
    print('the weight for each class is: ')
    print(weight)
    print('-----------------')
    print('the minimum sampling number for : ')
    for key in user_accuracy:
        minum_n_i = cal_minum_n(expected_accuracy=U[key], required_val=required_val)
        print('      ' + key + ' is: ' + str(round(minum_n_i)))
        minimum_n[key] = minum_n_i

    if allocate_type == 'equal' or allocate_type == 'prop':
        for key in user_accuracy:
            if allocate_type == 'equal':
                n = total_sample_size/float(len(user_accuracy.keys()))
            elif allocate_type == 'prop':
                n = cal_n_by_prop(weight=weight[key], sample_size=total_sample_size)
            else:
                continue
            allocated_n[key] = n
            print('allocated sampling number for ' + key + ' is: ' + str(allocated_n[key]))

    if allocate_type == 'olofsson':
        pre_allocated_n = {'alloc1': {'defore': 100, 'gain': 100},
                           'alloc2': {'defore': 75, 'gain': 75},
                           'alloc3': {'defore': 50, 'gain': 50}}

        for method in pre_allocated_n:  # for each allocation method
            already_allocated = sum(pre_allocated_n[method].values())
            remaining_sample_size = total_sample_size - already_allocated

            w_stable_forest = weight['stable_forest']/(weight['stable_forest'] + weight['stable_nonforest'] )

            w_stable_nonforest = weight['stable_nonforest']/(weight['stable_nonforest'] + weight['stable_forest'])
            pre_allocated_n[method]['stable_forest'] = cal_n_by_prop(w_stable_forest, remaining_sample_size)
            pre_allocated_n[method]['stable_nonforest'] = cal_n_by_prop(w_stable_nonforest, remaining_sample_size)

            allocated_n[method] = pre_allocated_n[method]
        print('allocated sample number under different scenario is: ')
        print(allocated_n)
    return allocated_n