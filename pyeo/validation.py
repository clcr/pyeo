"""A small set of functions for producing validation points from maps"""

import numpy as np
import gdal
import random
import ogr, osr
from pyeo import core
import datetime
import logging
import os
from tempfile import TemporaryDirectory

gdal.UseExceptions()


def produce_stratifed_validation_points(map_path, n_points, out_path, no_data=None, seed=None):
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
    """Produces a list of pixel coords. Takes about 1 minute on ~6GB RAM"""
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
    [class, coord_list]"""
    # Here be hacks to make this fit inside a regular computer.
    # Iterate through every pixel in class_array, keeping track of the index.
    # For each pixel that's not nodata:
    #   if first time seeing this class, create a laaaaarge memmap in a temporary directory.
    #   A memmap is like an numpy array, but stored in the filesystem instead of on RAM. SLower, but easier on the machine
    #   Also keep an index array tracking where we got up to with each memmap
    log = logging.getLogger(__name__)
    log.info("Sampling....")
    memmap_dict = {}
    ind_dict = {}
    it = np.nditer(class_array, flags=['multi_index'])
    with TemporaryDirectory() as td:
        while not it.finished:
            this_class = int(it.value)
            if this_class == no_data:
                it.iternext()
                continue
            if this_class in memmap_dict.keys():
                this_index = ind_dict[this_class]
                memmap_dict[this_class][this_index] = it.multi_index[0]
                memmap_dict[this_class][this_index + 1] = it.multi_index[1]
                ind_dict[this_class] += 2
            else:
                class_map = np.memmap(os.path.join(td, str(this_class)), mode="w+", shape=10000000)   # Wheeeee, kludge
                class_map[0] = it.multi_index[0]
                class_map[1] = it.multi_index[1]
                memmap_dict.update({this_class: class_map})
                ind_dict.update({this_class: 2})
            it.iternext()
        class_map = None
        out_dict = convert_memmap_to_tuple_dict(memmap_dict, ind_dict)
    log.info("Sampling done.\n{} classes detected.")
    for pixel_class, coord_list in memmap_dict.items():
        log.info("Class {}: {} samples".format(pixel_class, len(coord_list)))
    return out_dict


def convert_memmap_to_tuple_dict(memmap_dict, ind_dict):
    out_dict = {}
    for pixel_class, memmap in memmap_dict.items():
        last_used_index = ind_dict[pixel_class]-2
        memmap_view = memmap[:last_used_index]
        tuple_list = []
        pair_view = np.reshape(memmap_view, (2, int(memmap_view.size/2)))
        for pair in np.nditer(pair_view, flags=['external_loop'], order='F'):
            tuple_list.append(tuple(pair))
        out_dict.update({pixel_class: tuple_list})
    return out_dict
