"""A small set of functions for producing validation points from maps"""

import numpy as np
import gdal
import random
import ogr, osr
from pyeo import core

gdal.UseExceptions()

def random_sample(map_path, n_points):
    """Produces a random set of points from a map"""

    map = gdal.Open(map_path)
    map_array = map.ReadAsArray()
    point_array = []
    for point in n_points:
        point



def save_point_list_to_shapefile(point_list, out_path, geotransform, projection_wkt):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection_wkt)
    layer = data_source.CreateLayer("validation_points", srs, ogr.wkbPoint)
    for point in point_list:
        feature = ogr.Feature(layer.GetLayerDefn())
        coord = core.pixel_to_point_coordinates(point, geotransform)
        wkt = "POINT({} {})".format(coord[0], coord[1])
        new_point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(new_point)
        layer.CreateFeature(feature)
        feature = None
    layer = None
    data_source = None


def stratified_random_sample(map_path, n_points):
    """Produces a list of pixel coords. Takes about 1 minute on ~6GB RAM"""
    map = gdal.Open(map_path)
    map_array = map.ReadAsArray()
    class_dict = build_class_dict(map_array, no_data=0)
    map_array = None
    n_pixels = sum(len(coord_list) for coord_list in class_dict.values())
    out_coord_list = []
    for pixel_class, coord_list in class_dict.items():
        proportion = len(coord_list)/n_pixels
        n_sample_pixels = int(np.round(proportion*n_points))
        out_coord_list.extend(random.sample(coord_list, n_sample_pixels))
    return out_coord_list


def build_class_dict(class_array, no_data = None):
    """Returns a dict of coordinates of the following shape:
    [class, coord]"""
    out_dict = {}
    it = np.nditer(class_array, flags=['multi_index'])
    while not it.finished:
        this_class = int(it.value)
        if this_class == no_data:
            it.iternext()
        if this_class in out_dict.keys():
            out_dict[this_class].append(it.multi_index)
        else:
            out_dict.update({this_class: [it.multi_index]})
        it.iternext()
    return out_dict
