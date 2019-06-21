"""A small set of functions for producing validation points from maps"""

import numpy as np
import gdal
import random
import ogr, osr
from pyeo import core

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
    [class, coord]"""
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
