"""
pyeo.coordinate_manipulation
============================
Contains a set of functions for transforming spatial coordinates between projections and pixel indicies.

Unless otherwise stated, all functions assume that any geometry, rasters and shapefiles are using the same projection.
If they are not, there may be unexpected errors.

Some of these functions call for an AOI shapefile. This is a single-layer shapefile containing only the geometry
of one polygon.

These functions mainly work on the objects provided by the ogr and gdal libraries. If you wish to use them in your own
processing, a ``gdal.Image`` object is usually the output from gdal.Open() and an ogr.Geometry object can be obtained from
a well-known text (wkt) string using the  snippet ``object=ogr.ImportFromWkt("mywkt")``. For more information on wkt, see
https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry and the "QuickWKT" QGIS plugin.

Key functions
-------------
:py:func:`pixel_bounds_from_polygon` Gets the indicies of the bounding box of a polygon within a raster

Function reference
------------------
"""
import subprocess

import numpy as np
from osgeo import osr, ogr
import logging
log = logging.getLogger("pyeo")
import pyeo.windows_compatability

def reproject_geotransform(in_gt, old_proj_wkt, new_proj_wkt):
    """
    Reprojects a geotransform from the old projection to a new projection. See
    [https://gdal.org/user/raster_data_model.html]

    Parameters
    ----------
    in_gt : array_like
        A six-element numpy array, usually an output from gdal_image.GetGeoTransform()
    old_proj_wkt : str
        The projection of the old geotransform in well-known text.
    new_proj_wkt : str
        The projection of the new geotrasform in well-known text.
    Returns
    -------
    out_gt : array_like
        The geotransform in the new projection

    """
    old_proj = osr.SpatialReference()
    new_proj = osr.SpatialReference()
    old_proj.ImportFromWkt(old_proj_wkt)
    new_proj.ImportFromWkt(new_proj_wkt)
    transform = osr.CoordinateTransformation(old_proj, new_proj)
    (ulx, uly, _) = transform.TransformPoint(in_gt[0], in_gt[3])
    out_gt = (ulx, in_gt[1], in_gt[2], uly, in_gt[4], in_gt[5])
    return out_gt


def get_combined_polygon(rasters, geometry_mode ="intersect"):
    """
    Returns a polygon containing the combined boundary of each raster in rasters.

    Parameters
    ----------
    rasters : List of gdal.Image
        A list of raster objects opened with gdal.Open()
    geometry_mode : {'intersect', 'union'}
        If 'intersect', returns the boundary of the area that all rasters cover.
        If 'union', returns the boundary of the area that any raster covers.
    Returns
    -------
    combination : ogr.Geometry
        ogr.Geometry() containing a polygon.

    """
    raster_bounds = []
    for in_raster in rasters:
        raster_bounds.append(get_raster_bounds(in_raster))
    # Calculate overall bounding box based on either union or intersection of rasters
    if geometry_mode == "intersect":
        combined_polygons = multiple_intersection(raster_bounds)
    elif geometry_mode == "union":
        combined_polygons = multiple_union(raster_bounds)
    else:
        raise Exception("Invalid geometry mode")
    return combined_polygons


def multiple_union(polygons):
    """
    Takes a list of polygons and returns a polygon of the union of their perimeter

    Parameters
    ----------
    polygons : list of ogr.Geometry
        A list of ogr.Geometry objects, each containing a single polygon.

    Returns
    -------
    union : ogr.Geometry
        An ogr.Geometry object containing a single polygon

    """
    # Note; I can see this maybe failing(or at least returning a multipolygon)
    # if two consecutive polygons do not overlap at all. Keep eye on.
    running_union = polygons[0]
    for polygon in polygons[1:]:
        running_union = running_union.Union(polygon)
    return running_union.Simplify(0)


def multiple_intersection(polygons):
    """
    Takes a list of polygons and returns a geometry representing the intersection of all of them

    Parameters
    ----------
    polygons : list of ogr.Geometry
        A list of ogr.Geometry objects, each containing a single polygon.

    Returns
    -------
    intersection : ogr.Geometry
        An ogr.Geometry object containing a single polygon
    """
    running_intersection = polygons[0]
    for polygon in polygons[1:]:
        running_intersection = running_intersection.Intersection(polygon)
    return running_intersection.Simplify(0)


def pixel_bounds_from_polygon(raster, polygon):
    """
    Returns the bounding box of the overlap between a raster and a polygon in the raster

    Parameters
    ----------
    raster : gdal.Image
        A gdal raster object

    polygon : ogr.Geometry or str
        A ogr.Geometry object or wkt string containing a single polygon

    Returns
    -------
    x_min, x_max, y_min, y_max : int
        The bounding box in the pixels of the raster

    """
    if type(polygon) == str:
        polygon = ogr.CreateGeometryFromWkt(polygon)
    raster_bounds = get_raster_bounds(raster)
    intersection = get_poly_intersection(raster_bounds, polygon)
    bounds_geo = intersection.Boundary()
    x_min_geo, x_max_geo, y_min_geo, y_max_geo = bounds_geo.GetEnvelope()
    (x_min_pixel, y_min_pixel) = point_to_pixel_coordinates(raster, (x_min_geo, y_min_geo))
    (x_max_pixel, y_max_pixel) = point_to_pixel_coordinates(raster, (x_max_geo, y_max_geo))
    # Kludge time: swap the two values around if they are wrong
    if x_min_pixel >= x_max_pixel:
        x_min_pixel, x_max_pixel = x_max_pixel, x_min_pixel
    if y_min_pixel >= y_max_pixel:
        y_min_pixel, y_max_pixel = y_max_pixel, y_min_pixel
    return x_min_pixel, x_max_pixel, y_min_pixel, y_max_pixel


def point_to_pixel_coordinates(raster, point, oob_fail=False):
    """
    Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the geographic point in a projection.
     Assumes raster is north-up non rotated.

    Parameters
    ----------
    raster : gdal.Image
        A gdal raster object
    point : str, iterable or ogr.Geometry
        One of:
            A well-known text string of a single point
            An iterable of the form (x,y)
            An ogr.Geometry object containing a single point
    Returns
    -------
    A tuple of (x_pixel, y_pixel), containing the indicies of the point in the raster.

    Notes
    -----
    The equation is a rearrangement of the section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html

    """
    if isinstance(point, str):
        point = ogr.CreateGeometryFromWkt(point)
        x_geo = point.GetX()
        y_geo = point.GetY()
    if isinstance(point, list) or isinstance(point, tuple):  # There is a more pythonic way to do this
        x_geo = point[0]
        y_geo = point[1]
    if isinstance(point, ogr.Geometry):
        x_geo = point.GetX()
        y_geo = point.GetY()
    gt = raster.GetGeoTransform()
    x_pixel = int(np.floor((x_geo - floor_to_resolution(gt[0], gt[1]))/gt[1]))
    y_pixel = int(np.floor((y_geo - floor_to_resolution(gt[3], gt[5]*-1))/gt[5]))  # y resolution is -ve
    return x_pixel, y_pixel


def pixel_to_point_coordinates(pixel, GT):
    """
    Given a pixel and a geotransformation, returns the picaltion of that pixel's top left corner in the projection
    used by the geotransform.
    NOTE: At present, this takes input in the form of y,x! This is opposite to the output of point_to_pixel_coordinates!

    Parameters
    ----------
    pixel : iterable
        A tuple (y, x) of the coordinates of the pixel
    GT : iterable
        A six-element numpy array containing a geotransform

    Returns
    -------
    Xgeo, Ygeo : number
        The geographic coordinates of the top-left corner of the pixel.

    """
    Xpixel = pixel[1]
    Yline = pixel[0]
    Xgeo = GT[0] + Xpixel * GT[1] + Yline * GT[2]
    Ygeo = GT[3] + Xpixel * GT[4] + Yline * GT[5]
    return Xgeo, Ygeo


def write_geometry(geometry, out_path, srs_id=4326):
    """
    Saves the geometry in an ogr.Geometry object to a shapefile.

    Parameters
    ----------
    geometry : ogr.Geometry
        An ogr.Geometry object
    out_path : str
        The location to save the output shapefile
    srs_id : int or str, optional
        The projection of the output shapefile. Can be an EPSG number or a WKT string.

    Notes
    -----
    The shapefile consists of one layer named 'geometry'.


    """
    # TODO: Fix this needing an extra filepath on the end
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(out_path)
    srs = osr.SpatialReference()
    if type(srs_id) is int:
        srs.ImportFromEPSG(srs_id)
    if type(srs_id) is str:
        srs.ImportFromWkt(srs_id)
    layer = data_source.CreateLayer(
        "geometry",
        srs,
        geom_type=geometry.GetGeometryType())
    feature_def = layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)
    data_source.FlushCache()
    data_source = None


def get_vector_projection(dataset):
    # from Layer
    layer = dataset.GetLayer()
    spatialRef = layer.GetSpatialRef()
    return spatialRef


def reproject_vector(in_path, out_path, dest_srs):
    """
    Reprojects a vector file to a new SRS. Simple wrapper for ogr2ogr.

    Parameters
    ----------
    in_path : str
        Path to the vector file
    out_path : str
        Path to output vector file
    dest_srs : osr.SpatialReference or str
        The spatial reference system to reproject to

    """

    if type(dest_srs) == int:
        epsg_no = dest_srs
        dest_srs = osr.SpatialReference()
        dest_srs.ImportFromEPSG(epsg_no)
    if type(dest_srs) == osr.SpatialReference:
        dest_srs = dest_srs.ExportToWkt()

    subprocess.run(["ogr2ogr", out_path, in_path, '-t_srs', dest_srs])


def get_aoi_intersection(raster, aoi):
    """
    Returns a wkbPolygon geometry with the intersection of a raster and a shpefile containing an area of interest

    Parameters
    ----------
    raster : gdal.Image
        A raster containing image data
    aoi : ogr.DataSource
        A datasource with a single layer and feature
    Returns
    -------
    intersection : ogr.Geometry
        An ogr.Geometry object containing a single polygon with the area of intersection

    """
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)


def get_raster_intersection(raster1, raster2):
    """
    Returns a wkbPolygon geometry with the intersection of two raster bounding boxes.

    Parameters
    ----------
    raster1, raster2 : gdal.Image
        The gdal.Image
    Returns
    -------
    intersections : ogr.Geometry
        a ogr.Geometry object containing a single polygon

    """
    bounds_1 = get_raster_bounds(raster1)
    bounds_2 = get_raster_bounds(raster2)
    return bounds_1.Intersection(bounds_2)


def get_poly_intersection(poly1, poly2):
    """
    A functional wrapper for ogr.Geometry.Intersection()

    Parameters
    ----------
    poly1, poly2: ogr.Geometry
        The two geometries to intersect
    Returns
    -------
    intersection : ogr.Geometry
        A polygon of the intersection

    """
    return poly1.Intersection(poly2)


def check_overlap(raster, aoi):
    """
    A test to see if a raster and an AOI overlap.
    Parameters
    ----------
    raster : gdal.Image
        A gdal.Image object
    aoi : ogr.Dataset
        A ogr.Dataset object containing a single polygon
    Returns
    -------
    is_overlapped : bool
        True if the raster and the polygon overlap, otherwise False.

    """
    raster_shape = get_raster_bounds(raster)
    aoi_shape = get_aoi_bounds(aoi)
    if raster_shape.Intersects(aoi_shape):
        return True
    else:
        return False


def get_raster_bounds(raster):
    """
    Returns a wkbPolygon geometry with the bounding rectangle of a raster calculated from its geotransform.

    Parameters
    ----------
    raster : gdal.Image
        A gdal.Image object

    Returns
    -------
    boundary : ogr.Geometry
        An ogr.Geometry object containing a single wkbPolygon with four points defining the bounding rectangle of the
        raster.

    Notes
    -----
    Bounding rectangle is obtained from raster.GetGeoTransform(), with the top left corners rounded
    down to the nearest multiple of of the resolution of the geotransform. This is to avoid rounding errors in
    reprojected geotransformations.
    """
    raster_bounds = ogr.Geometry(ogr.wkbLinearRing)
    geotrans = raster.GetGeoTransform()
    # We can't rely on the top-left coord being whole numbers any more, since images may have been reprojected
    # So we floor to the resolution of the geotransform maybe?
    top_left_x = floor_to_resolution(geotrans[0], geotrans[1])
    top_left_y = floor_to_resolution(geotrans[3], geotrans[5]*-1)
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize * -1  # RasterYSize is +ve, but geotransform is -ve
    raster_bounds.AddPoint(top_left_x, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(raster_bounds)
    return bounds_poly


def floor_to_resolution(input, resolution):
    """
    Returns input rounded DOWN to the nearest multiple of resolution. Used to prevent float errors on pixel boarders.

    Parameters
    ----------
    input : number
        The value to be rounded
    resolution : number
        The resolution

    Returns
    -------
    output : number
        The largest value between input and 0 that is wholly divisible by resolution.

    Notes
    -----
    Uses the following formula: ``input-(input%resolution)``
    If resolution is less than 1, then this assumes that the projection is in decmial degrees and will be rounded to 6dp
    before flooring. However, it is not recommended to use this function in that situation.

    """
    if resolution > 1:
        return input - (input%resolution)
    else:
        log.warning("Low resolution detected, assuming in degrees. Rounding to 6 dp.\
                Probably safer to reproject to meters projection.")
        resolution = resolution * 1000000
        input = input * 1000000
        return (input-(input%resolution))/1000000



def get_raster_size(raster):
    """
    Return the width and height of a raster, in that raster's units.

    Parameters
    ----------
    raster : gdal.Image
        A gdal.Image object

    Returns
    -------
    width, height : number
        A tuple containing (width, height)
    """
    geotrans = raster.GetGeoTransform()
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize
    return width, height


def get_aoi_bounds(aoi):
    """
    Returns a wkbPolygon geometry with the bounding rectangle of a single-layer shapefile

    Parameters
    ----------
    aoi : ogr.Dataset
        An ogr.Dataset object containing a single layer.

    Returns
    -------
    bounds_poly : ogr.Geometry
        A polygon containing the bounding rectangle

    """
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    (x_min, x_max, y_min, y_max) = aoi.GetLayer(0).GetExtent()
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_max, y_min)
    aoi_bounds.AddPoint(x_max, y_max)
    aoi_bounds.AddPoint(x_min, y_max)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def align_bounds_to_whole_number(bounding_box):
    """
    Creates a new bounding box with it's height and width rounded to the nearest whole number.

    Parameters
    ----------
    bounding_box : ogr.Geometry
        An ogr.Geometry object containing a raster's bounding box as a polygon.

    Returns
    -------
    bounds_poly : ogr.Geometry
        An ogr.Geometry object containing the aligned bounding box.

    """
    # This should prevent off-by-one errors caused by bad image alignment
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    (x_min, x_max, y_min, y_max) = bounding_box.GetEnvelope()
    # This will create a box that has a whole number as its height and width
    x_new = x_min + np.floor(x_max-x_min)
    y_new = y_min + np.floor(y_max-y_min)
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_new, y_min)
    aoi_bounds.AddPoint(x_new, y_new)
    aoi_bounds.AddPoint(x_min, y_new)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def get_aoi_size(aoi):
    """
    Returns the width and height of the bounding box of an aoi.

    Parameters
    ----------
    aoi : ogr.Geometry
        A shapefile containing a single layer with a single polygon

    Returns
    -------
    width, height : number
        Width and height of the bounding box

    """
    (x_min, x_max, y_min, y_max) = aoi.GetLayer(0).GetExtent()
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_size(poly):
    """
    Returns the width and height of a bounding box of a polygon

    Parameters
    ----------
    poly : ogr.Geometry
        A ogr.Geometry object containing the polygon.

    Returns
    -------
    width, height : number
        Width and height of the polygon

    """
    boundary = poly.Boundary()
    x_min, y_min, not_needed = boundary.GetPoint(0)
    x_max, y_max, not_needed = boundary.GetPoint(2)
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_bounding_rect(poly):
    """
    Returns a polygon of the bounding rectangle of an input polygon.
    Parameters
    ----------
    poly : ogr.Geometry
        An ogr.Geometry object containing a polygon

    Returns
    -------
    bounding_rect : ogr.Geometry
        An ogr.Geometry object with a four-point polygon representing the bounding rectangle.

    """
    aoi_bounds = ogr.Geometry(ogr.wkbLinearRing)
    x_min, x_max, y_min, y_max = poly.GetEnvelope()
    aoi_bounds.AddPoint(x_min, y_min)
    aoi_bounds.AddPoint(x_max, y_min)
    aoi_bounds.AddPoint(x_max, y_max)
    aoi_bounds.AddPoint(x_min, y_max)
    aoi_bounds.AddPoint(x_min, y_min)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(aoi_bounds)
    return bounds_poly


def get_local_top_left(raster1, raster2):
    """
    Gets the top-left corner of raster1 in the array of raster 2.
    Assumes both rasters are in the same projection and units.

    Parameters
    ----------
    raster1 : gdal.Image
        The raster to get the top-left corner of.
    raster2 : gdal.Image
        The raster that raster1's top-left corner is over.

    Returns
    -------
    x_pixel, y_pixel : int
        The indicies of the pixel of top-left corner of raster 1 in raster 2.

    """
    # TODO: Test this
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])
