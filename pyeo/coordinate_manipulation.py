import numpy as np
from osgeo import osr, ogr


def reproject_geotransform(in_gt, old_proj_wkt, new_proj_wkt):
    """Reprojects a geotransform into a new projection."""
    old_proj = osr.SpatialReference()
    new_proj = osr.SpatialReference()
    old_proj.ImportFromWkt(old_proj_wkt)
    new_proj.ImportFromWkt(new_proj_wkt)
    transform = osr.CoordinateTransformation(old_proj, new_proj)
    (ulx, uly, _) = transform.TransformPoint(in_gt[0], in_gt[3])
    out_gt = (ulx, in_gt[1], in_gt[2], uly, in_gt[4], in_gt[5])
    return out_gt


def get_combined_polygon(rasters, geometry_mode ="intersect"):
    """Calculates the overall polygon boundary for multiple rasters"""
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
    """Takes a list of polygons and returns a geometry representing the union of all of them"""
    # Note; I can see this maybe failing(or at least returning a multipolygon)
    # if two consecutive polygons do not overlap at all. Keep eye on.
    running_union = polygons[0]
    for polygon in polygons[1:]:
        running_union = running_union.Union(polygon)
    return running_union.Simplify(0)


def pixel_bounds_from_polygon(raster, polygon):
    """Returns the pixel coordinates of the bounds of the
     intersection between polygon and raster """
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
    """Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the point.
    Point can be an ogr point object, a wkt string or an x, y tuple or list. Assumes north-up non rotated.
    Will floor() decimal output"""
    # Equation is rearrangement of section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html
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
    """Converts a pixel to it's coords implied by geotransform GT.
    NOTE: This will not give the pixels precise location on the earth's surface;
    you'll also need to make sure tha this is in the correct projection"""
    Xpixel = pixel[1]
    Yline = pixel[0]
    Xgeo = GT[0] + Xpixel * GT[1] + Yline * GT[2]
    Ygeo = GT[3] + Xpixel * GT[4] + Yline * GT[5]
    return Xgeo, Ygeo


def multiple_intersection(polygons):
    """Takes a list of polygons and returns a geometry representing the intersection of all of them"""
    running_intersection = polygons[0]
    for polygon in polygons[1:]:
        running_intersection = running_intersection.Intersection(polygon)
    return running_intersection.Simplify(0)


def write_geometry(geometry, out_path, srs_id=4326):
    """Saves a polygon to a shapefile"""
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


def get_aoi_intersection(raster, aoi):
    """Returns a wkbPolygon geometry with the intersection of a raster and an aoi"""
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)


def get_raster_intersection(raster1, raster2):
    """Returns a wkbPolygon geometry with the intersection of two raster bounding boxes"""
    bounds_1 = get_raster_bounds(raster1)
    bounds_2 = get_raster_bounds(raster2)
    return bounds_1.Intersection(bounds_2)


def get_poly_intersection(poly1, poly2):
    """Trivial function returns the intersection between two polygons. No test."""
    return poly1.Intersection(poly2)


def check_overlap(raster, aoi):
    """Checks that a raster and an AOI overlap"""
    raster_shape = get_raster_bounds(raster)
    aoi_shape = get_aoi_bounds(aoi)
    if raster_shape.Intersects(aoi_shape):
        return True
    else:
        return False


def get_raster_bounds(raster):
    """Returns a wkbPolygon geometry with the bounding rectangle of a raster calculated from its geotransform"""
    raster_bounds = ogr.Geometry(ogr.wkbLinearRing)
    geotrans = raster.GetGeoTransform()
    # We can't rely on the top-left coord being whole numbers any more, since images may have been reprojected
    # So we floor to the resolution of the geotransform maybe?
    top_left_x = floor_to_resolution(geotrans[0], geotrans[1])
    top_left_y = floor_to_resolution(geotrans[3], geotrans[5]*-1)
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize * -1  # RasterYSize is +ve, but geotransform is -ve so this should go good
    raster_bounds.AddPoint(top_left_x, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y)
    raster_bounds.AddPoint(top_left_x + width, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y - height)
    raster_bounds.AddPoint(top_left_x, top_left_y)
    bounds_poly = ogr.Geometry(ogr.wkbPolygon)
    bounds_poly.AddGeometry(raster_bounds)
    return bounds_poly


def floor_to_resolution(input, resolution):
    """Returns input rounded DOWN to the nearest multiple of resolution."""
    return input - (input%resolution)


def get_raster_size(raster):
    """Return the height and width of a raster"""
    geotrans = raster.GetGeoTransform()
    width = geotrans[1]*raster.RasterXSize
    height = geotrans[5]*raster.RasterYSize
    return width, height


def get_aoi_bounds(aoi):
    """Returns a wkbPolygon geometry with the bounding rectangle of a single-polygon shapefile"""
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
    """Shifts bounding_box polygon so that its size becomes a whole number"""
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
    """Returns the width and height of the bounding box of an aoi. No test"""
    (x_min, x_max, y_min, y_max) = aoi.GetLayer(0).GetExtent()
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_size(poly):
    """Returns the width and height of a bounding box of a polygon. No test"""
    boundary = poly.Boundary()
    x_min, y_min, not_needed = boundary.GetPoint(0)
    x_max, y_max, not_needed = boundary.GetPoint(2)
    out = (x_max - x_min, y_max-y_min)
    return out


def get_poly_bounding_rect(poly):
    """Returns a polygon of the bounding rectangle of input polygon. Can probably be combined with
    get_aoi_bounds."""
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
    """Gets the top-left corner of raster1 in the array of raster 2; WRITE A TEST FOR THIS"""
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])