import os, sys
from tempfile import TemporaryDirectory
import numpy as np
import gdal, ogr
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..','..')))
import pyeo.core as pyeo


# see _conftest.py for definition of managed_multiple_geotiff_dir
def test_stack_images(managed_multiple_geotiff_dir):
    test_dir = managed_multiple_geotiff_dir
    images = [os.path.join(test_dir.path, image) for image in os.listdir(test_dir.path)]
    images.sort()
    result_path = os.path.join(test_dir.path, "test_out.tif") # This isn't turning up in the expected order
    pyeo.stack_images(images, result_path)
    result = gdal.Open(result_path)
    assert result.ReadAsArray().shape[0] == 4  # If result has 4 bands, success
    assert result.ReadAsArray()[0,0,4] == 3
    assert result.ReadAsArray()[1,0,4] == 13


def test_stack_and_trim_images(managed_noncontiguous_geotiff_dir):
    # Test data is two five band 11x12 pixel geotiffs and a 10x10 polygon
    # The geotiffs upper left corners ar at 90,90 and 100,100
    test_dir = managed_noncontiguous_geotiff_dir
    old_image_path = os.path.join(test_dir.path, "se_test")
    new_image_path = os.path.join(test_dir.path, "ne_test")
    aoi_path = os.path.join(test_dir.path, "aoi")
    result_path = os.path.join(test_dir.path, "test_out.tif")
    pyeo.stack_and_trim_images(old_image_path, new_image_path, aoi_path, result_path)
    result = gdal.Open(result_path).ReadAsArray()
    assert result.shape == (10, 10, 10)  # Result should have 10 bands and be 10 by 10


def test_multiple_union():
    # http://dev.openlayers.org/examples/vector-formats.html to test the wkt
    test_polys = [
        ogr.CreateGeometryFromWkt(r"POLYGON((10 10, 12 12, 12 10, 10 10))"),
        ogr.CreateGeometryFromWkt(r"POLYGON((11 11, 13 13, 13 11, 11 11))")
    ]
    target = ogr.CreateGeometryFromWkt(
        r"POLYGON((10 10, 13 13, 13 11, 12 11, 12 10, 10 10))"
    )
    out = pyeo.multiple_union(test_polys)
    assert out.__str__() == target.__str__()


def test_pixel_bounds_from_polygon(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    raster = gdal.Open(os.path.join(test_dir.path, "ne_test"))
    aoi = ogr.Open(os.path.join(test_dir.path, "aoi"))
    layer = aoi.GetLayer(0)
    aoi_feature = layer.GetFeature(0)
    polygon = aoi_feature.GetGeometryRef()
    result = pyeo.pixel_bounds_from_polygon(raster, polygon)
    assert result == (1, 11, 1, 11)


def test_point_to_pixel_coordinates(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    point = r"POINT (101 112)"
    raster = gdal.Open(os.path.join(test_dir.path, "ne_test"))
    out = pyeo.point_to_pixel_coordinates(raster, point)
    # Since ne_test is an 11 by 12 image, tl corner 0,0 w 10m pixels,
    # we'd expect the br coords to be 10, 11
    assert out == (10, 11)


def test_multiple_intersection():
    # http://dev.openlayers.org/examples/vector-formats.html to test the wkt
    test_polys = [
        ogr.CreateGeometryFromWkt(r"POLYGON((10 10, 12 12, 12 10, 10 10))"),
        ogr.CreateGeometryFromWkt(r"POLYGON((11 11, 13 13, 13 11, 11 11))")
    ]
    target = ogr.CreateGeometryFromWkt(
        r"POLYGON((11 11, 12 12, 12 11, 11 11))"
    )
    out = pyeo.multiple_intersection(test_polys)
    assert out.__str__() == target.__str__()


def test_clip_raster(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_raster = os.path.join(test_dir.path, "se_test")
    test_aoi = os.path.join(test_dir.path, "aoi")
    result_path = os.path.join(test_dir.path, "test_out.tif")
    pyeo.clip_raster(test_raster, test_aoi, result_path)
    result = gdal.Open(result_path)
    assert result.ReadAsArray().shape == (5, 10, 10)
    assert result.GetGeoTransform() == (10, 10, 0, 10, 0, -10)


def test_write_polygon(managed_geotiff_dir):
    test_dir = managed_geotiff_dir
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(1,1)
    ring.AddPoint(1,5)
    ring.AddPoint(5,1)
    ring.AddPoint(1,1)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    result_path = os.path.join(test_dir.path, "test_poly")
    pyeo.write_polygon(poly, result_path)
    assert os.path.exists(result_path)


def test_check_overlap(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_raster = gdal.Open(os.path.join(test_dir.path, "ne_test"))
    test_aoi = ogr.Open(os.path.join(test_dir.path, "aoi"))
    assert pyeo.check_overlap(test_raster, test_aoi) is True


def test_get_raster_bounds(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_raster = gdal.Open(os.path.join(test_dir.path, "ne_test"))
    target = (0, 110, 0, 120)
    result = pyeo.get_raster_bounds(test_raster)
    assert result.GetEnvelope() == target


def test_get_aoi_bounds(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_aoi = ogr.Open(os.path.join(test_dir.path, "aoi"))
    target = (10, 20, 10, 20)
    result = pyeo.get_aoi_bounds(test_aoi)
    assert result.GetEnvelope() == target


def test_get_aoi_intersection(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_raster = gdal.Open(os.path.join(test_dir.path, "se_test"))
    test_aoi = ogr.Open(os.path.join(test_dir.path, "aoi"))
    result = pyeo.get_aoi_intersection(test_raster, test_aoi)
    assert result.GetEnvelope() == (10, 20, 10, 20)


def test_aoi_to_mask(managed_noncontiguous_geotiff_dir):
    test_dir = managed_noncontiguous_geotiff_dir
    test_aoi = ogr.Open(os.path.join(test_dir.path, "aoi"))
    result_path = os.path.join(test_dir.path, "test_out.tif")
    pyeo.aoi_to_mask(test_aoi, result_path)
    result = gdal.Open(result_path)
    assert result.ReadAsArray()[2, 2] == 1


def test_classify_change(managed_ml_geotiff_dir):
    test_dir = managed_ml_geotiff_dir
    test_path = os.path.join(test_dir.path, "training_data")
    model = None
    pyeo.classify_image(test_path, model, test_dir.path, test_dir.path)


def test_reshape_raster_for_ml(managed_ml_geotiff_dir):
    test_dir = managed_ml_geotiff_dir
    test_image_path = os.path.join(test_dir.path, "training_image")
    test_image = gdal.Open(test_image_path)
    array = test_image.GetVirtualMemArray()
    array = pyeo.reshape_raster_for_ml(array)
    assert array.shape == (120, 8)
    assert np.all(array[2, :] == [2]*8)
    assert np.all(array[119, :] == [119]*8)


def test_reshape_ml_out_to_raster():
    ml_array = np.arange(stop=60)
    out = pyeo.reshape_ml_out_to_raster(ml_array, 10, 6)
    assert out.shape == (6, 10)
    assert out[6, 0] == 7


def test_get_training_data(managed_ml_geotiff_dir):
    test_dir = managed_ml_geotiff_dir
    training_image = os.path.join(test_dir.path, "training_image")
    training_shapes = os.path.join(test_dir.path, "training_shape/geometry.shp")
    out = pyeo.get_training_data(training_image, training_shapes)
    assert out[0].shape == (2, 100)


def test_sort_by_timestamp():
    with TemporaryDirectory() as td:
        image_4 = os.path.join(td, r"S2A_MSIL1C_20180514T073611_N0206_R092_T37NCA_20180514T095515.SAFE")
        image_3 = os.path.join(td, r"S2A_MSIL1C_20180524T073731_N0206_R092_T37NBA_20180524T113104.SAFE")
        image_2 = os.path.join(td, r"S2B_MSIL1C_20180608T073609_N0206_R092_T37NBA_20180608T095659.SAFE")
        image_1 = os.path.join(td, r"S2A_MSIL1C_20180703T073611_N0206_R092_T37NBA_20180703T094637.SAFE")
        input = [image_2, image_3, image_4, image_1]
        target = [image_1, image_2, image_3, image_4]
        out_paths = pyeo.sort_by_s2_timestamp(input)
        assert out_paths == target


#def test_combine_masks_or():
#    with Tempor