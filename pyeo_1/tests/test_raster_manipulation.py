import glob
import os
import shutil

import numpy as np

from osgeo import gdal, ogr
import osr
import pytest

import pyeo.filesystem_utilities
import pyeo.raster_manipulation

import pyeo.windows_compatability


@pytest.mark.skip
def test_composite_images_with_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                 r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif"]
    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask(test_data, out_file, generate_date_image=True)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


@pytest.mark.skip
def test_composite_across_projections():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(r"test_outputs/reprojected")
    except FileNotFoundError:
        pass
    os.mkdir(r"test_outputs/reprojected")
    epsg = 4326
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    projection = proj.ExportToWkt() # Refactor this terrible nonsense later
    test_data = [r"test_data/S2A_MSIL2A_20180703T073611_N0206_R092_T36MZE_20180703T094637.tif",
                 r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"]
    pyeo.raster_manipulation.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.raster_manipulation.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.filesystem_utilities.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.filesystem_utilities.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)

    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask([
       r"test_outputs/reprojected/0.tif",
        r"test_outputs/reprojected/1.tif"],
        out_file)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


@pytest.mark.skip
def test_composite_across_projections_meters():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(r"test_outputs/reprojected")
    except FileNotFoundError:
        pass
    os.mkdir(r"test_outputs/reprojected")
    epsg = 32736
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg)
    projection = proj.ExportToWkt() # Refactor this terrible nonsense later

    test_data = [r"test_data/S2A_MSIL2A_20180703T073611_N0206_R092_T36MZE_20180703T094637.tif",
                 r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"]
    pyeo.raster_manipulation.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.raster_manipulation.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.filesystem_utilities.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.raster_manipulation.reproject_image(pyeo.filesystem_utilities.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)

    out_file = r"test_outputs/composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask([
       r"test_outputs/reprojected/0.tif",
        r"test_outputs/reprojected/1.tif"],
        out_file)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 1


def test_reprojection():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/reprojection_test.tif")
    except FileNotFoundError:
        pass
    new_projection = r"""PROJCS["WGS 84 / UTM zone 36S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",33],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32736"]]"""
    image = r"test_data/composite_T36MZE_20190509T073621_20190519T073621_clipped.tif"
    out_file = r"test_outputs/reprojection_test.tif"
    pyeo.raster_manipulation.reproject_image(image, out_file, new_projection, do_post_resample=False)
    result = gdal.Open(out_file)
    assert result
   # assert result.GetProjection() == new_projection
    result_array = result.GetVirtualMemArray()
    assert result_array.max() > 10

@pytest.mark.skip
def test_buffered_composite():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/buffered_composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/buffered_masks/20180103T172709.tif",
                 r"test_data/buffered_masks/20180319T172021.tif",
                 r"test_data/buffered_masks/20180329T171921.tif"]
    out_file = r"test_outputs/buffered_composite_test.tif"
    pyeo.raster_manipulation.composite_images_with_mask(test_data, out_file)


@pytest.mark.skip
def test_composite_off_by_one():
    """The images in test_outputs/off_by_one were producing off-by-one errors"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/off_by_one_error")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/off_by_one_error")
    pyeo.raster_manipulation.composite_directory("test_data/off_by_one", "test_outputs/off_by_one_error", generate_date_images=True)


def test_raster_sum():
    test_dir = "test_data"
    out_path = "test_outputs/raster_sum.tif"

    test_image_list = [
        "test_data/all.tif",
        "test_data/original_all.tif"
    ]
    pyeo.raster_manipulation.raster_sum(inRstList=test_image_list, outFn=out_path)


def test_mask_buffering():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_masks = [r"test_data/buffered_masks/20180103T172709.msk",
                  r"test_data/buffered_masks/20180319T172021.msk",
                  r"test_data/buffered_masks/20180329T171921.msk"]
    try:
        [os.remove(mask) for mask in test_masks]
    except FileNotFoundError:
        pass
    [shutil.copy(mask, "test_data/buffered_masks/") for mask in
     ["test_data/20180103T172709.msk", "test_data/20180319T172021.msk", r"test_data/20180329T171921.msk"]]
    [pyeo.raster_manipulation.buffer_mask_in_place(mask, 2) for mask in test_masks]
    assert [os.path.exists(mask) for mask in test_masks]


@pytest.mark.slow
def test_ml_masking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/ml_mask_test.tif")
        os.remove("test_outputs/ml_mask_test.msk")
    except FileNotFoundError:
        pass
    shutil.copy(r"test_data/20180103T172709.tif", r"test_outputs/ml_mask_test.tif")
    pyeo.raster_manipulation.create_mask_from_model(
        r"test_outputs/ml_mask_test.tif",
        r"test_data/cloud_model_v0.1.pkl",
        buffer_size=10
    )


def test_mask_combination():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    masks = ["test_data/S2B_MSIL2A_20200322T074609_N0214_R135_T36MYE_20200322T121000.msk",
             "test_data/S2B_MSIL2A_20200322T074609_N0214_R135_T36MZE_20200322T121000.msk",
             "test_data/S2B_MSIL2A_20200322T074609_N0214_R135_T36NZF_20200322T121000.msk"]
    try:
        os.remove("test_outputs/union_or_combination.tif")
        os.remove("test_outputs/intersection_and_combination.tif")
        os.remove("test_outputs/intersection_or_combination.tif")
        os.remove("test_outputs/union_and_combination.tif")

    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.combine_masks(masks, "test_outputs/union_or_combination.tif",
                                           geometry_func="union", combination_func="or")
    pyeo.raster_manipulation.combine_masks(masks, "test_outputs/intersection_and_combination.tif",
                                           geometry_func="intersect", combination_func="and")
    pyeo.raster_manipulation.combine_masks(masks, "test_outputs/intersection_or_combination.tif",
                                           geometry_func="intersect", combination_func="or")
    pyeo.raster_manipulation.combine_masks(masks, "test_outputs/union_and_combination.tif",
                                           geometry_func="union", combination_func="and")
    mask_1 = gdal.Open("test_outputs/union_or_combination.tif")
    assert not mask_1.GetVirtualMemArray().all == False
    mask_2 = gdal.Open("test_outputs/intersection_and_combination.tif")
    assert not mask_2.GetVirtualMemArray().all == False
    mask_3 = gdal.Open("test_outputs/intersection_or_combination.tif")
    assert not mask_3.GetVirtualMemArray().all == False
    mask_4 = gdal.Open("test_outputs/union_and_combination.tif")
    assert not mask_4.GetVirtualMemArray().all == False


def test_mask_from_confidence_layer():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/confidence_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.create_mask_from_confidence_layer(
        "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE",
        "test_outputs/masks/confidence_mask.tif",
        cloud_conf_threshold=0,
        buffer_size=3)


def test_fmask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.apply_fmask(
        "test_data/S2A_MSIL1C_20170922T025541_N0205_R032_T48MXU_20191021T161210.SAFE",
        "test_outputs/masks/fmask.tif"
    )
    assert gdal.Open("test_outputs/masks/fmask.tif")


def test_fmask_cloud_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/fmask.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.create_mask_from_fmask(
        "test_data/S2A_MSIL1C_20170922T025541_N0205_R032_T48MXU_20191021T161210.SAFE",
        "test_outputs/masks/fmask_cloud_and_shadow.tif"
    )


def test_combination_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/combined_mask.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.create_mask_from_sen2cor_and_fmask(
        "test_data/S2A_MSIL1C_20170922T025541_N0205_R032_T48MXU_20191021T161210.SAFE",
        "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE",
        "test_outputs/masks/combined_mask.tif")


def test_mask_joining():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/masks/joining_test.tif")
    except FileNotFoundError:
        pass
    pyeo.raster_manipulation.combine_masks(["test_data/masks/fmask_cloud_and_shadow.tif", "test_data/masks/confidence_mask.tif"],
                       "test_outputs/masks/joining_test.tif", combination_func="and", geometry_func="union")
    out_image = gdal.Open("test_outputs/masks/joining_test.tif")
    assert out_image
    out_array = out_image.GetVirtualMemArray()
    assert 1 in out_array
    assert 0 in out_array


def test_s2_band_stacking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/band_stacking")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/band_stacking")
    test_safe_file = "test_data/S2A_MSIL2A_20170922T025541_N0205_R032_T48MXU_20170922T031450.SAFE"
    # Stacking only the default 10m bands
    default_out = "test_outputs/band_stacking/default.tif"
    pyeo.raster_manipulation.stack_sentinel_2_bands(test_safe_file, default_out)
    some_bands_10m = "test_outputs/band_stacking/with_reproj.tif"
    pyeo.raster_manipulation.stack_sentinel_2_bands(test_safe_file, some_bands_10m, out_resolution=20)
    weird_bands = "test_outputs/band_stacking/weird_bands.tif"
    pyeo.raster_manipulation.stack_sentinel_2_bands(test_safe_file, weird_bands,
                                                    bands=["B02", "B08", "SCL", "B8A", "B11", "B12"],
                                                    out_resolution=60)


def test_band_maths():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/ndvi.tif")
    except FileNotFoundError:
        pass

    test_file = "test_data/bands.tif"
    test_outputs = "test_outputs/ndvi.tif"
    pyeo.raster_manipulation.apply_band_function(test_file,
                                                 pyeo.raster_manipulation.ndvi_function,
                                                 [2, 3],
                                                 test_outputs,
                                                 out_datatype=gdal.GDT_Float32)


def test_averaging():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/averaged.tif")
    except FileNotFoundError:
        pass
    test_files = ["test_data/S2A_MSIL2A_20180626T110621_N0208_R137_T31UCU_20180626T120032.SAFE/GRANULE/L2A_T31UCU_A015721_20180626T111413/IMG_DATA/R10m/T31UCU_20180626T110621_B02_10m.jp2",
                  "test_data/S2A_MSIL2A_20180626T110621_N0208_R137_T31UCU_20180626T120032.SAFE/GRANULE/L2A_T31UCU_A015721_20180626T111413/IMG_DATA/R10m/T31UCU_20180626T110621_B03_10m.jp2",
                  "test_data/S2A_MSIL2A_20180626T110621_N0208_R137_T31UCU_20180626T120032.SAFE/GRANULE/L2A_T31UCU_A015721_20180626T111413/IMG_DATA/R10m/T31UCU_20180626T110621_B04_10m.jp2"
                ]
    test_output = "test_outputs/averaged.tif"
    pyeo.raster_manipulation.average_images(test_files, test_output)


def test_clip_geojson():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_path = "test_outputs/clip_from_json.tif"
    try:
        os.remove(out_path)
    except FileNotFoundError:
        pass
    test_image = "test_data/class_composite_T36MZE_20190509T073621_20190519T073621.tif"  # epsg 4326; watch out
    test_json = "test_data/mt_kippiri.geojson"
    pyeo.raster_manipulation.clip_raster(test_image, test_json, out_path)
    result = gdal.Open(out_path)
    assert result
    assert result.GetVirtualMemArray().max() > 0


def test_clip_geojson_projection():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_path = "test_outputs/clip_from_json_reprojection.tif"
    try:

        os.remove(out_path)
    except FileNotFoundError:
        pass
    test_image = "test_data/original_all.tif"  # epsg something else; watch out
    test_json = "test_data/merak.geojson"
    pyeo.raster_manipulation.clip_raster(test_image, test_json, out_path)
    result = gdal.Open(out_path)
    assert result
    assert result.GetVirtualMemArray().max() > 0


def test_remove_raster_band():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_file = "test_data/composite_T36MZE_20190509T073621_20190519T073621_clipped.tif"
    try:
        os.remove("test_outputs/band_removal.tif")
    except FileNotFoundError:
        pass
    out_path = "test_outputs/band_removal.tif"
    pyeo.raster_manipulation.strip_bands(test_file, out_path, [2])
    out = gdal.Open(out_path)
    assert out
    assert out.RasterCount == 7


def test_create_new_image_from_polygon():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataset = ogr.Open("test_data/wuhan_aoi_epsg32650.shp")
    layer = dataset.GetLayerByIndex(0)
    feature = layer.GetNextFeature()
    polygon = feature.GetGeometryRef()
    out_path = "test_data/polygon_test.tif"
    x_res=10
    y_res=10
    bands=2
    projection = polygon.GetSpatialReference().ExportToWkt()
    pyeo.raster_manipulation.create_new_image_from_polygon(polygon, out_path, x_res, y_res, bands,
                                  projection, format="GTiff", datatype=gdal.GDT_Int32, nodata=-4)
    out = gdal.Open(out_path)
    assert np.all(out.ReadAsArray() == -4)