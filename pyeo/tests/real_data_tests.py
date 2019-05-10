"""A set of integrations tests using the data in real_data. Each major test simulates a step in the processing chain.
A test is considered passed if it produces a valid gdal object that contains a sensible range of values (ie not all 0s
or 1s). It does NOT check to see if the image is valid or not; visual inspection using QGIS is strongly recommended!
This is designed to run slow and keep the test outputs after running; around 20 minutes normal, and around ~2.5 hours
if --runslow is used.

To set up:
    - Download test_data.zip from https://s3.eu-central-1.amazonaws.com/pyeodata/test_data.zip (~15gb)
    - Unzip in pyeo/pyeo/tests (so that gives you pyeo/pyeo/tests/test_data)
    - If you want to run the download and preprocessing tests, edit test_config.ini with your ESA Hub credentials

Notes:
    - Anything in test_data should not be touched by code and will remain as constant input for inputs. It will be
    updated if the API significantly changes.
    - Every file in test_outputs get deleted at the start of the relevant test and re-created;
    this means that test outputs persist between test runs for inspection and tweaking

Recommended running augments:
cd .../pyeo/tests
pytest real_data_tests.py --log-cli-level DEBUG   (runs all non-slow tests with log output printed to stdout)
pytest real_data_tests.py --log-cli-level DEBUG -k composite   (runs all non-slow tests with 'composite' in the function
name)
pytest real_data_tests.py --log-cli-level DEBUG --runslow  (runs all tests)
"""
import os, sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))
import pyeo.core as pyeo
from pyeo.core import gdal
import osr
import numpy as np
import pytest
import configparser
import glob

gdal.UseExceptions()


def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    pyeo.init_log("test_log.log")


def load_test_conf():
    test_conf = configparser.ConfigParser()
    test_conf.read("test_data/test_creds.ini")
    return test_conf


@pytest.mark.webtest
def test_query_and_download():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_conf = load_test_conf()
    user = test_conf["sent_2"]["user"]
    passwd = test_conf["sent_2"]["pass"]
    images = pyeo.sent2_query(test_conf["sent_2"]["user"], test_conf["sent_2"]["pass"],
                     "test_data/marque_de_com_really_simple.geojson",
                     "20180101", "20180110")
    assert len(images) > 0
    try:
        shutil.rmtree("test_outputs/L1")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/L1")
    pyeo.download_s2_data(images, "test_outputs/L1", source='scihub', user=user, passwd=passwd)
    for image_id in images:
        assert os.path.exists("test_outputs/L1/{}".format(images[image_id]['title']+".SAFE"))


@pytest.mark.webtest
def test_google_cloud_dl():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/google_data")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/google_data")
    product_ids = ["S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"]
    pyeo.download_from_google_cloud(product_ids, "test_outputs/google_data")
    for id in product_ids:
        assert os.path.exists("test_outputs/google_data/{}".format(id))


@pytest.mark.webtest
@pytest.mark.xfail
def test_old_format_google_cloud_dl():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/google_data")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/google_data")
    product_ids = ["S2B_MSIL1C_20170715T151709_N0205_R125_T18NXH_20170715T151704.SAFE"]
    pyeo.download_from_google_cloud(product_ids, "test_outputs/google_data")
    for id in product_ids:
        assert os.path.exists("test_outputs/google_data/{}".format(id))


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
    [pyeo.buffer_mask_in_place(mask, 2) for mask in test_masks]
    assert [os.path.exists(mask) for mask in test_masks]


def test_composite_images_with_mask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove(r"test_outputs/composite_test.tif")
    except FileNotFoundError:
        pass
    test_data = [r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                 r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif"]
    out_file = r"test_outputs/composite_test.tif"
    pyeo.composite_images_with_mask(test_data, out_file, generate_date_image=True)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


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
    pyeo.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.reproject_image(pyeo.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.reproject_image(pyeo.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)
    
    out_file = r"test_outputs/composite_test.tif"
    pyeo.composite_images_with_mask([
       r"test_outputs/reprojected/0.tif",
        r"test_outputs/reprojected/1.tif"],
        out_file)
    image = gdal.Open("test_outputs/composite_test.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert image_array.max() > 10


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
    pyeo.reproject_image(test_data[0], r"test_outputs/reprojected/0.tif", projection)
    pyeo.reproject_image(test_data[1], r"test_outputs/reprojected/1.tif", projection)
    pyeo.reproject_image(pyeo.get_mask_path(test_data[0]), r"test_outputs/reprojected/0.msk", projection)
    pyeo.reproject_image(pyeo.get_mask_path(test_data[1]), r"test_outputs/reprojected/1.msk", projection)
    
    out_file = r"test_outputs/composite_test.tif"
    pyeo.composite_images_with_mask([
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
    image = r"test_data/S2B_MSIL2A_20180728T073609_N0206_R092_T37MBV_20180728T114325.tif"
    out_file = r"test_outputs/reprojection_test.tif"
    pyeo.reproject_image(image, out_file, new_projection)
    result = gdal.Open(out_file)
    assert result
   # assert result.GetProjection() == new_projection
    result_array = result.GetVirtualMemArray()
    assert result_array.max() > 10


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
    pyeo.composite_images_with_mask(test_data, out_file)


@pytest.mark.slow
def test_ml_masking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/ml_mask_test.tif")
        os.remove("test_outputs/ml_mask_test.msk")
    except FileNotFoundError:
        pass
    shutil.copy(r"test_data/20180103T172709.tif", r"test_outputs/ml_mask_test.tif")
    pyeo.create_mask_from_model(
        r"test_outputs/ml_mask_test.tif",
        r"test_data/cloud_model_v0.1.pkl",
        buffer_size=10
    )


@pytest.mark.slow
def test_preprocessing():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/L2")
    except FileNotFoundError:
        pass
    conf = load_test_conf()
    pyeo.atmospheric_correction("test_data/L1", "test_outputs/L2",
                                conf['sen2cor']['path'])
    assert os.path.isfile(
        "test_outputs/L2/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.SAFE/GRANULE/L2A_T13QFB_A004328_20180103T172711/IMG_DATA/R10m/T13QFB_20180103T172709_B08_10m.jp2"
    )
    assert os.path.isfile(
        "test_outputs/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE/GRANULE/L2A_T13QFB_A014452_20180329T173239/IMG_DATA/R10m/T13QFB_20180329T173239_B08_10m.jp2"
    )


def test_merging():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif")
        os.remove("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.msk")
        os.remove("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.msk")
    except FileNotFoundError:
        pass
    pyeo.preprocess_sen2_images("test_data/L2/", "test_outputs/", "test_data/L1/", buffer_size=5)
    assert os.path.exists("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif")
    assert os.path.exists("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif")
    assert os.path.exists("test_outputs/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.msk")
    assert os.path.exists("test_outputs/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.msk")


def test_stacking():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.msk")
        os.remove("test_outputs/T13QFB_20180329T171921_20180103T172709.tif")
    except FileNotFoundError:
        pass
    pyeo.stack_old_and_new_images(r"test_data/S2B_MSIL2A_20180103T172709_N0206_R012_T13QFB_20180103T192359.tif",
                                  r"test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
                                  r"test_outputs")
    image = gdal.Open("test_outputs/T13QFB_20180319T172021_20180103T172709.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    for layer in range(image_array.shape(0)):
        assert image_array[layer, :, :].max > 10


@pytest.mark.slow
def test_classification():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.remove("test_outputs/class_20180103T172709_20180319T172021.tif")
    except FileNotFoundError:
        pass
    pyeo.classify_image("test_data/T13QFB_20180103T172709_20180329T171921.tif", "test_data/manantlan_v1.pkl",
                        "test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif", num_chunks=4)
    image = gdal.Open("test_outputs/class_T13QFB_20180103T172709_20180319T172021.tif")
    assert image
    image_array = image.GetVirtualMemArray()
    assert not np.all(image_array == 0)


def test_mask_combination():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    masks = [r"test_data/"+mask for mask in os.listdir("test_data") if mask.endswith(".msk")]
    try:
        os.remove("test_outputs/union_or_combination.tif")
        os.remove("test_outputs/intersection_and_combination.tif")
    except FileNotFoundError:
        pass
    pyeo.combine_masks(masks, "test_outputs/union_or_combination.tif",
                       geometry_func="union", combination_func="or")
    pyeo.combine_masks(masks, "test_outputs/intersection_and_combination.tif",
                       geometry_func="intersect", combination_func="and")
    mask_1 = gdal.Open("test_outputs/union_or_combination.tif")
    assert not mask_1.GetVirtualMemArray().all == False
    mask_2 = gdal.Open("test_outputs/intersection_and_combination.tif")
    assert not mask_1.GetVirtualMemArray().all == False


def test_composite_off_by_one():
    """The images in test_outputs/off_by_one were producing off-by-one errors"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/off_by_one_error")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/off_by_one_error")
    pyeo.composite_directory("test_data/off_by_one", "test_outputs/off_by_one_error",generate_date_images=True)


def test_mask_closure():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    out_mask_path = "test_outputs/test_mask.msk"
    if os.path.exists(out_mask_path):
        os.remove(out_mask_path)
    shutil.copy("test_data/20180103T172709.msk", out_mask_path)
    pyeo.buffer_mask_in_place(out_mask_path, 3)


def test_get_l1():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    output = pyeo.get_l1_safe_file(
        "test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
        "test_data/L1"
    )
    assert output == "test_data/L1/S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_get_l2():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    output = pyeo.get_l2_safe_file(
        "test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif",
        "test_data/L2"
    )
    assert output == "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_get_tile():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    assert pyeo.get_sen_2_image_tile("test_data/T13QFB_20180103T172709_20180329T171921.tif") == "T13QFB"
    assert pyeo.get_sen_2_image_tile("test_data/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.tif") == "T13QFB"


def test_get_preceding_image():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_dir = "test_data/L2"
    test_image_name = "S2B_MSIL2A_20180713T172709_N0206_R012_T13QFB_2018073T192359.SAFE"
    assert pyeo.get_preceding_image_path(test_image_name, test_dir) == "test_data/L2/S2A_MSIL2A_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"


def test_raster_reclass_binary():
    test_image_name = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/cherangany/class_composite_T36NYF_20180112T075259_20180117T075241.tif'
    test_value = 1
    out_fn = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/cherangany/class_composite_T36NYF_20180112T075259_20180117T075241_rcl.tif'
    a = pyeo.raster_reclass_binary(test_image_name, test_value, outFn=out_fn)
    print(np.unique(a) == [0, 1])


def test_raster_reclass_directory():
    test_dir = '/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/mt_elgon'
    rst_list = glob.glob(os.path.join(test_dir, '*.tif'))
    test_value = 1
    suffix = '_rcl.'
    for i, in_rst in enumerate(rst_list):
        path, fn = os.path.split(in_rst)
        n, fmt = fn.split('.', 2)
        fn = n+suffix+fmt
        out_fn = os.path.join(path, fn)
        pyeo.raster_reclass_binary(in_rst, test_value, outFn=out_fn)


def test_raster_sum():
    test_dir = "/media/ubuntu/data_archive/F2020/Kenya/outputs/classifications/mt_elgon"
    test_pattern = '*_T37MDR_*_rcl.tif'
    tile_id = 'T37MDR'
    fn = 'KEN_cherangany_' + tile_id + '_forestChange_sum2018.tif'
    out_fn = os.path.join(test_dir, fn)
    test_image_list = glob.glob(os.path.join(test_dir, test_pattern))
    pyeo.raster_sum(inRstList=test_image_list, outFn=out_fn)


if __name__ == "__main__":
    print(sys.path)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log = pyeo.init_log("test_log.log")
    test_raster_reclass_directory()
    # test_raster_sum()
