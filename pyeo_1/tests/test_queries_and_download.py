import os
import shutil

import pytest
from sklearn.externals import joblib

import pyeo.queries_and_downloads
from pyeo.tests.utilities import load_test_conf


@pytest.mark.webtest
def test_query_and_download():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_conf = load_test_conf()
    user = test_conf["sent_2"]["user"]
    passwd = test_conf["sent_2"]["pass"]
    images = pyeo.queries_and_downloads.sent2_query(test_conf["sent_2"]["user"], test_conf["sent_2"]["pass"],
                     "test_data/mt_kippiri.geojson",
                     "20210501T00.00.00", "20210504", cloud=100,
                     query_func=pyeo.queries_and_downloads._rest_query)
    assert len(images) > 0
    try:
        shutil.rmtree("test_outputs/L1")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("test_outputs/L2")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/L1")
    os.mkdir("test_outputs/L2")
    pyeo.queries_and_downloads.download_s2_data(images, "test_outputs/L1", "test_outputs/L2", source='aws', user=user, passwd=passwd)
    for image_id in images:
        assert os.path.exists("test_outputs/L1/{}".format(images[image_id]['title']+".SAFE"))


@pytest.mark.webtest
def test_query_non_4326():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_conf = load_test_conf()
    query_out = pyeo.queries_and_downloads.sent2_query(test_conf["sent_2"]["user"], test_conf["sent_2"]["pass"],
                     "test_data/wuhan_aoi_epsg32650.shp",
                     "20180101", "20180130")
    assert len(query_out) > 0


def test_rest_query():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_conf = load_test_conf()
    query_out = pyeo.queries_and_downloads.sent2_query(test_conf["sent_2"]["user"], test_conf["sent_2"]["pass"],
                     "test_data/wuhan_large.shp",
                     "2021-01-01T00:00:00.000Z", "2021-03-30T00:00:00.000Z",
                     cloud=100, query_func=pyeo.queries_and_downloads._rest_query)
    assert len(query_out) > 0


def test_landsat_query_and_download():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_conf = load_test_conf()

    images = pyeo.queries_and_downloads.landsat_query(test_conf, "test_data/merak.geojson", "20180101", "20180110")
    assert len(images) > 0
    out_dir = "test_data/landsat_from_usgs"
    try:
        shutil.rmtree(out_dir)
    except FileNotFoundError:
        pass
    os.mkdir(out_dir)
    products = pyeo.queries_and_downloads.download_landsat_data(images, out_dir, test_conf)


@pytest.mark.webtest
def test_google_cloud_dl():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        shutil.rmtree("test_outputs/google_data")
    except FileNotFoundError:
        pass
    os.mkdir("test_outputs/google_data")
    product_ids = ["S2A_MSIL1C_20180329T171921_N0206_R012_T13QFB_20180329T221746.SAFE"]
    pyeo.queries_and_downloads.download_from_google_cloud(product_ids, "test_outputs/google_data")
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
    pyeo.queries_and_downloads.download_from_google_cloud(product_ids, "test_outputs/google_data")
    for id in product_ids:
        assert os.path.exists("test_outputs/google_data/{}".format(id))


@pytest.mark.webtest
def test_pair_filter_with_dl():
    # Pickle this at some point so it down't depend on having a download
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_aoi = r"test_data/merak.geojson"
    start = "20180101"
    end = "20181210"
    conf = load_test_conf()
    test_results = pyeo.queries_and_downloads.check_for_s2_data_by_date(test_aoi, start, end, conf)
    filtered_test_results = pyeo.queries_and_downloads.filter_non_matching_s2_data(test_results)
    assert len(filtered_test_results) != 0


def test_list_filter():
    input = joblib.load("test_data/test_query.pkl")
    out = pyeo.queries_and_downloads.filter_non_matching_s2_data(input)
    assert len(out) == 10


def mock_download(to_download, l1_dir, l2_dir, user, passwd):
    print("Not downloading {}".format(to_download))
    assert len(to_download) == 2


def test_download_s2_pairs(monkeypatch):
    monkeypatch.setattr(pyeo.queries_and_downloads,"download_s2_data", mock_download)
    l1_dir = "test_data/test_pairs/L1"
    l2_dir = "test_data/test_pairs/L2"
    test_conf = load_test_conf()
    pyeo.queries_and_downloads.download_s2_pairs(l1_dir, l2_dir, test_conf)
