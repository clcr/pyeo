"""
pyeo.queries_and_downloads
==========================
Functions for querying, filtering and downloading data.

Key functions
-------------
:py:func:`check_for_s2_data_by_date` Queries the Sentinel 2 archive for products between two dates
:py:func:`download_s2_data` Downloads Sentinel 2 data from Scihub by default


SAFE files
----------
Sentinel 2 data is downloaded in the form of a .SAFE file; all download functions will end with data in this structure.
This is a directory structure that contains the imagery, metadata and supplementary data of a Sentinel 2 image. The
rasters themeselves are the in the GRANULE/[granule_id]/IMG_DATA/[resolution]/ folder; each band is contained in
its own .jp2 file. For full details, see https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats

There are two ways to refer to a given Sentinel-2 products: the UUID and the product ID.
The UUID is a set of five five-character strings (EXMAPLE HERE)
The product ID is a human-readable string (more or less) containing all the information needed for unique identification
of an product, split by the underscore character. For more information on the structure of a product ID,
see (EXAMPLE HERE)

Query data structure
--------------------
All query functions return a dictionary. The key of the dictionary is the UUID of the product id; the product is
a further set of nested dictionaries containing information about the product to be downloaded. (PUT STRUCTURE HERE)

Data download sources
---------------------
This library presently offers three options for download sources; Scihub and Amazon Web Services, for Sentinel, and
USGS, for Landsat. If in doubt, use Scihub.

- Scihub

   The Copernicus Open-Access Hub is the default option for downloading sentinel-2 images. Images are downloaded in .zip
   format, and then automatically unzipped. Users are required to register with a username and password before downloading,
   and there is a limit to no more than two concurrent downloads per username at a time. Scihub is entirely free.

- AWS

   Sentinel data is also publicly hosted on Amazon Web Services. This storage is provided by Sinergise, and is normally
   updated a few hours after new products are made available. There is a small charge associated with downloading this
   data. To access the AWS repository, you are required to register an Amazon Web Services account (including providing
   payment details) and obtain an API key for that account. See https://aws.amazon.com/s3/pricing/ for pricing details;
   the relevant table is Data Transfer Pricing for the EU (Frankfurt) region. There is no limit to the concurrent
   downloads for the AWS bucket.

- USGS

   Landsat data is hosted and provided by the US Geological Survey. You can sign up at https://ers.cr.usgs.gov/register/


Functions
---------
"""

import datetime as dt
import glob
import itertools
import json
import logging
import os
import re
import shutil
import tarfile
import zipfile
from multiprocessing.dummy import Pool
from urllib.parse import urlencode
import numpy as np
from bs4 import BeautifulSoup  # I didn't really want to use BS, but I can't see a choice.
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import ogr, osr
import requests
import tenacity
from botocore.exceptions import ClientError
from requests import Request

from sentinelhub import download_safe_format
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

from pyeo.filesystem_utilities import check_for_invalid_l2_data, check_for_invalid_l1_data, get_sen_2_image_tile
import pyeo.filesystem_utilities as fu
from pyeo.coordinate_manipulation import reproject_vector, get_vector_projection
from pyeo.exceptions import NoL2DataAvailableException, BadDataSourceExpection, TooManyRequests, \
    InvalidGeometryFormatException, InvalidDateFormatException

log = logging.getLogger("pyeo")

import pyeo.windows_compatability

try:
    from google.cloud import storage
except ImportError:
    pass


def _rest_query(user, passwd, footprint_wkt, start_date, end_date, cloud=50):
    session = requests.Session()
    session.auth = (user, passwd)
    rest_url = "https://apihub.copernicus.eu/apihub/search"

    search_params = {
        "platformname": "((Sentinel-2))",
        "footprint": '(\"Intersects({})\")'.format(footprint_wkt),
        "beginposition": "[{} TO {}]".format(start_date, end_date),
        "endposition": "[{} TO {}]".format(start_date, end_date),
        "cloudcoverpercentage": "[0 TO {}]".format(cloud)
    }
    search_string = " AND ".join([f"{term}:{query}" for term, query in search_params.items()])

    request_params = {
        "q": search_string,
    }

    results = session.get(rest_url, params=request_params)
    if results.status_code >= 400:
        print("Bad request: code {}".format(results.status_code))
        print(results.content)
        raise requests.exceptions.RequestException
    return _rest_out_to_json(results)


def _rest_out_to_json(result):
    root = ElementTree.fromstring(result.content.replace(b"\n", b""))
    total_results = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
    if total_results > 10:
        log.warning("Local querying does not yet return more than 10 entries in search.")
    if total_results == 0:
        log.warning("Query produced no results.")
    out = {}
    for element in root.findall("{http://www.w3.org/2005/Atom}entry"):
        id = element.find("{http://www.w3.org/2005/Atom}id").text
        out[id] = _parse_element(element)
        out[id].pop(None)
        out[id]['title'] = out[id]['identifier']+".SAFE"
    return out


def _parse_element(element):
    if len(element) == 0:
        if element.get('name'):
            return(element.text)
        else:
            return None
    else:
        out = {}
        for subelement in element:
            out[subelement.get('name')] = _parse_element(subelement)
        return out



def _sentinelsat_query(user, passwd, footprint_wkt, start_date, end_date, cloud=50):
    """
    Fetches a list of Sentienl-2 products
    """
    # Originally by Ciaran Robb
    api = SentinelAPI(user, passwd)
    products = api.query(footprint_wkt,
                         date=(start_date, end_date), platformname="Sentinel-2",
                         cloudcoverpercentage="[0 TO {}]".format(cloud),
                         url="https://apihub.copernicus.eu/apihub/")
    return products


def _is_4326(geom):
    proj_geom = get_vector_projection(geom)
    proj_4326 = osr.SpatialReference()
    proj_4326.ImportFromEPSG(4326)
    if proj_geom == proj_4326:
        return True
    else:
        return False


def sent2_query(user, passwd, geojsonfile, start_date, end_date, cloud=50, query_func=_rest_query):
    """
    Fetches a list of Sentienl-2 products

    Parameters
    -----------

    user : string
           Username for ESA hub. Register at https://scihub.copernicus.eu/dhus/#/home

    passwd : string
             password for the ESA Open Access hub

    geojsonfile : string
                  Path to a geometry file containing a polygon of the outline of the area you wish to download.
                  Can be a geojson (.json/.geojson) or a shapefile (.shp)
                  See www.geojson.io for a tool to build these.

    start_date : string
                 Date of beginning of search in the format YYYY-MM-DDThh:mm:ssZ (ISO standard)

    end_date : string
               Date of end of search in the format yyyy-mm-ddThh:mm:ssZ
               See https://www.w3.org/TR/NOTE-datetime, or use check_for_s2_data_by_date

    cloud : int, optional
            The maximum cloud clover percentage (as calculated by Copernicus) to download. Defaults to 50%

    queryfunc : function
                A function that takes the following args: user, passwd, footprint_wkt, start_date, end_date, cloud

    Returns
    -------
    products : dict
        A dictionary of Sentinel-2 granule products that are touched by your AOI polygon, keyed by product ID.
        Returns both level 1 and level 2 data.

    Notes
    -----
    If you get a 'request too long' error, it is likely that your polygon is too complex. The following functions
    download by granule; there is no need to have a precise polygon at this stage.

    """
    with TemporaryDirectory() as td:
        # Preprocessing geometry
        geom = ogr.Open(geojsonfile)
        if not _is_4326(geom):
            reproj_geom_path = os.path.join(td, "temp.shp")
            reproject_vector(geojsonfile, os.path.join(td, "temp.shp"), 4326)
            geojsonfile = reproj_geom_path
        if geojsonfile.endswith("json"):
            footprint = geojson_to_wkt(read_geojson(geojsonfile))
        elif geojsonfile.endswith("shp"):
            footprint = shapefile_to_wkt(geojsonfile)
        else:
            raise InvalidGeometryFormatException("Please provide a .json, .geojson or a .shp as geometry.")

        # Preprocessing dates
        start_date = _date_to_timestamp(start_date)
        end_date = _date_to_timestamp(end_date)

        log.info("Sending Sentinel-2 query:\nfootprint: {}\nstart_date: {}\nend_date: {}\n cloud_cover: {} ".format(
            footprint, start_date, end_date, cloud))
        return query_func(user,passwd, footprint, start_date, end_date, cloud)


def _date_to_timestamp(date):
    #
    if type(date) == str:
        # Matches yyyy-mm-dd, yyyymmdd, yyyy-mm-ddThhmmssMMMZ (but ignores time)
        # Full regex explanation at: https://regex101.com/r/FjEoUD/1
        m = re.match(r"(\d{4})\W?(\d{2})\W?(\d{2})", date)
        if not m:
            raise InvalidDateFormatException
        year, month, day = m.groups()
        date = dt.date(int(year), int(month), int(day))
    if type(date) == dt.date:
        return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def shapefile_to_wkt(shapefile_path):
    """
    Converts a shapefile to a well-known text (wkt) format

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile to convert

    Returns
    -------
    wkt : str
        A wkt - string containing the geometry of the first feature of the first layer of the shapefile shapefile
    """
    dataset = ogr.Open(shapefile_path)
    layer = dataset.GetLayer(0)
    feature = layer.GetFeature(0)
    geometry = feature.GetGeometryRef()
    wkt = geometry.ExportToWkt()
    geometry, feature, layer, dataset = None, None, None, None
    return wkt



def landsat_query(conf, geojsonfile, start_date, end_date, cloud=50):
    """
    Queries the USGS dataset LANDSAT_8_C1 for imagery between the start_date and end_date, inclusive.
    This downloads all imagery touched by the bounding box of the provided geojson file.

    Parameters
    ----------
    conf : dict
        A dictionary with ['landsat']['user'] and ['landsat']['pass'] values, containing your USGS credentials.
    geojsonfile : str
        The geojson file
    start_date : str
        The start date, in "yyyymmdd" format. Will truncate any longer string.
    end_date : str
        The end query date, in "yyyymmdd" format. Will truncate any longer string.
    cloud : float
        The maximum cloud cover to return.

    Returns
    -------
    products : list of dict
        A list of products; each item being a dictionary returned from the USGS API.
        See https://earthexplorer.usgs.gov/inventory/documentation/datamodel#Scene

    """

    footprint = ogr.Open(geojsonfile)
    feature = footprint.GetLayer(0).GetFeature(0)
    geometry = feature.GetGeometryRef()
    lon_south, lon_north, lat_west, lat_east = geometry.GetEnvelope()
    geometry = None
    feature = None
    footprint = None
    start_date = "{}-{}-{}".format(start_date[0:4], start_date[4:6], start_date[6:8])
    end_date = "{}-{}-{}".format(end_date[0:4], end_date[4:6], end_date[6:8])

    session = requests.Session()
    api_root = "https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/"
    session_key = get_landsat_api_key(conf, session)

    if not session_key:
        log.error("Login to USGS failed.")
        return None

    data_request = {
        "apiKey": session_key,
        "datasetName": "LANDSAT_8_C1",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                "latitude": np.round(lat_west, 4),
                "longitude": np.round(lon_south, 4)
            },
            "upperRight": {
                "latitude": np.round(lat_east, 4),
                "longitude": np.round(lon_north, 4),
            },
        },
        "temporalFilter": {
            "startDate": start_date,
            "endDate": end_date
        },
        "maxCloudCover": cloud
    }
    log.info("Sending Landsat query:\n{}".format(data_request))
    request = Request("GET",
                      url=api_root + "search",
                      params={"jsonRequest": json.dumps(data_request)},
                      headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
                      )
    req_string = session.prepare_request(request)
    req_string.url = req_string.url.replace("+", "").replace("%27",
                                                             "%22")  # usgs why dont you like real url encoding -_-
    response = session.send(req_string)
    products = response.json()["data"]["results"]
    log.info("Retrieved {} product(s)".format(len(products)))
    log.info("Logging out of USGS")
    session.get(
        url=api_root + "logout",
        params={"jsonRequest": json.dumps({"apiKey": session_key})},
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
    )
    return products


def download_landsat_data(products, out_dir, conf):
    """
    Given an output from landsat_query, will download al L1C products to out_dir.

    Parameters
    ----------
    products : str
        Dictionary of landsat products; must include downloadUrl and displayId
    out_dir : str
        Directory to save Landsat files in. Folder structure is out_dir->displayId->products
    conf : dict
        Dictionary containing USGS login credentials. See docs for :py:func:`landsat_query`.
    """
    # The API key is no good here, we need the auth cookie. Time to pretend to be a browser.
    dl_session = requests.Session()
    page = dl_session.get("https://ers.cr.usgs.gov/login/").content
    # We also need the cross-site request forgery prevention token and the __ncforminfo (dunno?) value.
    # For this, we use BeautifulSoup; a library for finding things in webpages.
    login_soup = BeautifulSoup(page)
    inputs = login_soup.find_all("input")
    token = list(input.attrs['value'] for input in inputs
                 if 'name' in input.attrs
                 and input.attrs['name'] == 'csrf_token')[0]
    ncforminfo = list(input.attrs['value'] for input in inputs
                 if 'name' in input.attrs
                 and input.attrs['name'] == '__ncforminfo')[0]

    dl_session.post("https://ers.cr.usgs.gov/login/",
                    data={
                        "username": conf["landsat"]["user"],
                        "password": conf["landsat"]["pass"],
                        "csrf_token": token,
                        "__ncforminfo": ncforminfo
                    },
                    headers={
                        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
                    })

    # for each product in landsat, do stuff
    for product in products:
        download_landing_url = product["downloadUrl"]
        # BeautifulSoup is a library for finding things in webpages - in this case, every download link
        lp_soup = BeautifulSoup(requests.get(download_landing_url).content, 'html.parser')
        download_buttons = lp_soup.find_all("input")
        dirty_url = \
            list(button.attrs['onclick'] for button in download_buttons if "STANDARD" in button.attrs['onclick'])[0]
        clean_url = dirty_url.partition("=")[2].strip("\\'")
        log.info("Downloading landsat imagery from {}".format(clean_url))
        out_folder_path = os.path.join(out_dir, product['displayId'])
        os.mkdir(out_folder_path)
        tar_path = out_folder_path + ".tar.gz"
        with open(tar_path, 'wb+') as fp:
            image_response = dl_session.get(clean_url)
            fp.write(image_response.content)
            log.info("Item {} downloaded to {}".format(product['displayId'], tar_path))
        log.info("Unzipping {} to {}".format(tar_path, out_folder_path))
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(out_folder_path)
        log.info("Removing {}".format(tar_path))
        os.remove(tar_path)


def get_landsat_api_key(conf, session):
    """
    :meta private:
    Parameters
    ----------
    conf
    session

    Returns
    -------

    """
    user = conf['landsat']['user']
    passwd = conf['landsat']['pass']
    api_root = "https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/"
    log.info("Logging into USGS")
    login_post = {
        "username": user,
        "password": passwd,
        "catalogId": "EE"
    }
    session_key = session.post(
        url=api_root + "login/",
        data=urlencode({"jsonRequest": login_post}).replace("+", "").replace("%27", "%22"),
        # Hand-mangling the request for POST. Might remove later.
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
    ).json()["data"]
    return session_key


def check_for_s2_data_by_date(aoi_path, start_date, end_date, conf, cloud_cover=50):
    """
    Gets all the products between start_date and end_date. Wraps sent2_query to avoid having passwords and
    long-format timestamps in code.

    Parameters
    ----------
    aoi_path : str
        Path to a geojson file containing a polygon of the outline of the area you wish to download.
        See www.geojson.io for a tool to build these.

    start_date : str
        Start date in the format yyyymmdd.

    end_date : str
        End date of the query in the format yyyymmdd

    conf : dict
        Output from a configuration file containing your username and password for the ESA hub.
        If needed, this can be dummied with a dictionary of the following format:

        .. code:: python

            conf={'sent_2':{'user':'your_username', 'pass':'your_pass'}}

    cloud_cover : int
        The maximum level of cloud cover in images to be downloaded.

    Returns
    -------
    result : dict
        A dictionary of Sentinel 2 products.

    """
    log.info("Querying for imagery between {} and {} for aoi {}".format(start_date, end_date, aoi_path))
    user = conf['sent_2']['user']
    password = conf['sent_2']['pass']
    start_timestamp = dt.datetime.strptime(start_date, '%Y%m%d').isoformat(timespec='seconds') + 'Z'
    end_timestamp = dt.datetime.strptime(end_date, '%Y%m%d').isoformat(timespec='seconds') + 'Z'
    result = sent2_query(user, password, aoi_path, start_timestamp, end_timestamp, cloud=cloud_cover)
    log.info("Search returned {} images".format(len(result)))
    return result


def filter_to_l1_data(query_output):
    """
    Takes list of products from check_for_s2_data_by_date and removes all non Level 1 products.

    Parameters
    ----------
    query_output : dict
        A dictionary of products from a S2 query

    Returns
    -------
    filtered_query : dict
        A dictionary of products containing only the L1C data products

    """
    log.info("Extracting only L1 data from {} products".format(len(query_output)))
    filtered_query = {key: value for (key, value) in query_output.items() if get_query_level(value) == "Level-1C"}
    return filtered_query


def filter_to_l2_data(query_output):
    """
    Takes list of products from check_for_s2_data_by_date and removes all non Level 2A products.

    Parameters
    ----------
    query_output : dict
        A dictionary of products from a S2 query

    Returns
    -------
    filtered_query : dict
        A dictionary of products containing only the L2A data products

    """
    log.info("Extracting only L2 data from {} products".format(len(query_output)))
    filtered_query = {key: value for (key, value) in query_output.items() if get_query_level(value) == "Level-2A"}
    return filtered_query


def filter_non_matching_s2_data(query_output):
    """
    Filters a query such that it only contains paired level 1 and level 2 data products.

    Parameters
    ----------
    query_output : dict
        Query list

    Returns
    -------
    filtered_query : dict
        A dictionary of products containing only L1 and L2 data.

    """
    # Here be algorithms
    # A L1 and L2 image are related if and only if the following fields match:
    #    Satellite (S2[A|B])
    #    Intake date (FIRST timestamp)
    #    Orbit number (Rxxx)
    #    Granule ID (Txxaaa)
    # So if we succeviely partition the query, we should get a set of products with either 1 or
    # 2 entries per granule / timestamp combination
    sorted_query = sorted(query_output.values(), key=get_query_granule)
    granule_groups = {str(key): list(group) for key, group in itertools.groupby(sorted_query, key=get_query_granule)}
    granule_date_groups = {}

    # Partition as above.
    # We can probably expand this to arbitrary lengths of queries. If you catch me doing this, please restrain me.
    for granule, item_list in granule_groups.items():
        item_list.sort(key=get_query_datatake)
        granule_date_groups.update(
            {str(granule) + str(key): list(group) for key, group in
             itertools.groupby(item_list, key=get_query_datatake)})

    # On debug inspection, turns out sometimes S2 products get replicated. Lets filter those.
    out_set = {}
    for key, image_set in granule_date_groups.items():
        # if sum(1 for image in image_set if get_query_level(image) == "Level-2A") <= 2:
        # list(filter(lambda x: get_query_level(x) == "Level-2A", image_set)).sort(key=get_query_processing_time)[0].pop()
        if (sum(1 for image in image_set if get_query_level(image) == "Level-2A") == 1
                and sum(1 for image in image_set if get_query_level(image) == "Level-1C") == 1):
            out_set.update({image["uuid"]: image for image in image_set})

    # Finally, check that there is actually something here.
    if len(out_set) == 0:
        log.error(
            "No L2 data detected for query. Please remove the --download_l2_data flag or request more recent images.")
        raise NoL2DataAvailableException
    return out_set


def get_query_datatake(query_item):
    """
    Gets the datatake timestamp of a query item.

    Parameters
    ----------
    query_item : dict
        An item from a query results dictionary.

    Returns
    -------
    timestamp : str
        The timestamp of that item's datatake in the format yyyymmddThhmmss (Ex: 20190613T123002)
    """
    return query_item['beginposition']


def get_query_granule(query_item):
    """
    Gets the granule ID (ex: 48MXU) of a query

    Parameters
    ----------
    query_item : dict
        An item from a query results dictionary.

    Returns
    -------
    granule_id : str
        The granule ID of that item.

    """
    return query_item["title"].split("_")[5]


def get_query_processing_time(query_item):
    """
    Returns the processing timestamps of a query item

    Parameters
    ----------
    query_item : dict
        An item from a query results dictionary.

    Returns
    -------
    processing_time : str
        The date processing timestamp in the format yyyymmddThhmmss (Ex: 20190613T123002)

    """
    ingestion_string = query_item["title"].split("_")[6]
    return dt.datetime.strptime(ingestion_string, "%Y%m%dT%H%M%S")


def get_query_level(query_item):
    """
    Returns the processing level of the query item.

    Parameters
    ----------
    query_item : dict
         An item from a query results dictionary.

    Returns
    -------
    query_level : str
        A string of either 'Level-1C' or 'Level-2A'.

    """
    return query_item["processinglevel"]


def get_granule_identifiers(safe_product_id):
    """
    Returns the parts of a S2 name that uniquely identify that granulate at a moment in time
    Parameters
    ----------
    safe_product_id : str
        The filename of a SAFE product

    Returns
    -------
    satellite : str
        A string of either "L2A" or "L2B"
    intake_date : str
        The timestamp of the data intake of this granule
    orbit number : str
        The orbit number of this granule
    granule : str
        The ID of this granule

    """
    satellite, _, intake_date, _, orbit_number, granule, _ = safe_product_id.split('_')
    return satellite, intake_date, orbit_number, granule


def download_s2_data(new_data, l1_dir, l2_dir, source='scihub', user=None, passwd=None, try_scihub_on_fail=False):
    """
    Downloads S2 imagery from AWS, google_cloud or scihub. new_data is a dict from Sentinel_2.

    Parameters
    ----------
    new_data : dict
        A query dictionary contining the products you want to download
    l1_dir : str
        The directory to download level 1 products to.
    l2_dir : str
        The directory to download level 2 products to.
    source : {'scihub', 'aws'}
        The source to download the data from. Can be 'scihub' or 'aws'; see section introduction for details
    user : str, optional
        The username for sentinelhub
    passwd : str, optional
        The password for sentinelhub
    try_scihub_on_fail : bool, optional
        If true, this function will roll back to downloading from Scihub on a failure of any other downloader. Defaults
        to `False`.

    Raises
    ------
    BadDataSource
        Raised when passed either a bad datasource or a bad image ID

    """
    for image_uuid in new_data:
        identifier = new_data[image_uuid]['identifier']
        if 'L1C' in identifier:
            out_path = os.path.join(l1_dir, identifier + ".SAFE")
            if check_for_invalid_l1_data(out_path) == 1:
                log.info("L1 imagery exists, skipping download")
                continue
        elif 'L2A' in identifier:
            out_path = os.path.join(l2_dir, identifier + ".SAFE")
            if check_for_invalid_l2_data(out_path) == 1:
                log.info("L2 imagery exists, skipping download")
                continue
        else:
            log.error("{} is not a Sentinel 2 product".format(identifier))
            raise BadDataSourceExpection
        out_path = os.path.dirname(out_path)
        log.info("Downloading {} from {} to {}".format(new_data[image_uuid]['identifier'], source, out_path))
        if source == 'aws':
            if try_scihub_on_fail:
                download_from_aws_with_rollback(product_id=new_data[image_uuid]['identifier'], folder=out_path,
                                                uuid=image_uuid, user=user, passwd=passwd)
            else:
                download_safe_format(product_id=new_data[image_uuid]['identifier'], folder=out_path)
        elif source == 'google':
            download_from_google_cloud([new_data[image_uuid]['identifier']], out_folder=out_path)
        elif source == "scihub":
            download_from_scihub(image_uuid, out_path, user, passwd)
        else:
            log.error("Invalid data source; valid values are 'aws', 'google' and 'scihub'")
            raise BadDataSourceExpection


def download_s2_pairs(l1_dir, l2_dir, conf):
    """
    Given a pair of folders, one containing l1 products and the other containing l2 products, will query and download
    missing data. At the end of the run, you will have two folders with a set of paired L1 and L2 products.
    Parameters
    ----------
    l1_dir : str
        The directory to download level 1 products to. May contain existing products.
    l2_dir : str
        The directory to download level 2 products to. May contain existing products.
    conf : dict
        A dictionary containing ['sent_2']['user'] and ['sent_2']['pass']

    """
    # God, this is a faff.
    l1_product_list = os.listdir(l1_dir)
    l2_product_list = os.listdir(l2_dir)
    missing_products = []
    for l1_prod in l1_product_list:
        if not fu.get_l2_safe_file(l1_prod, l2_dir):
            missing_products.append(l1_prod)
    for l2_prod in l2_product_list:
        if not fu.get_l1_safe_file(l2_prod, l1_dir):
            missing_products.append(l2_prod)
    to_download = {}
    log.info("{} missing products: {}".format(len(missing_products), missing_products))
    for prod in missing_products:
        to_download.update(query_for_corresponding_image(prod, conf))
    if len(to_download) < len(missing_products):
        log.warning("Could not find all corresponding products - please check folder after download")
    download_s2_data(to_download, l1_dir, l2_dir, user=conf['sent_2']['user'], passwd=conf['sent_2']['pass'])


def query_for_corresponding_image(prod,conf):
    """
    Queries Copernicus Hub for the corresponding l1/l2 image to 'prod'

    Parameters
    ----------
    prod : str
        The product name to query

    conf : dict
        A dictionary containing ['sent_2']['user'] and ['sent_2']['pass']

    Returns
    -------
    out : dict
        A Sentinel-2 product dictionary

    """
    date_string = fu.get_sen_2_image_timestamp(prod)
    date = dt.datetime.strptime(date_string, "%Y%m%dT%H%M%S").date()
    tile = fu.get_sen_2_image_tile(prod)[1:]  # Strip first 'T'
    if fu.get_safe_product_type(prod) == "MSIL1C":
        level = "Level-2A"
    elif fu.get_safe_product_type(prod) == "MSIL2A":
        level = "Level-1C"
    user = conf['sent_2']['user']
    passwd = conf['sent_2']['pass']
    api = SentinelAPI(user, passwd)
    # These from https://sentinelsat.readthedocs.io/en/stable/api.html#search-sentinel-2-by-tile
    query_kwargs = {
        'platformname': 'Sentinel-2',
        'date': (date-dt.timedelta(days=1), date+dt.timedelta(days=1)),
        'tileid': tile,
        'processinglevel': level
    }
    out = api.query(**query_kwargs)
    return out


def download_from_aws_with_rollback(product_id, folder, uuid, user, passwd):
    """
    Attempts to download a single product from AWS using product_id; if not found, rolls back to Scihub using the UUID

    Parameters
    ----------
    product_id : str
        The product ID ("L2A_...")
    folder : str
        The folder to download the .SAFE file to.
    uuid : str
        The product UUID (4dfB4-432df....)
    user : str
        Scihub username
    passwd : str
        Scihub password

    """
    log = logging.getLogger(__file__)
    try:
        download_safe_format(product_id=product_id, folder=folder)
    except ClientError:
        log.warning(
            "Something wrong with AWS for products id {}; rolling back to Scihub using uuid {}".format(product_id,
                                                                                                       uuid))
        download_from_scihub(uuid, folder, user, passwd)


def download_from_scihub(product_uuid, out_folder, user, passwd):
    """
    Downloads and unzips product_uuid from scihub

    Parameters
    ----------
    product_uuid : str
        The product UUID (4dfB4-432df....)
    out_folder : str
        The folder to save the .SAFE file to
    user : str
        Scihub username
    passwd : str
        Scihub password

    Notes
    -----
    If interrupted mid-download, there will be a .incomplete file in the download folder. You might need to remove
    this for further processing.

    """
    api = SentinelAPI(user, passwd)
    api.api_url = "https://apihub.copernicus.eu/apihub/"
    log.info("Downloading {} from scihub".format(product_uuid))
    prod = api.download(product_uuid, out_folder)
    if not prod:
        log.error("{} not found. Please check.".format(product_uuid))
    if not prod["Online"]:
        log.info("{} is being retrieved from long-term archive. Please try again later.".format(product_uuid))
        return 1
    zip_path = os.path.join(out_folder, prod['title'] + ".zip")
    log.info("Unzipping {} to {}".format(zip_path, out_folder))
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(out_folder)
    zip_ref.close()
    log.info("Removing {}".format(zip_path))
    os.remove(zip_path)
    return 0


def download_from_google_cloud(product_ids, out_folder, redownload=False):
    """
    :meta private:
    Still experimental.
    """
    log = logging.getLogger(__name__)
    log.info("Downloading following products from Google Cloud:".format(product_ids))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("gcp-public-data-sentinel-2")
    for safe_id in product_ids:
        if not safe_id.endswith(".SAFE"):
            safe_id = safe_id + ".SAFE"
        if check_for_invalid_l1_data(os.path.join(out_folder, safe_id)) and not redownload:
            log.info("File exists, skipping.")
            return
        if redownload:
            log.info("Removing {}".format(os.path.join(out_folder, safe_id)))
            shutil.rmtree(os.path.join(out_folder, safe_id))
        tile_id = get_sen_2_image_tile(safe_id)
        utm_zone = tile_id[1:3]
        lat_band = tile_id[3]
        grid_square = tile_id[4:6]
        object_prefix = r"tiles/{}/{}/{}/{}/".format(
            utm_zone, lat_band, grid_square, safe_id
        )
        object_iter = bucket.list_blobs(prefix=object_prefix, delimiter=None)
        if object_iter.num_results == 0:
            log.error("{} missing from Google Cloud, continuing".format(safe_id))
            continue
        for s2_object in object_iter:
            download_blob_from_google(bucket, object_prefix, out_folder, s2_object)
        # Need to make these two empty folders for sen2cor to work properly
        try:
            os.mkdir(os.path.join(os.path.abspath(out_folder), safe_id, "AUX_DATA"))
            os.mkdir(os.path.join(os.path.abspath(out_folder), safe_id, "HTML"))
        except FileExistsError:
            pass


def download_blob_from_google(bucket, object_prefix, out_folder, s2_object):
    """
    :meta private:
    Still experimental.
    """
    log = logging.getLogger(__name__)
    blob = bucket.get_blob(s2_object.name)
    object_out_path = os.path.join(
        os.path.abspath(out_folder),
        s2_object.name.replace(os.path.dirname(object_prefix.rstrip('/')), "").strip('/')
    )
    os.makedirs(os.path.dirname(object_out_path), exist_ok=True)
    log.info("Downloading from {} to {}".format(s2_object, object_out_path))
    with open(object_out_path, 'w+b') as f:
        blob.download_to_file(f)


def load_api_key(path_to_api):
    """
    Returns an API key from a single-line text file containing that API

    Parameters
    ----------
    path_to_api : str
        The path a text file containing only the API key

    Returns
    -------
    api_key : str
        Returns the API key
    """
    with open(path_to_api, 'r') as api_file:
        return api_file.read()


def get_planet_product_path(planet_dir, product):
    """
    :meta private:
    Returns the path to a Planet product within a Planet directory
    """
    planet_folder = os.path.dirname(planet_dir)
    product_file = glob.glob(planet_folder + '*' + product)
    return os.path.join(planet_dir, product_file)


def download_planet_image_on_day(aoi_path, date, out_path, api_key, item_type="PSScene4Band", search_name="auto",
                                 asset_type="analytic", threads=5):
    """
    :meta private:
    Queries and downloads all images on the date in the aoi given
    """
    log = logging.getLogger(__name__)
    start_time = date + "T00:00:00.000Z"
    end_time = date + "T23:59:59.000Z"
    try:
        planet_query(aoi_path, start_time, end_time, out_path, api_key, item_type, search_name, asset_type, threads)
    except IndexError:
        log.warning("IndexError exception; likely no imagery available for chosen date")


def planet_query(aoi_path, start_date, end_date, out_path, api_key, item_type="PSScene4Band", search_name="auto",
                 asset_type="analytic", threads=5):
    """
    Downloads data from Planetlabs for a given time period in the given AOI

    Parameters
    ----------
    aoi : str
        Filepath of a single-polygon geojson containing the aoi

    start_date : str
        the inclusive start of the time window in UTC format

    end_date : str
        the inclusive end of the time window in UTC format

    out_path : filepath-like object
        A path to the output folder
        Any identically-named imagery will be overwritten

    item_type : str
        Image type to download (see Planet API docs)

    search_name : str
        A name to refer to the search (required for large searches)

    asset_type : str
        Planet asset type to download (see Planet API docs)

    threads : int
        The number of downloads to perform concurrently

    Notes
    -----
    IMPORTANT: Will not run for searches returning greater than 250 items.

    """
    feature = read_aoi(aoi_path)
    aoi = feature['geometry']
    session = requests.Session()
    session.auth = (api_key, '')
    search_request = build_search_request(aoi, start_date, end_date, item_type, search_name)
    search_result = do_quick_search(session, search_request)

    thread_pool = Pool(threads)
    threaded_dl = lambda item: activate_and_dl_planet_item(session, item, asset_type, out_path)
    thread_pool.map(threaded_dl, search_result)


def build_search_request(aoi, start_date, end_date, item_type, search_name):
    """
    :meta private:
    Builds a search request for the planet API
    """
    date_filter = planet_api.filters.date_range("acquired", gte=start_date, lte=end_date)
    aoi_filter = planet_api.filters.geom_filter(aoi)
    query = planet_api.filters.and_filter(date_filter, aoi_filter)
    search_request = planet_api.filters.build_search_request(query, [item_type])
    search_request.update({'name': search_name})
    return search_request


def do_quick_search(session, search_request):
    """
    :meta private:
    Tries the quick search; returns a dict of features
    """
    search_url = "https://api.planet.com/data/v1/quick-search"
    search_request.pop("name")
    print("Sending quick search")
    search_result = session.post(search_url, json=search_request)
    if search_result.status_code >= 400:
        raise requests.ConnectionError
    return search_result.json()["features"]


def do_saved_search(session, search_request):
    """
    :meta private:
    Does a saved search; this doesn't seem to work yet.
    """
    search_url = "https://api.planet.com/data/v1/searches/"
    search_response = session.post(search_url, json=search_request)
    search_id = search_response.json()['id']
    if search_response.json()['_links'].get('_next_url'):
        return get_paginated_items(session)
    else:
        search_url = "https://api-planet.com/data/v1/searches/{}/results".format(search_id)
        response = session.get(search_url)
        items = response.content.json()["features"]
        return items


def get_paginated_items(session, search_id):
    """
    :meta private:
    Let's leave this out for now.
    """
    raise Exception("pagination not handled yet")


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_delay(10000),
    retry=tenacity.retry_if_exception_type(TooManyRequests)
)
def activate_and_dl_planet_item(session, item, asset_type, file_path):
    """
    :meta private:
    Activates and downloads a single planet item
    """
    log = logging.getLogger(__name__)
    #  TODO: Implement more robust error handling here (not just 429)
    item_id = item["id"]
    item_type = item["properties"]["item_type"]
    item_url = "https://api.planet.com/data/v1/" + \
               "item-types/{}/items/{}/assets/".format(item_type, item_id)
    item_response = session.get(item_url)
    log.info("Activating " + item_id)
    activate_response = session.post(item_response.json()[asset_type]["_links"]["activate"])
    while True:
        status = session.get(item_url)
        if status.status_code == 429:
            log.warning("ID {} too fast; backing off".format(item_id))
            raise TooManyRequests
        if status.json()[asset_type]["status"] == "active":
            break
    dl_link = status.json()[asset_type]["location"]
    item_fp = os.path.join(file_path, item_id + ".tif")
    log.info("Downloading item {} from {} to {}".format(item_id, dl_link, item_fp))
    # TODO Do we want the metadata in a separate file as well as embedded in the geotiff?
    with open(item_fp, 'wb+') as fp:
        image_response = session.get(dl_link)
        if image_response.status_code == 429:
            raise TooManyRequests
        fp.write(image_response.content)  # Don't like this; it might store the image twice. Check.
        log.info("Item {} download complete".format(item_id))


def read_aoi(aoi_path):
    """
    Opens the geojson file for the aoi. If FeatureCollection, return the first feature.

    Parameters
    ----------
    aoi_path : str
        The path to the geojson file
    Returns
    -------
    aoi_dict : dict
        A dictionary translation of the feature inside the .json file

    """
    with open(aoi_path, 'r') as aoi_fp:
        aoi_dict = json.load(aoi_fp)
        if aoi_dict["type"] == "FeatureCollection":
            aoi_dict = aoi_dict["features"][0]
        return aoi_dict
