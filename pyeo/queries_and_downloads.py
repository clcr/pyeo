"""
Functions for querying and downloading data from the Copernicus Dataspace Ecosystem (CDSE).

Key Functions
-------------
To interact with the CDSE, two key functions are required.

1. :py:func:`query_dataspace_by_polygon` Queries the Copernicus Dataspace Ecosystem API for products between two dates that conform to the Area of Interest and maximum cloud cover supplied.

Once the appropriate products are identified, then the query :code:pd.DataFrame can be downloaded using:

2. :py:func:`download_s2_data_from_dataspace` Passes a DataFrame of L2A and L1C products to :py:func:`download_dataspace_product`, which handles for authentication errors, URL redirects and token refreshes.

SAFE Files
----------
Sentinel-2 data is downloaded in the form of a .SAFE file; all download functions will end with data in this structure.
This is a directory structure that contains the imagery, metadata and supplementary data of a Sentinel 2 image. The
rasters themeselves are the in the GRANULE/[granule_id]/IMG_DATA/[resolution]/ folder; each band is contained in
its own .jp2 file. For full details, see https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats

There are two ways to refer to a given Sentinel-2 products: the UUID and the product ID.
The UUID is an alphanumeric string (e.g. 22e7af63-07ad-4076-8541-f6655388dc5e), whereas the product ID is a human-readable string (more or less) containing all the information needed for unique identification of an product, split by the underscore character.

Query Data Structure
--------------------
All query functions return a dictionary. The key of the dictionary is the UUID of the product id; the product is a further set of nested dictionaries containing information about the product to be downloaded.

Data Download Source
--------------------
The only download source currently provided is via the Copernicus Dataspace Ecosystem (CDSE): https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html 

- Copernicus DataSpace Ecosystem

    Images are downloaded in .zip format, and `pyeo` handles the unzipping, conversion from .jp2 to .tif.
    Users do not need to be registered with the CDSE to query images, but `pyeo` expects the user to have provided a valid username and password. 
    The main change from SciHub to CDSE is that Sentinel-2 products are no longer archived beyond a certain time-frame, i.e. the products in the CDSE are always online.

Legacy Download Sources
-----------------------
There are three other *legacy* download sources which this library no longer supports as they are deprecated in favour of the now active CDSE.

- Scihub

   The Copernicus Open-Access Hub is the default option for downloading sentinel-2 images. Images are downloaded in .zip
   format, and then automatically unzipped. Users are required to register with a username and password before downloading,
   and there is a limit to no more than two concurrent downloads per username at a time. Scihub is entirely free.
   Older images are moved to the long-term archive and have to be requested.

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


# :py:func:`check_for_s2_data_by_date` Queries the Sentinel 2 archive for products between two dates
# :py:func:`download_s2_data` Downloads Sentinel 2 data from Scihub by default

import datetime as dt
import glob
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from tqdm import tqdm
import zipfile
from multiprocessing.dummy import Pool
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pyeo.filesystem_utilities as fu
import pyeo.windows_compatability
import requests
import tenacity
from botocore.exceptions import ClientError
from bs4 import \
    BeautifulSoup  # I didn't really want to use BS, but I can't see a choice.

from osgeo import ogr, osr
from pyeo.coordinate_manipulation import (get_vector_projection,
                                            reproject_vector)
from pyeo.exceptions import (BadDataSourceExpection,
                               InvalidDateFormatException,
                               InvalidGeometryFormatException,
                               NoL2DataAvailableException, TooManyRequests)
from pyeo.filesystem_utilities import (check_for_invalid_l1_data,
                                         check_for_invalid_l2_data,
                                         get_sen_2_image_tile)
from requests import Request
from sentinelhub.data_request import download_safe_format
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

log = logging.getLogger("pyeo")

try:
    from google.cloud import storage
except ImportError:
    pass

api_url = "https://scihub.copernicus.eu/dhus/"
rest_url = "https://apihub.copernicus.eu/apihub/search"
# api_url = "https://apihub.copernicus.eu/apihub/"

# dataspace constants
DATASPACE_API_ROOT = "http://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
DATASPACE_DOWNLOAD_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DATASPACE_REFRESH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

def query_dataspace_by_polygon(
    max_cloud_cover: int,
    start_date: str,
    end_date: str,
    area_of_interest: str,
    max_records: int,
    log: logging.Logger
) -> pd.DataFrame:
    """
    This function:
    
    Returns a DataFrame of available Sentinel-2 imagery from the Copernicus Dataspace API.

    Parameters
    ----------
    max_cloud_cover : int
        Maximum Cloud Cover
    start_date : str
        Start date of the images to query from in (YYYY-MM-DD) format
    end_date : str
        End date of the images to query from in (YYYY-MM-DD) format
    area_of_interest : str
        Region of interest centroid in WKT format
    max_records : int
        Maximum records to return

    Returns
    -------
    None
    """

    request_string = build_dataspace_request_string(
        max_cloud_cover=max_cloud_cover,
        start_date=start_date,
        end_date=end_date,
        area_of_interest=area_of_interest,
        max_records=max_records,
    )
    response = requests.get(request_string)
    if response.status_code == 200:
        response = response.json()["features"]
        # log.info(json.dumps(response, indent=4))
        # sys.exit(1)
        response_dataframe = pd.DataFrame.from_dict(response)
        response_dataframe = pd.DataFrame.from_records(response_dataframe["properties"])
        return response_dataframe
    elif response.status_code == 401:
        log.error("Dataspace returned a 401 HTTP Status Code")
        log.error("Which means that user credentials for the Copernicus Dataspace Ecosystem are incorrect.")
        log.error("Now exiting the pipeline, please check your credentials in your credentials_ini")
        sys.exit(1)
    else:
        log.error("Dataspace returned a non-200 HTTP Status Code")
        log.error(f"The Status Code returned was  : HTTP Status Code {response.status_code}")
        log.error("Now exiting the pipeline, please rerun when DataSpace API is back online")
        # this could be improved by catching more specific status codes
        sys.exit(1)

def build_dataspace_request_string(
    max_cloud_cover: int,
    start_date: str,
    end_date: str,
    area_of_interest: str,
    max_records: int,
) -> str:
    """
    This function:
    
    Builds the API product request string based on given properties and constraints.

    Parameters
    ----------
    max_cloud_cover : int
        Maximum cloud cover to allow in the queried products
    start_date : str
        Starting date of the observations (YYYY-MM-DD format)
    end_date : str
        Ending date of the observations (YYYY-MM-DD format)
    area_of_interest : str
        Area of interest geometry as a string in WKT format
    max_records : int
        Maximum number of products to show per query (queries with very high numbers may not complete in time)

    Returns
    -------
    request_string : str
        API Request String

    """
    cloud_cover_props = f"cloudCover=[0,{max_cloud_cover}]"
    start_date_props = f"startDate={start_date}"
    end_date_props = f"completionDate={end_date}"
    geometry_props = f"geometry={area_of_interest}"
    max_records_props = f"maxRecords={max_records}"

    request_string = f"{DATASPACE_API_ROOT}?{cloud_cover_props}&{start_date_props}&{end_date_props}&{geometry_props}&{max_records_props}"
    return request_string

def get_access_token(dataspace_username: str = None,
                     dataspace_password: str = None,
                     refresh_token: str = None) -> str:
    """

    This function:
    
    Creates an access token to use during download for verification purposes.

    Parameters
    ----------

    dataspace_username : str
        The username registered with the Copernicus Open Access Dataspace

    dataspace_password : str
        The password registered with the Copernicus Open Access Dataspace

    refresh : bool
        Refreshes an old access token, Default false - returns new access token


    Returns
    -------
    response : str

    """

    if refresh_token:
        print("refreshing access token...")
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": "cdse-public",
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(DATASPACE_REFRESH_TOKEN_URL, data=payload, headers=headers).json()
        
    else:
        payload = {
            "grant_type": "password",
            "username": dataspace_username,
            "password": dataspace_password,
            "client_id": "cdse-public",
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            response = requests.post(
                DATASPACE_REFRESH_TOKEN_URL, data=payload, headers=headers
            ).json()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {response}"
                )
    return response

def download_s2_data_from_dataspace(product_df: pd.DataFrame,
                                    l1c_directory: str,
                                    l2a_directory: str,
                                    dataspace_username: str,
                                    dataspace_password: str,
                                    log: logging.Logger
                                    ) -> None:
    """
    
    This function:
    
    Wraps around `download_dataspace_product`, providing the necessary directories dependent on product type (L1C/L2A).

    Parameters
    ----------

    product_df : pd.DataFrame
        A Pandas DataFrame containing the products to download. 
        
    l1c_directory : str
        The path to the L1C download directory.
    
    l2a_directory : str
        The path to the L2A download directory.
    
    dataspace_username : str
        The username registered with the Copernicus Open Access Dataspace.

    dataspace_password : str
        The password registered with the Copernicus Open Access Dataspace.

    log : logging.Logger
        Log object to write to.

    Returns
    -------
    None

    """
        
    for counter, product in enumerate(product_df.itertuples(index=False)):
        log.info(f"    Checking {counter+1} of {len(product_df)} : {product.title}")
        # if L1C have been passed, download to the l1c_directory
        if product.processinglevel == "Level-1C":

            out_path = os.path.join(l1c_directory, product.title)
            if check_for_invalid_l1_data(out_path) == 1:
                log.info(f"        {out_path} imagery already exists, skipping download")
                # continue means skip the current iteration and move to the next iteration of the for loop
                continue
            try:
                log.info(f"        Downloading : {product.title}")
                download_dataspace_product(
                    product_uuid=product.uuid,
                    dataspace_username=dataspace_username,
                    dataspace_password=dataspace_password,
                    product_name=product.title,
                    safe_directory=l1c_directory,
                    log=log
                )
            
            except Exception as error:
                log.error(f"Download from dataspace of L1C Product did not finish")
                log.error(f"Received this error :  {error}")
            
        # if L2A have been passed, download to the l1c_directory
        elif product.processinglevel == "Level-2A":
            out_path = os.path.join(l2a_directory, product.title)
            if check_for_invalid_l2_data(out_path) == 1:
                log.info(f"        {out_path} imagery already exists, skipping download")
                # continue means to skip the current iteration and move to the next iteration of the for loop
                continue
            try:
                log.info(f"        Downloading  : {product.title}")
                download_dataspace_product(
                    product_uuid=product.uuid,
                    dataspace_username=dataspace_username,
                    dataspace_password=dataspace_password,
                    product_name=product.title,
                    safe_directory=l2a_directory,
                    log=log
                )
            except Exception as error:
                log.error(f"Download from dataspace of L2A Product did not finish")
                log.error(f"Received error   {error}")

        else:
            log.error(f"Neither 'Level-1C' or 'Level-2A' were in {product.processinglevel}")
            log.error("could be a bad data source, therefore skipping")

    return

def download_dataspace_product(product_uuid: str,
                               dataspace_username: str,
                               dataspace_password: str,
                               product_name: str,
                               safe_directory: str,
                               log: logging.Logger
                               ) -> None:
    """
    This function:
    
    Downloads a Sentinel-2 product using the given product UUID from the ESA servers.

    Parameters
    ----------
    product_uuid : str
        UUID of the product to download
    dataspace_username: str
        username used to access the CDSE.
    dataspace_password: str
        password associated with username.
    product_name : str
        Name of the product
    safe_directory : str
        The directory (path) to write the SAFE files to

    Returns
    -------
    None
    
    Notes
    -----

    Registration to the Copernicus Dataspace Ecosystem (CDSE) is free. Register here: https://dataspace.copernicus.eu
    """

    auth_response = get_access_token(
        dataspace_username=dataspace_username,
        dataspace_password=dataspace_password,
        )

    ############################
    # auth limited to 10 minutes
    auth_access_token = auth_response["access_token"]
    # # refresh limited to 1 hour

    session = requests.Session()
    session.headers.update({'Authorization': f"Bearer {auth_access_token}"})
    url=f"{DATASPACE_DOWNLOAD_URL}({product_uuid})/$value"

    log.info('Obtaining the download URL - via redirect from url constructed from uuid')
    response = session.get(url, allow_redirects=False)
    while response.status_code in (301, 302, 303, 307):
        log.info(f"response.status_code: {response.status_code}")
        log.info(f"download url = response.headers['Location']: {url}")
        url = response.headers['Location']
        response = session.get(url, allow_redirects=False)

    log.info(f"Final response redirects to url: {url}")

    log.info('Refresh access token as url redirect may take longer than 600s token expiry time')
    auth_refresh_token = auth_response["refresh_token"]
    auth_response = get_access_token(
        dataspace_username=dataspace_username,
        dataspace_password=dataspace_password,
        refresh_token=auth_refresh_token,
    )
    auth_access_token = auth_response["access_token"]
    session.headers.update({'Authorization': f"Bearer {auth_access_token}"})

    log.info('Download the zipped image file')
    file = session.get(url, verify=False, allow_redirects=True)

    min_file_size = 2000  # in bytes
    if (len(file.content) <= min_file_size):
        log.info(f'  Downloaded file too small, length: {len(file.content)} bytes, contents: {file.content}')

    # The following fix is needed on Windows to avoid errors because of long file names.
    # But it causes a directory error on the Linux HPC because it resolves the user home directory wrongly.
    if sys.platform.startswith("win"):
        temp_dir_platform_specific = os.path.expanduser('~')
    else:
        temp_dir_platform_specific = os.path.split(safe_directory)[0]

    with TemporaryDirectory(dir=temp_dir_platform_specific) as temp_dir:
        temporary_path = f"{temp_dir}{os.sep}{product_name}.zip"
        log.info(f"Downloaded file temporary_path: {temporary_path}")

        with open(temporary_path, 'wb') as download:

            download.write(file.content)
        unzipped_path = os.path.splitext(temporary_path)[0]
        destination_path = f"{safe_directory}{os.sep}{product_name}"
        log.info(f"Downloaded file destination path: {destination_path}")

        downloaded_file_size = os.path.getsize(temporary_path)
        log.info(f"Downloaded file size: {downloaded_file_size} bytes")
        if (downloaded_file_size < min_file_size):
            log.info(f'  Downloaded file too small, contents are:')
            file_dnld = open(temporary_path, 'r')
            log.info(file_dnld.readline())
            file_dnld.close()

        log.info("    unpacking archive...")
        shutil.unpack_archive(temporary_path, unzipped_path)

        log.info(f"Unpacked Archive Path: {unzipped_path}")

        # # restructure paths
        within_folder_path = glob.glob(os.path.join(unzipped_path, "*"))
        log.info(f"Downloaded file within_folder path: {within_folder_path[0]}")

        log.info("    moving directory...")
        shutil.move(src=within_folder_path[0], dst=destination_path)

    return

def filter_unique_dataspace_products(l1c_products: pd.DataFrame,
                                     l2a_products: pd.DataFrame,
                                     log: logging.Logger
                                     ) -> pd.DataFrame:
    """

    Parameters
    ----------
    l1c_products : pd.DataFrame
        Pandas DataFrame containing a list of L1C products.
    l2a_products : pd.DataFrame
        Pandas DataFrame containing a list of L2A products.
    log : logging.Logger
        logging object.

    Returns
    -------
    unique_l1c_products: pd.DataFrame
        A Pandas DataFrame of L1C products that do not have counterpart L2A products.
    """

    log.info(f"Before filtering, {len(l1c_products)} L1C and {len(l2a_products)} L2A")
    for rows in l1c_products.itertuples():
        log.info(f"L1C : {rows.title}")
    for rows in l2a_products.itertuples():
        log.info(f"L2A : {rows.title}")
    # Perform left anti-join
    unique_l1c_products = l1c_products.merge(
        l2a_products, 
        on='beginposition', 
        how='left', 
        indicator=True
    ).query('_merge == "left_only"')

    # Remove the indicator column
    unique_l1c_products = unique_l1c_products.drop('_merge', axis=1)
    # remove the suffix _x
    unique_l1c_products.columns = [col.replace('_x', '') for col in unique_l1c_products.columns]

    # Output the missing l1c_products
    log.info(f"After filtering, {len(unique_l1c_products)} L1C and {len(l2a_products)} L2A")
    for rows in unique_l1c_products.itertuples():
        log.info(f"Unique L1C : {rows.title}")
    for rows in l2a_products.itertuples():
        log.info(f"Unique L2A : {rows.title}")

    return unique_l1c_products


def _rest_query(
    user: str,
    passwd: str,
    footprint_wkt: str,
    start_date: str,
    end_date: str,
    cloud: int=100,
    start_row: int=0,
    producttype: str=None,
    filename: str=None,
):
    """

    Parameters
    ----------
    user : str
    passwd (str): [description]
    footprint_wkt (str): [description]
    start_date (str): [description]
    end_date (str): [description]
    cloud (int, optional): [description]. Defaults to 100.
    start_row (int, optional): [description]. Defaults to 0.
    producttype (str, optional): [description]. Defaults to None.
    filename (str, optional): [description]. Defaults to None.

    Raises
    ------
    requests.exceptions.RequestException

    Returns
    -------
    rest_out_to_json
    """
    # Allows for more than 10 search results by implementing pagination
    # https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/OpenSearchAPI?redirectedfrom=SciHubUserGuide.6OpenSearchAPI
    # Results sets over the maximum can be obtained through paging of from different start values.
    #    Page 1: https://scihub.copernicus.eu/dhus/search?start=0&rows=100&q=*
    #    Page 2: https://scihub.copernicus.eu/dhus/search?start=100&rows=100&q=*
    #    Page 3: https://scihub.copernicus.eu/dhus/search?start=200&rows=100&q=*
    session = requests.Session()
    session.auth = (user, passwd)
    search_params = {
        "platformname": "((Sentinel-2))",
        "footprint": '("Intersects({})")'.format(footprint_wkt),
        "beginposition": "[{} TO {}]".format(start_date, end_date),
        "endposition": "[{} TO {}]".format(start_date, end_date),
        "cloudcoverpercentage": "[0 TO {}]".format(cloud),
        "producttype": "({})".format(producttype),
        "filename": "({})".format(filename),
    }
    search_string = " AND ".join(
        [f"{term}:{query}" for term, query in search_params.items()]
    )
    request_params = {"q": search_string, "rows": 100, "start": start_row}
    results = session.get(rest_url, timeout=600, params=request_params)
    if results.status_code >= 400:
        print("Bad request: code {}".format(results.status_code))
        print(results.content)
        raise requests.exceptions.RequestException
    return _rest_out_to_json(results)


def _file_api_query(
    user, passwd, start_date, end_date, filename, cloud=100, producttype="S2MSI2A"
):
    api = SentinelAPI(user, passwd, timeout=600)

    # try 20 times to connect to the server if it is not responding before producing an error
    @tenacity.retry(stop=tenacity.stop_after_attempt(20), wait=tenacity.wait_fixed(300))
    def query(*args, **kwargs):
        return api.query(*args, **kwargs)

    query_kwargs = {
        "platformname": "Sentinel-2",
        "date": (start_date, end_date),
        "cloudcoverpercentage": (0, cloud),
        "producttype": producttype,
    }
    kw = query_kwargs.copy()
    kw["raw"] = f"filename:{filename}"
    # products = api.query(**kw)
    products = query(**kw)  # call the query function with the tenacity decorator
    return products


def _tile_api_query(
    user,
    passwd,
    tile_id,
    start_date,
    end_date,
    cloud=100,
    start_row=0,
    producttype="S2MSI1C",
    filename=None,
):
    api = SentinelAPI(user, passwd, timeout=600)

    # try 20 times to connect to the server if it is not responding before producing an error
    @tenacity.retry(stop=tenacity.stop_after_attempt(20), wait=tenacity.wait_fixed(300))
    def query(*args, **kwargs):
        return api.query(*args, **kwargs)

    query_kwargs = {
        "platformname": "Sentinel-2",
        "cloudcoverpercentage": (0, cloud),
        "date": (start_date, end_date),
        "tileid": tile_id,
        #'rows': 100,
        #'startrow': start_row,
        #'url': api_url,
        "producttype": producttype,
    }
    kw = query_kwargs.copy()
    if filename is not None:
        kw["raw"] = f"filename:{filename}"
    # products = api.query(**kw)
    products = query(**kw)  # call the query function with the tenacity decorator
    return products


def _tile_query(
    user,
    passwd,
    tile_id,
    start_date,
    end_date,
    cloud=100,
    start_row=0,
    producttype="S2MSI1C",
):
    session = requests.Session()
    session.auth = (user, passwd)
    # rest_url = "https://apihub.copernicus.eu/apihub/search"

    search_params = {
        "platformname": "((Sentinel-2))",
        "producttype": producttype,
        "tileid": tile_id,
        "beginposition": "[{} TO {}]".format(start_date, end_date),
        "endposition": "[{} TO {}]".format(start_date, end_date),
        "cloudcoverpercentage": "[0 TO {}]".format(cloud),
    }
    search_string = " AND ".join(
        [f"{term}:{query}" for term, query in search_params.items()]
    )

    request_params = {"q": search_string, "rows": 100, "start": start_row}

    results = session.get(rest_url, timeout=600, params=request_params)
    if results.status_code >= 400:
        print("Bad request: code {}".format(results.status_code))
        print(results.content)
        raise requests.exceptions.RequestException
    return _rest_out_to_json(results)


def _rest_out_to_json(result):
    root = ElementTree.fromstring(result.content.replace(b"\n", b""))
    total_results = int(
        root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text
    )
    # if total_results > 10:
    #    log.info("Local querying can now return more than 10 search results.")
    if total_results == 0:
        log.warning("Query produced no results.")
    out = {}
    for element in root.findall("{http://www.w3.org/2005/Atom}entry"):
        id = element.find("{http://www.w3.org/2005/Atom}id").text
        out[id] = _parse_element(element)
        out[id].pop(None)
        out[id]["title"] = out[id]["identifier"] + ".SAFE"
    return out


def _parse_element(element):
    if len(element) == 0:
        if element.get("name"):
            return element.text
        else:
            return None
    else:
        out = {}
        for subelement in element:
            out[subelement.get("name")] = _parse_element(subelement)
        return out


def _sentinelsat_query(user, passwd, footprint_wkt, start_date, end_date, cloud=100):
    """
    Fetches a list of Sentinel-2 products
    timeout option below indicates how long to wait for a response (in seconds)
    """
    # Originally by Ciaran Robb
    api = SentinelAPI(user, passwd, timeout=600)

    # try 20 times to connect to the server if it is not responding before producing an error
    @tenacity.retry(stop=tenacity.stop_after_attempt(20), wait=tenacity.wait_fixed(300))
    def query(*args, **kwargs):
        return api.query(*args, **kwargs)

    products = query(
        footprint_wkt,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        cloudcoverpercentage="[0 TO {}]".format(cloud),
        url=rest_url,
    )
    return products


def _is_4326(geom):
    proj_geom = get_vector_projection(geom)
    proj_4326 = osr.SpatialReference()
    proj_4326.ImportFromEPSG(4326)
    if proj_geom == proj_4326:
        return True
    else:
        return False


def sent2_query(
    user,
    passwd,
    geojsonfile,
    start_date,
    end_date,
    cloud=100,
    tile_id="None",
    start_row=0,
    producttype=None,
    filename=None,
):
    """
    This function:
    
    Fetches a list of Sentinel-2 products

    Parameters
    -----------

    user : str
        Username for ESA hub. Register at https://scihub.copernicus.eu/dhus/#/home

    passwd : str
        password for the ESA Open Access hub

    geojsonfile : str
        Path to a geojson file containing a polygon of the outline of the area you wish to download.
        See www.geojson.io for a tool to build these.
        If no geojson file is provided, a tile_id is required. In that case, aoi_path should point
        to the root directory for the processing run in which all subdirectories will be created.

    start_date : str
        Date of beginning of search in the format YYYY-MM-DDThh:mm:ssZ (ISO standard)

    end_date : str
        Date of end of search in the format yyyy-mm-ddThh:mm:ssZ
        See https://www.w3.org/TR/NOTE-datetime, or use check_for_s2_data_by_date

    cloud : int, optional
        The maximum cloud clover percentage (as calculated by Copernicus) to download. Defaults to 100%

    tile_id : str
        Sentinel-2 granule ID - only required in no geojson file is given and tile-based processing
        is selected. Default: 'None' - do the search by geojson extent.

    start_row : int
        integer of the start row of the query results, can be 0,100,200,... if more than 100 results are returned

    producttype : str
        string describing the product type, e.g. 'S2MSI2A' or 'S2MSI1C'

    filename : str
        file name pattern to be used in the query

    Returns
    -------
    products : dict
        A dictionary of Sentinel-2 granule products that are touched by your AOI polygon, keyed by product ID.
        Returns both level 1 and level 2 data.

    Notes
    -----
    If you get a 'request too long' error, it is likely that your polygon is too complex. The following functions download by granule; there is no need to have a precise polygon at this stage.

    """
	# The following fix is needed on Windows to avoid errors because of long file names.
	# But it causes a directory error on the Linux HPC because it resolves the user home directory wrongly.
    if sys.platform.startswith("win"):
        temp_dir_platform_specific = os.path.expanduser('~')
    else:
        temp_dir_platform_specific = os.path.split(geojson_file)[0]

    with TemporaryDirectory(dir=temp_dir_platform_specific) as td:
        # Preprocessing dates
        start_date = _date_to_timestamp(start_date)
        end_date = _date_to_timestamp(end_date)
        if tile_id == "None" or tile_id == "":
            # Preprocessing geojson geometry
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
                raise InvalidGeometryFormatException(
                    "Please provide a .json, .geojson or a .shp as geometry."
                )
            return _rest_query(
                user,
                passwd,
                footprint,
                start_date,
                end_date,
                cloud,
                start_row,
                producttype,
                filename,
            )
        else:
            return _tile_api_query(
                user,
                passwd,
                tile_id,
                start_date,
                end_date,
                cloud,
                start_row,
                producttype,
                filename,
            )


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
                "longitude": np.round(lon_south, 4),
            },
            "upperRight": {
                "latitude": np.round(lat_east, 4),
                "longitude": np.round(lon_north, 4),
            },
        },
        "temporalFilter": {"startDate": start_date, "endDate": end_date},
        "maxCloudCover": cloud,
    }
    log.info("Sending Landsat query:\n{}".format(data_request))
    request = Request(
        "GET",
        url=api_root + "search",
        params={"jsonRequest": json.dumps(data_request)},
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
    )
    req_string = session.prepare_request(request)
    req_string.url = req_string.url.replace("+", "").replace(
        "%27", "%22"
    )  # usgs why dont you like real url encoding -_-
    response = session.send(req_string)
    products = response.json()["data"]["results"]
    log.info("Retrieved {} product(s)".format(len(products)))
    log.info("Logging out of USGS")
    session.get(
        url=api_root + "logout",
        params={"jsonRequest": json.dumps({"apiKey": session_key})},
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
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
    token = list(
        input.attrs["value"]
        for input in inputs
        if "name" in input.attrs and input.attrs["name"] == "csrf_token"
    )[0]
    ncforminfo = list(
        input.attrs["value"]
        for input in inputs
        if "name" in input.attrs and input.attrs["name"] == "__ncforminfo"
    )[0]

    dl_session.post(
        "https://ers.cr.usgs.gov/login/",
        data={
            "username": conf["landsat"]["user"],
            "password": conf["landsat"]["pass"],
            "csrf_token": token,
            "__ncforminfo": ncforminfo,
        },
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        },
    )

    # for each product in landsat, do stuff
    for product in products:
        download_landing_url = product["downloadUrl"]
        # BeautifulSoup is a library for finding things in webpages - in this case, every download link
        lp_soup = BeautifulSoup(
            requests.get(download_landing_url).content, "html.parser"
        )
        download_buttons = lp_soup.find_all("input")
        dirty_url = list(
            button.attrs["onclick"]
            for button in download_buttons
            if "STANDARD" in button.attrs["onclick"]
        )[0]
        clean_url = dirty_url.partition("=")[2].strip("\\'")
        log.info("Downloading landsat imagery from {}".format(clean_url))
        out_folder_path = os.path.join(out_dir, product["displayId"])
        os.mkdir(out_folder_path)
        tar_path = out_folder_path + ".tar.gz"
        with open(tar_path, "wb+") as fp:
            image_response = dl_session.get(clean_url)
            fp.write(image_response.content)
            log.info("Item {} downloaded to {}".format(product["displayId"], tar_path))
        log.info("Unzipping {} to {}".format(tar_path, out_folder_path))
        with tarfile.open(tar_path, "r:gz") as tar_ref:
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
    user = conf["landsat"]["user"]
    passwd = conf["landsat"]["pass"]
    api_root = "https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/"
    log.info("Logging into USGS")
    login_post = {"username": user, "password": passwd, "catalogId": "EE"}
    session_key = session.post(
        url=api_root + "login/",
        data=urlencode({"jsonRequest": login_post})
        .replace("+", "")
        .replace("%27", "%22"),
        # Hand-mangling the request for POST. Might remove later.
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
    ).json()["data"]
    return session_key


def check_for_s2_data_by_date(
    aoi_path,
    start_date,
    end_date,
    conf,
    cloud_cover=100,
    tile_id="None",
    verbose=False,
    producttype=None,
    filename=None,
):
    """
    Gets all the products between start_date and end_date. Wraps sent2_query to avoid having passwords and
    long-format timestamps in code.

    Parameters
    ----------
    aoi_path : str
        Path to a geojson file containing a polygon of the outline of the area you wish to download.
        See www.geojson.io for a tool to build these.
        If no geojson file is provided, a tile_id is required. In that case, aoi_path should point
        to the root directory for the processing run in which all subdirectories will be created.

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
        The maximum level of cloud cover in images to be downloaded. Default: 100 (all images returned)

    tile_id : str
        Sentinel-2 granule ID - only required if no geojson file is given and tile-based processing
        is selected. Default: 'None' - no tile-based search but aoi-based search

    verbose : boolean
        If True, log additional text output.

    producttype : str
        Sentinel-2 product type to be used in the query. Default: None

    filename : str
        Sentinel-2 file name pattern to be used in the query. Default: None

    Returns
    -------
    result : dict
        A dictionary of Sentinel 2 products.

    """
    user = conf["sent_2"]["user"]
    password = conf["sent_2"]["pass"]
    start_timestamp = (
        dt.datetime.strptime(start_date, "%Y%m%d").isoformat(timespec="seconds") + "Z"
    )
    end_timestamp = (
        dt.datetime.strptime(end_date, "%Y%m%d").isoformat(timespec="seconds") + "Z"
    )
    rolling_n = 0  # rolling number of search results
    n = 100  # number of individual search results of each query
    if tile_id == "None":
        # TODO: check that a valid GeoJSON file path is provided
        log.info("Sending Sentinel-2 queries for GeoJSON footprint:")
        # I.R.
        # log.info("   footprint: {}".format(footprint))
        log.info("   footprint: {}".format(aoi_path))
        log.info("   start_date: {}".format(start_date))
        log.info("   end_date: {}".format(end_date))
        log.info("   cloud_cover: {}".format(cloud_cover))
        log.info("   product_type: {}".format(producttype))
        log.info("   file_name: {}".format(filename))
    else:
        # TODO: check that a valid Tile ID is provided
        log.info("Sending Sentinel-2 query for Tile ID:")
        log.info("   tile_id: {}".format(tile_id))
        log.info("   start_date: {}".format(start_date))
        log.info("   end_date: {}".format(end_date))
        log.info("   cloud_cover: {}".format(cloud_cover))
        log.info("   product_type: {}".format(producttype))
        log.info("   file_name: {}".format(filename))
    while n == 100:
        result = sent2_query(
            user,
            password,
            aoi_path,
            start_timestamp,
            end_timestamp,
            cloud=cloud_cover,
            tile_id=tile_id,
            start_row=rolling_n,
            producttype=producttype,
            filename=filename,
        )
        try:
            rolling_result
        except NameError:
            rolling_result = result  # initialise on first query
        else:
            rolling_result = {
                **rolling_result,
                **result,
            }  # concatenate the new results with the previous dataframe
            n = len(result)
            rolling_n = rolling_n + n
            if verbose:
                log.info(
                    "This query returned {} new images. The overall list now has {} images.".format(
                        len(result), len(rolling_result)
                    )
                )
        if n == 100:
            if verbose:
                log.info(
                    "Submitting new query starting from row number {}".format(rolling_n)
                )
            pass
    if verbose:
        log.info("Queries returned {} new images in total.".format(len(rolling_result)))
    return rolling_result


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
    filtered_query = {
        key: value
        for (key, value) in query_output.items()
        if get_query_level(value) == "Level-1C"
    }
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
    filtered_query = {
        key: value
        for (key, value) in query_output.items()
        if get_query_level(value) == "Level-2A"
    }
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
    # A L1C and L2A image are related if and only if the following fields match:
    #    Satellite (S2[A|B])
    #    Intake date (FIRST timestamp)
    #    Orbit number (Rxxx)
    #    Granule ID (Txxaaa)
    # So if we succesively partition the query, we should get a set of products with either 1 or
    # 2 entries per granule / timestamp combination
    sorted_query = sorted(query_output.values(), key=get_query_granule)
    granule_groups = {
        str(key): list(group)
        for key, group in itertools.groupby(sorted_query, key=get_query_granule)
    }
    granule_date_groups = {}
    # TODO: generate log info
    # filenames = [f for key, f in itertools.groupby(sorted_query, key=get_query_filename)]
    # log.info("Sorted input query results: {}".format(filenames))

    # Partition as above.
    # We can probably expand this to arbitrary lengths of queries. If you catch me doing this, please restrain me.
    for granule, item_list in granule_groups.items():
        item_list.sort(key=get_query_datatake)
        granule_date_groups.update(
            {
                str(granule) + str(key): list(group)
                for key, group in itertools.groupby(item_list, key=get_query_datatake)
            }
        )

    # On debug inspection, turns out sometimes S2 products get replicated. Lets filter those.
    out_set = {}
    for key, image_set in granule_date_groups.items():
        # if sum(1 for image in image_set if get_query_level(image) == "Level-2A") <= 2:
        # list(filter(lambda x: get_query_level(x) == "Level-2A", image_set)).sort(key=get_query_processing_time)[0].pop()
        if (
            sum(1 for image in image_set if get_query_level(image) == "Level-2A") == 1
            and sum(1 for image in image_set if get_query_level(image) == "Level-1C")
            == 1
        ):
            out_set.update({image["uuid"]: image for image in image_set})

    # Finally, check that there is actually something here.
    if len(out_set) == 0:
        log.warning("No L2A data detected for query. Searching for L1C data instead.")
    # raise NoL2DataAvailableException
    # TODO: generate log info
    # filenames = [f for key, f in itertools.groupby(out_set, key=get_query_filename)]
    # log.info("Sorted output query results: {}".format(filenames))
    return out_set


def filter_unique_l1c_and_l2a_data(df: pd.DataFrame,
                                   log: logging.Logger):
    """
    This function:

    Filters a dataframe from a query result such that it contains only unique Sentinel-2
    datatakes, based on 'beginposition'.
    Retains L2A metadata and only retains L1C metadata if no L2A product for that datatake
    has been found.

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe with query results

    Returns
    -------
    l1c, l2a : Tuple[pd.DataFrame, pd.DataFrame]
        Pandas dataframes containing only unique L1C and L2A datatakes

    """

    products_df = df.sort_values("beginposition")
    #  find those pairs of rows where the beginposition is duplicated
    log.info("printing titles and beginpositions")
    for row in products_df.itertuples():
        log.info(f"{row.title}, {row.beginposition}")
    n = products_df.shape[0]
    rows2drop = []
    for r in range(n - 1):
        r1 = products_df.iloc[r, :]["beginposition"]
        r2 = products_df.iloc[r + 1, :]["beginposition"]
        if r1 == r2:
            if products_df.iloc[r, :]["processinglevel"] == "Level-1C":
                rows2drop.append(r)
            else:
                if products_df.iloc[r + 1, :]["processinglevel"] == "Level-1C":
                    rows2drop.append(r + 1)
                else:
                    log.warning(
                        "Neither of the matching rows in the query result is L1C."
                    )
    if len(rows2drop) > 0:
        log.info("L1C data with matching beginposition to be dropped:")
        for i in list(products_df.iloc[rows2drop, :].index):
            log.info("  {}".format(i))
        products_df = products_df.drop(index=list(products_df.iloc[rows2drop, :].index))
    #  find all L1C data that remain in the dataframe
    n = products_df.shape[0]
    l1c = []
    for r in range(n):
        if products_df.iloc[r, :]["processinglevel"] == "Level-1C":
            l1c.append(r)
    if len(l1c) > 0:
        log.info("L1C data that need atmospheric correction:")
        for i in list(products_df.iloc[l1c, :].index):
            log.info("  {}".format(i))
        l1c_df = products_df[products_df.loc[:, "processinglevel"] == "Level-1C"]
    else:
        l1c_df = []
    l2a = []
    for r in range(n):
        if products_df.iloc[r, :]["processinglevel"] == "Level-2A":
            l2a.append(r)
    if len(l2a) > 0:
        l2a_df = products_df[products_df.loc[:, "processinglevel"] == "Level-2A"]
    else:
        l2a_df = []
    return (l1c_df, l2a_df)


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
    return query_item["beginposition"]


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


def get_query_filename(query_item):
    """
    Gets the filename element of a query

    Parameters
    ----------
    query_item : dict
        An item from a query results dictionary.

    Returns
    -------
    filename : str
        The filename element of that item.

    """
    return query_item["filename"]


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
    satellite, _, intake_date, _, orbit_number, granule, _ = safe_product_id.split("_")
    return satellite, intake_date, orbit_number, granule


def download_s2_data(
    new_data,
    l1_dir,
    l2_dir,
    source="scihub",
    user=None,
    passwd=None,
    try_scihub_on_fail=False,
):
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
        identifier = new_data[image_uuid]["identifier"]
        if "L1C" in identifier:
            out_path = os.path.join(l1_dir, identifier + ".SAFE")
            if check_for_invalid_l1_data(out_path) == 1:
                log.info(
                    "{} imagery already exists, skipping download".format(out_path)
                )
                continue
        elif "L2A" in identifier:
            out_path = os.path.join(l2_dir, identifier + ".SAFE")
            if check_for_invalid_l2_data(out_path) == 1:
                log.info(
                    "{} imagery already exists, skipping download".format(out_path)
                )
                continue
        else:
            log.error("{} is not a Sentinel 2 product".format(identifier))
            raise BadDataSourceExpection
        out_path = os.path.dirname(out_path)
        log.info(
            "Downloading {} from {} to {}".format(
                new_data[image_uuid]["identifier"], source, out_path
            )
        )
        if source == "aws":
            if try_scihub_on_fail:
                download_from_aws_with_rollback(
                    product_id=new_data[image_uuid]["identifier"],
                    folder=out_path,
                    uuid=image_uuid,
                    user=user,
                    passwd=passwd,
                )
            else:
                download_safe_format(
                    product_id=new_data[image_uuid]["identifier"], folder=out_path
                )
        # elif source == 'google':
        #    download_from_google_cloud([new_data[image_uuid]['identifier']], out_folder=out_path)
        elif source == "scihub":
            e = download_from_scihub(image_uuid, out_path, user, passwd)
            if e == 1:
                log.warning(
                    "Something went wrong in the download from Copernicus SciHub."
                )
        else:
            log.error("Invalid data source; valid values are 'aws' and 'scihub'")
            raise BadDataSourceExpection
        """
        elif source == 'scihub_lta':
            # SciHub download script for the long-term archive with 80 tries with 3 minute intervals
            cmd = dhus_get_path + ' -d https://scihub.copernicus.eu/dhus -u ' + user + ' -p ' + passwd + \
                  ' -F \'identifier:' + str(image_uuid) + '\' ' + \
                  '-o product -N 5 -w 5 -W 80 -O ' + out_path
            log.info('Running cmd: {}'.format(cmd))
            os.system(cmd)

            dhus_args = [
                         dhus_get_path,
                         "-d",
                         "https://scihub.copernicus.eu/dhus",
                         "-u",
                         user,
                         "-p",
                         passwd,
                         "-F",
                         "\'identifier:" + str(image_uuid) + "\'",
                         "-o",
                         "product",
                         "-N",
                         "5",
                         "-w",
                         "5",
                         "-W",
                         "80",
                         "-O",
                         out_path
                        ]
            log.warning("running cmd with args: {}".format(dhus_args))
            dhus_get_proc = subprocess.Popen(dhus_args, stdout=subprocess.PIPE)
            for line in dhus_get_proc.stdout:
                log.warning("dhusget.sh: >{}".format(line))
            dhus_get_proc.wait()
            log.warning("return code = {}".format(dhus_get_proc.returncode))
        """


def download_s2_data_from_df(
    new_data,
    l1_dir,
    l2_dir,
    source="scihub",
    user=None,
    passwd=None,
    try_scihub_on_fail=False,
):
    """
    Downloads S2 imagery from AWS, google_cloud or scihub. new_data is a dict from Sentinel_2.

    Parameters
    ----------
    new_data : pandas dataframe
        A query dataframe containing the products you want to download
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
    for index, image_uuid in new_data.iterrows():
        identifier = image_uuid["identifier"]
        log.info("  {}   {}".format(index, identifier))
        if "L1C" in identifier:
            out_path = os.path.join(l1_dir, identifier + ".SAFE")
            if check_for_invalid_l1_data(out_path) == 1:
                log.info(
                    "{} imagery already exists, skipping download".format(out_path)
                )
                continue
        elif "L2A" in identifier:
            out_path = os.path.join(l2_dir, identifier + ".SAFE")
            if check_for_invalid_l2_data(out_path) == 1:
                log.info(
                    "{} imagery already exists, skipping download".format(out_path)
                )
                continue
        else:
            log.error("{} is not a Sentinel 2 product".format(identifier))
            raise BadDataSourceExpection
        out_path = os.path.dirname(out_path)
        log.info("Downloading {} from {} to {}".format(identifier, source, out_path))
        if source == "aws":
            if try_scihub_on_fail:
                download_from_aws_with_rollback(
                    product_id=identifier,
                    folder=out_path,
                    uuid=image_uuid,
                    user=user,
                    passwd=passwd,
                )
            else:
                download_safe_format(product_id=identifier, folder=out_path)
        # elif source == 'google':
        #    download_from_google_cloud([identifier], out_folder=out_path)
        elif source == "scihub":
            e = download_from_scihub(index, out_path, user, passwd)
            if e == 1:
                log.warning(
                    "Something went wrong in the download from Copernicus SciHub."
                )

        else:
            log.error("Invalid data source; valid values are 'aws' and 'scihub'")
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
        log.warning(
            "Could not find all corresponding products - please check folder after download"
        )
    download_s2_data(
        to_download,
        l1_dir,
        l2_dir,
        user=conf["sent_2"]["user"],
        passwd=conf["sent_2"]["pass"],
    )


def query_for_corresponding_image(prod, conf):
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
    user = conf["sent_2"]["user"]
    passwd = conf["sent_2"]["pass"]
    api = SentinelAPI(user, passwd, timeout=600)
    # These from https://sentinelsat.readthedocs.io/en/stable/api.html#search-sentinel-2-by-tile
    query_kwargs = {
        "platformname": "Sentinel-2",
        "date": (date - dt.timedelta(days=1), date + dt.timedelta(days=1)),
        "tileid": tile,
        "processinglevel": level,
    }
    out = api.query(**query_kwargs)
    return out


def download_from_aws_with_rollback(product_id: str,
                                    folder: str,
                                    uuid: str,
                                    user: str,
                                    passwd: str) -> None:
    """
    Attempts to download a single product from AWS using product_id; if not found, rolls back to Scihub using the UUID.

    Parameters
    ----------
    product_id : str
        The product ID (`L2A...`)
    folder : str
        The folder to download the .SAFE file to.
    uuid : str
        The product UUID (4dfB4-432df....)
    user : str
        Scihub username
    passwd : str
        Scihub password
    
    Returns
    -------
    None

    """
    log = logging.getLogger(__file__)
    try:
        download_safe_format(product_id=product_id, folder=folder)
    except ClientError:
        log.warning(
            "Something wrong with AWS for products id {}; rolling back to Scihub using uuid {}".format(
                product_id, uuid
            )
        )
        e = download_from_scihub(uuid, folder, user, passwd)
        if e == 1:
            log.warning("Something went wrong in the download from Copernicus SciHub.")


def download_from_scihub(product_uuid: str, out_folder: str, user: str, passwd: str) -> None:
    """
    Downloads and unzips product_uuid from scihub.

    Parameters
    ----------
    product_uuid : str
        The product UUID (e.g. 4dfB4-432df....)
    out_folder : str
        The folder to save the .SAFE file to
    user : str
        Scihub username
    passwd : str
        Scihub password

    Returns
    -------
    0 : No error
    1 : HTTP Error in server response

    Notes
    -----
    If interrupted mid-download, there will be a .incomplete file in the download folder. You might need to remove
    this for further processing.
    Copernicus Open Access Hub no longer stores all products online for immediate retrieval.
    Offline products can be requested from the Long Term Archive (LTA) and should become
    available within 24 hours. Copernicus Open Access Hub's quota currently permits users
    to request an offline product every 30 minutes.
    A product's availability can be checked with a regular OData query by evaluating the
    Online property value or by using the is_online() convenience method.
    When trying to download an offline product with download() it will trigger its retrieval
    from the LTA.
    Given a list of offline and online products, download_all() will download online products,
    while concurrently triggering the retrieval of offline products from the LTA.
    Offline products that become online while downloading will be added to the download queue.
    download_all() terminates when the download queue is empty, even if not all products were
    retrieved from the LTA. We suggest repeatedly calling download_all() to download all products,
    either manually or using a third-party library, e.g. tenacity.

    Source: https://sentinelsat.readthedocs.io/en/latest/api_overview.html

    """
    api = SentinelAPI(user, passwd, timeout=600)
    api.api_url = api_url
    # log.info("Downloading {} from scihub".format(product_uuid))
    is_online = api.is_online(product_uuid)
    if is_online:
        log.info("Product {} is online. Starting download.".format(product_uuid))
        # I.R. START Try/Except test added to skip download and so stop blocking when a file won't download after multiple retries
        try:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(601)
            )
            def download(*args, **kwargs):
                log.info("I.R. Tenacity will retry download up to 5 times.")
                return api.download(*args, **kwargs)

            # I.R. To use Tenacity call download- instead of api.download
            # prod = download(product_uuid, out_folder, max_attempts=20, checksum=True, n_concurrent_dl=2, lta_retry_delay=600)
            prod = download(product_uuid, out_folder)
            # prod = api.download(product_uuid, out_folder)
            if not prod:
                log.error(
                    "Product {} not found. Please check manually on the Copernicus Open Data Hub.".format(
                        product_uuid
                    )
                )
                return 1
        except:
            log.info("\n-------------------------------------------------------------")
            log.warning(
                "I.R. Exception Triggered: Download Failed despite multiple retries - skipping file: {}".format(
                    product_uuid
                )
            )
            log.info("-------------------------------------------------------------\n")
            return 1
        # I.R. END
    else:
        log.info(
            "Product {} is not online. Triggering retrieval from long-term archive.".format(
                product_uuid
            )
        )
        log.info(
            "Remember: 'Patience is bitter, but its fruit is sweet.' (Jean-Jacques Rousseau)"
        )
        try:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(20), wait=tenacity.wait_fixed(601)
            )
            def download_all(*args, **kwargs):
                return api.download_all(*args, **kwargs)

            # I.R. To use Tenacity call download_all - instead of api.download_all
            downloaded, triggered, failed = api.download_all(
                [product_uuid],
                out_folder,
                max_attempts=20,
                checksum=True,
                n_concurrent_dl=2,
                lta_retry_delay=600,
            )
            prod = downloaded[product_uuid]
            if len(downloaded) > 0:
                log.info("Downloaded: {}".format(prod))
            if len(triggered) > 0:
                log.info("Triggered: {}".format(triggered))
            if len(failed) > 0:
                log.warning("Failed: {}".format(failed))
        except:
            e = sys.exc_info()[0]
            log.info("\n-------------------------------------------------------------")
            log.warning("Server Error: {}".format(e))
            log.info("-------------------------------------------------------------\n")
            return 1
    zip_path = os.path.join(out_folder, prod["title"] + ".zip")
    log.info("Unzipping {} to {}".format(zip_path, out_folder))
    zip_ref = zipfile.ZipFile(zip_path, "r")
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
        if (
            check_for_invalid_l1_data(os.path.join(out_folder, safe_id))
            and not redownload
        ):
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
        s2_object.name.replace(os.path.dirname(object_prefix.rstrip("/")), "").strip(
            "/"
        ),
    )
    os.makedirs(os.path.dirname(object_out_path), exist_ok=True)
    log.info("Downloading from {} to {}".format(s2_object, object_out_path))
    with open(object_out_path, "w+b") as f:
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
    with open(path_to_api, "r") as api_file:
        return api_file.read()


def get_planet_product_path(planet_dir, product):
    """
    :meta private:
    Returns the path to a Planet product within a Planet directory
    """
    planet_folder = os.path.dirname(planet_dir)
    product_file = glob.glob(planet_folder + "*" + product)
    return os.path.join(planet_dir, product_file)


def download_planet_image_on_day(
    aoi_path,
    date,
    out_path,
    api_key,
    item_type="PSScene4Band",
    search_name="auto",
    asset_type="analytic",
    threads=5,
):
    """
    :meta private:
    Queries and downloads all images on the date in the aoi given
    """
    log = logging.getLogger(__name__)
    start_time = date + "T00:00:00.000Z"
    end_time = date + "T23:59:59.000Z"
    try:
        planet_query(
            aoi_path,
            start_time,
            end_time,
            out_path,
            api_key,
            item_type,
            search_name,
            asset_type,
            threads,
        )
    except IndexError:
        log.warning("IndexError exception; likely no imagery available for chosen date")


def planet_query(
    aoi_path,
    start_date,
    end_date,
    out_path,
    api_key,
    item_type="PSScene4Band",
    search_name="auto",
    asset_type="analytic",
    threads=5,
):
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
    aoi = feature["geometry"]
    session = requests.Session()
    session.auth = (api_key, "")
    search_request = build_search_request(
        aoi, start_date, end_date, item_type, search_name
    )
    search_result = do_quick_search(session, search_request)

    thread_pool = Pool(threads)
    threaded_dl = lambda item: activate_and_dl_planet_item(
        session, item, asset_type, out_path
    )
    thread_pool.map(threaded_dl, search_result)


def build_search_request(aoi, start_date, end_date, item_type, search_name):
    """
    :meta private:
    Builds a search request for the planet API
    """
    date_filter = planet_api.filters.date_range(
        "acquired", gte=start_date, lte=end_date
    )
    aoi_filter = planet_api.filters.geom_filter(aoi)
    query = planet_api.filters.and_filter(date_filter, aoi_filter)
    search_request = planet_api.filters.build_search_request(query, [item_type])
    search_request.update({"name": search_name})
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
    search_id = search_response.json()["id"]
    if search_response.json()["_links"].get("_next_url"):
        return get_paginated_items(session)
    else:
        search_url = "https://api-planet.com/data/v1/searches/{}/results".format(
            search_id
        )
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
    retry=tenacity.retry_if_exception_type(TooManyRequests),
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
    item_url = (
        "https://api.planet.com/data/v1/"
        + "item-types/{}/items/{}/assets/".format(item_type, item_id)
    )
    item_response = session.get(item_url)
    log.info("Activating " + item_id)
    activate_response = session.post(
        item_response.json()[asset_type]["_links"]["activate"]
    )
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
    with open(item_fp, "wb+") as fp:
        image_response = session.get(dl_link)
        if image_response.status_code == 429:
            raise TooManyRequests
        fp.write(
            image_response.content
        )  # Don't like this; it might store the image twice. Check.
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
    with open(aoi_path, "r") as aoi_fp:
        aoi_dict = json.load(aoi_fp)
        if aoi_dict["type"] == "FeatureCollection":
            aoi_dict = aoi_dict["features"][0]
        return aoi_dict


def fetch_s2_nodata_percentage(api, uuid):
    # No data pixel percentage values for L2A products can be found at:
    # https://apihub.copernicus.eu/apihub/odata/v1/Products('d6cf01d2-3243-4681-8f52-cae50b08e1a2')/Attributes('No%20data%20pixel%20percentage')
    url = (
        api.api_url
        + "odata/v1/Products('{}')/Attributes('No data pixel percentage')".format(uuid)
    )
    response = api.session.get(url)
    qi_info = {}
    log.info(response.content)
    response_str = str(response.content).split("Value>")[1].split("</d")[0]
    qi_info["No_data_percentage"] = float(response_str)
    return qi_info


def get_nodata_percentage(user, passwd, products):
    for uuid, metadata in products.items():
        log.info("Querying metadata for {}: {}".format(uuid, metadata["title"]))
        api = SentinelAPI(user, passwd)
        qi_info = fetch_s2_nodata_percentage(api, uuid)
        if len(qi_info.items()) > 0:
            products[uuid]["No_data_percentage"] = qi_info["No_data_percentage"]
        else:
            log.error("Error - no metadata found for {}".format(metadata["title"]))
    return products
