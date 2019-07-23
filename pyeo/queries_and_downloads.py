"""
Functions for querying, filtering and downloading data.
"""

import datetime as dt
import glob
import itertools
import json
import logging
import os
import shutil
import zipfile
from multiprocessing.dummy import Pool

import requests
import tenacity
from botocore.exceptions import ClientError
from google.cloud import storage
from sentinelhub import download_safe_format
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

from pyeo.sen2_funcs import get_sen_2_image_tile
from pyeo.filesystem_utilities import check_for_invalid_l2_data, check_for_invalid_l1_data
from pyeo.exceptions import NoL2DataAvailableException, BadDataSourceExpection, TooManyRequests


def sent2_query(user, passwd, geojsonfile, start_date, end_date, cloud=50):
    """


    From Geospatial Learn by Ciaran Robb, embedded here for portability.

    Produces a dict of sentinel-2 IDs and

    Notes
    -----------


    Parameters
    -----------

    user : string
           username for esa hub

    passwd : string
             password for hub

    geojsonfile : string
                  AOI polygon of interest in EPSG 4326

    start_date : string
                 date of beginning of search

    end_date : string
               date of end of search

    cloud : string (optional)
            include a cloud filter in the search

    product : string (optional)
            Product type for Sentinel 2. Valid values are S2MSI1C and S2MS2Ap


    """
    ##set up your copernicus username and password details, and copernicus download site... BE CAREFUL if you share this script with others though!
    log = logging.getLogger(__name__)
    api = SentinelAPI(user, passwd)
    footprint = geojson_to_wkt(read_geojson(geojsonfile))
    log.info("Sending query:\nfootprint: {}\nstart_date: {}\nend_date: {}\n cloud_cover: {} ".format(
        footprint, start_date, end_date, cloud))
    products = api.query(footprint,
                         date=(start_date, end_date), platformname="Sentinel-2",
                         cloudcoverpercentage="[0 TO {}]".format(cloud))
    return products


def check_for_s2_data_by_date(aoi_path, start_date, end_date, conf, cloud_cover=50):
    log = logging.getLogger(__name__)
    log.info("Querying for imagery between {} and {} for aoi {}".format(start_date, end_date, aoi_path))
    user = conf['sent_2']['user']
    password = conf['sent_2']['pass']
    start_timestamp = dt.datetime.strptime(start_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    end_timestamp = dt.datetime.strptime(end_date, '%Y%m%d').isoformat(timespec='seconds')+'Z'
    result = sent2_query(user, password, aoi_path, start_timestamp, end_timestamp, cloud=cloud_cover)
    log.info("Search returned {} images".format(len(result)))
    return result


def filter_non_matching_s2_data(query_output):
    """Removes any L2/L1 product that does not have a corresponding L1/L2 data"""
    # Here be algorithms
    # A L1 and L2 image are related iff the following fields match:
    #    Satellite (S2[A|B])
    #    Intake date (FIRST timestamp)
    #    Orbit number (Rxxx)
    #    Granule ID (Txxaaa)
    # So if we succeviely partition the query, we should get a set of products with either 1 or
    # 2 entries per granule / timestamp combination
    log = logging.getLogger(__name__)
    sorted_query = sorted(query_output.values(), key=get_query_granule)
    granule_groups = {str(key): list(group) for key, group in itertools.groupby(sorted_query, key=get_query_granule)}
    granule_date_groups = {}

    # Partition as above.
    # We can probably expand this to arbitrary lengths of queries. If you catch me doing this, please restrain me.
    for granule, item_list in granule_groups.items():
        item_list.sort(key=get_query_datatake)
        granule_date_groups.update(
            {str(granule)+str(key): list(group) for key, group in itertools.groupby(item_list, key=get_query_datatake)})

    # On debug inspection, turns out sometimes S2 products get replicated. Lets filter those.
    out_set = {}
    for key, image_set in granule_date_groups.items():
        #if sum(1 for image in image_set if get_query_level(image) == "Level-2A") <= 2:
            #list(filter(lambda x: get_query_level(x) == "Level-2A", image_set)).sort(key=get_query_processing_time)[0].pop()
        if (sum(1 for image in image_set if get_query_level(image) == "Level-2A") == 1
        and sum(1 for image in image_set if get_query_level(image) == "Level-1C") == 1):
            out_set.update({image["uuid"]: image for image in image_set})

    #Now rebuild into uuid:data_object format
    if len(out_set) == 0:
        log.error("No L2 data detected for query. Please remove the --download_l2_data flag or request more recent images.")
        raise NoL2DataAvailableException
    return out_set


def get_query_datatake(query_item):
    return query_item['beginposition']


def get_query_granule(query_item):
    return query_item["title"].split("_")[5]


def get_query_processing_time(query_item):
    ingestion_string = query_item["title"].split("_")[6]
    return dt.datetime.strptime(ingestion_string, "%Y%m%dT%H%M%S")


def get_query_level(query_item):
    return query_item["processinglevel"]


def get_granule_identifiers(safe_product_id):
    """Returns the parts of a S2 name that uniquely identify that granulate at a moment in time"""
    satellite, _, intake_date, _, orbit_number, granule, _ = safe_product_id.split('_')
    return satellite, intake_date, orbit_number, granule


def download_s2_data(new_data, l1_dir, l2_dir, source='scihub', user=None, passwd=None, try_scihub_on_fail=False):
    """Downloads S2 imagery from AWS, google_cloud or scihub. new_data is a dict from Sentinel_2."""
    log = logging.getLogger(__name__)

    for image_uuid in new_data:

        if 'L1C' in new_data[image_uuid]['identifier']:
            out_path = l1_dir
            if check_for_invalid_l1_data(out_path) == 1:
                log.info("L1 imagery exists, skipping download")
                continue
        elif 'L2A' in new_data[image_uuid]['identifier']:
            out_path = l2_dir
            if check_for_invalid_l2_data(out_path) == 1:
                log.info("L2 imagery exists, skipping download")
                continue
        else:
            log.error("{} is not a Sentinel 2 product")
            raise BadDataSourceExpection

        log.info("Downloading {} from {} to {}".format(new_data[image_uuid]['identifier'], source, out_path))
        if source=='aws':
            if try_scihub_on_fail:
                download_from_aws_with_rollback(product_id=new_data[image_uuid]['identifier'], folder=out_path,
                                                uuid=image_uuid, user=user, passwd=passwd)
            else:
                download_safe_format(product_id=new_data[image_uuid]['identifier'], folder=out_path)
        elif source=='google':
            download_from_google_cloud([new_data[image_uuid]['identifier']], out_folder=out_path)
        elif source=="scihub":
            download_from_scihub(image_uuid, out_path, user, passwd)
        else:
            log.error("Invalid data source; valid values are 'aws', 'google' and 'scihub'")
            raise BadDataSourceExpection


def download_from_aws_with_rollback(product_id, folder, uuid, user, passwd):
    """Attempts to download product from AWS using product_id; if not found, rolls back to Scihub using uuid"""
    log = logging.getLogger(__file__)
    try:
        download_safe_format(product_id=product_id, folder=folder)
    except ClientError:
        log.warning("Something wrong with AWS for products id {}; rolling back to Scihub using uuid {}".format(product_id, uuid))
        download_from_scihub(uuid, folder, user, passwd)


def download_from_scihub(product_uuid, out_folder, user, passwd):
    """Downloads and unzips product_uuid from scihub"""
    log = logging.getLogger(__name__)
    api = SentinelAPI(user, passwd)
    log.info("Downloading {} from scihub".format(product_uuid))
    prod = api.download(product_uuid, out_folder)
    if not prod:
        log.error("{} failed to download".format(product_uuid))
    zip_path = os.path.join(out_folder, prod['title']+".zip")
    log.info("Unzipping {} to {}".format(zip_path, out_folder))
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(out_folder)
    zip_ref.close()
    log.info("Removing {}".format(zip_path))
    os.remove(zip_path)


def download_from_google_cloud(product_ids, out_folder, redownload = False):
    """Passed a list of S2 product ids , downloads them into out_for"""
    log = logging.getLogger(__name__)
    log.info("Downloading following products from Google Cloud:".format(product_ids))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("gcp-public-data-sentinel-2")
    for safe_id in product_ids:
        if not safe_id.endswith(".SAFE"):
            safe_id = safe_id+".SAFE"
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
    """Returns an API key from a single-line text file containing that API"""
    with open(path_to_api, 'r') as api_file:
        return api_file.read()


def get_planet_product_path(planet_dir, product):
    """Returns the path to a Planet product within a Planet directory"""
    planet_folder = os.path.dirname(planet_dir)
    product_file = glob.glob(planet_folder + '*' + product)
    return os.path.join(planet_dir, product_file)


def download_planet_image_on_day(aoi_path, date, out_path, api_key, item_type="PSScene4Band", search_name="auto",
                 asset_type="analytic", threads=5):
    """Queries and downloads all images on the date in the aoi given"""
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
    """Builds a search request for the planet API"""
    date_filter = planet_api.filters.date_range("acquired", gte=start_date, lte=end_date)
    aoi_filter = planet_api.filters.geom_filter(aoi)
    query = planet_api.filters.and_filter(date_filter, aoi_filter)
    search_request = planet_api.filters.build_search_request(query, [item_type])
    search_request.update({'name': search_name})
    return search_request


def do_quick_search(session, search_request):
    """Tries the quick search; returns a dict of features"""
    search_url = "https://api.planet.com/data/v1/quick-search"
    search_request.pop("name")
    print("Sending quick search")
    search_result = session.post(search_url, json=search_request)
    if search_result.status_code >= 400:
        raise requests.ConnectionError
    return search_result.json()["features"]


def do_saved_search(session, search_request):
    """Does a saved search; this doesn't seem to work yet."""
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
    """Let's leave this out for now."""
    raise Exception("pagination not handled yet")


@tenacity.retry(
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_delay(10000),
    retry=tenacity.retry_if_exception_type(TooManyRequests)
)
def activate_and_dl_planet_item(session, item, asset_type, file_path):
    """Activates and downloads a single planet item"""
    log = logging.getLogger(__name__)
    #  TODO: Implement more robust error handling here (not just 429)
    item_id = item["id"]
    item_type = item["properties"]["item_type"]
    item_url = "https://api.planet.com/data/v1/"+ \
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
        fp.write(image_response.content)    # Don't like this; it might store the image twice. Check.
        log.info("Item {} download complete".format(item_id))


def read_aoi(aoi_path):
    """Opens the geojson file for the aoi. If FeatureCollection, return the first feature."""
    with open(aoi_path, 'r') as aoi_fp:
        aoi_dict = json.load(aoi_fp)
        if aoi_dict["type"] == "FeatureCollection":
            aoi_dict = aoi_dict["features"][0]
        return aoi_dict