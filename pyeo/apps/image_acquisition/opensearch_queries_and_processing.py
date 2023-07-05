import logging
import os.path
import zipfile
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import toml
from tqdm import tqdm

CONFIG = toml.load("/data/clcr/shared/IMPRESS/matt/pyeo/pyeo_production/pyeo_production/pyeo/apps/image_acquisition/config/image_acquisition_config.toml")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename=f"pyeo_run.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("pyeo")
ROI_GEOMETRY = gpd.read_file(CONFIG["sentinel2_properties"]["tiles_to_process"])
ROI_GEOMETRY["centroid"] = ROI_GEOMETRY.representative_point()
TILE_DICT = {
    ROI_GEOMETRY["name"][i]: str(ROI_GEOMETRY["centroid"][i])
    for i in range(ROI_GEOMETRY.shape[0])
}

API_ROOT = "http://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
DOWNLOAD_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
REFRESH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

USERNAME = CONFIG["authentication"]["username"]
PASSWORD = CONFIG["authentication"]["password"]
REFRESH_TOKEN = CONFIG["authentication"]["refresh_token"]

SAFE_DOWNLOAD_PATH = CONFIG["directories"]["l2a_safes"]
MIN_IMAGE_SIZE = CONFIG["sentinel2_properties"]["min_image_size"]


def build_request_string(
    max_cloud_cover: int,
    start_date: str,
    end_date: str,
    area_of_interest: str,
    max_records: int,
) -> str:
    """
    This function builds the API product request string based on given properties and constraints.

    Parameters
    ----------
    max_cloud_cover: Maximum cloud cover to allow in the queried products
    start_date: Starting date of the observations (YYYY-MM-DD format)
    end_date: Ending date of the observations (YYYY-MM-DD format)
    area_of_interest: Area of interest geometry as a string in WKT format
    max_records: Maximum number of products to show per query (queries with very high numbers may not complete in time)

    Returns
    -------
    request_string: API Request String

    """
    cloud_cover_props = f"cloudCover=[0,{max_cloud_cover}]"
    start_date_props = f"startDate={start_date}"
    end_date_props = f"completionDate={end_date}"
    geometry_props = f"geometry={area_of_interest}"
    max_records_props = f"maxRecords={max_records}"

    request_string = f"{API_ROOT}?{cloud_cover_props}&{start_date_props}&{end_date_props}&{geometry_props}&{max_records_props}"
    return request_string


def get_s2_tile_centroids(s2_geometry_path: str) -> gpd.GeoDataFrame:
    sentinel2_tile_geoms = gpd.read_file(s2_geometry_path)

    sentinel2_tile_geoms["centroid"] = sentinel2_tile_geoms.representative_point()

    return sentinel2_tile_geoms


def query_by_polygon(
    max_cloud_cover: int,
    start_date: str,
    end_date: str,
    area_of_interest: str,
    max_records: int,
) -> pd.DataFrame:
    """
    This function returns a DataFrame of available Sentinel-2 imagery from the Copernicus Dataspace API.

    Parameters
    ----------
    max_cloud_cover: Maximum Cloud Cover
    start_date: Start date of the images to query from in (YYYY-MM-DD) format
    end_date: End date of the images to query from in (YYYY-MM-DD) format
    area_of_interest: Region of interest centroid in WKT format
    max_records: Maximum records to return

    Returns
    -------

    """

    request_string = build_request_string(
        max_cloud_cover=max_cloud_cover,
        start_date=start_date,
        end_date=end_date,
        area_of_interest=area_of_interest,
        max_records=max_records,
    )

    response = requests.get(request_string).json()["features"]

    response_dataframe = pd.DataFrame.from_dict(response)
    response_dataframe = pd.DataFrame.from_records(response_dataframe["properties"])
    return response_dataframe


def filter_valid_size_s2_products(response_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function filters out invalid Sentinel-2 products. An invalid Sentinel-2 product is
    defined as a product with size of less than 500MB. This function filters out tiles which do not have enough data,
    e.g. mostly black tiles or otherwise unusable Sentinel-2 data.

    Parameters
    ----------
    response_dataframe: All products returned from the API request

    Returns
    -------

    """

    sizes = [elem["download"]["size"] for elem in response_dataframe["services"]]
    response_dataframe["size"] = sizes

    response_dataframe["size"] = response_dataframe["size"].apply(
        lambda x: float(x) * 1e-6
    )
    response_dataframe = response_dataframe.query("size >= " + str(MIN_IMAGE_SIZE))

    response_dataframe = response_dataframe.reset_index(drop=True)
    return response_dataframe


def stratify_products_by_orbit_number(response_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function stratifies the products by relative orbit number, so that imagery
    from across different orbits per tile is sampled, and hence the imagery is representative.

    Parameters
    ----------
    response_dataframe: Sentinel-2 Product dataframe

    Returns
    -------

    """

    stratified_products = response_dataframe.copy(deep=True)

    uuids = [
        elem["download"]["url"].split("/")[-1]
        for elem in response_dataframe["services"]
    ]
    stratified_products["uuid"] = uuids

    max_image_number = 30
    rel_orbits = np.unique(response_dataframe["relativeOrbitNumber"])

    if len(rel_orbits) > 0:
        if stratified_products.shape[0] > max_image_number / len(rel_orbits):
            uuids = []
            for orb in rel_orbits:
                uuids = (
                    uuids
                    + list(
                        stratified_products[
                            stratified_products["relativeOrbitNumber"] == orb
                        ].sort_values(by=["cloudCover"], ascending=True)["uuid"]
                    )[: int(max_image_number / len(rel_orbits))]
                )
            stratified_products = stratified_products[
                stratified_products["uuid"].isin(uuids)
            ]

    return stratified_products


def get_access_token(refresh: bool = False) -> str:
    """
    This function creates an access token to use during download for verification purposes.

    Parameters
    ----------
    refresh: Refreshes an old access token, Default false - returns new access token

    Returns
    -------

    """
    if refresh:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": "cdse-public",
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(REFRESH_TOKEN_URL, data=payload, headers=headers)
    else:
        payload = {
            "grant_type": "password",
            "username": USERNAME,
            "password": PASSWORD,
            "client_id": "cdse-public",
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(
            REFRESH_TOKEN_URL, data=payload, headers=headers
        ).json()

    return response["access_token"]


def download_product(product_uuid: str, auth_token: str, product_name: str) -> None:
    """
    This function downloads a given Sentinel product, with a given product UUID from the ESA servers.

    Parameters
    ----------
    product_uuid: UUID of the product to download
    auth_token: Authentication bearer token
    product_name: Name of the product

    Returns
    -------

    """

    response = requests.get(
        f"{DOWNLOAD_URL}({product_uuid})/$value",
        headers={"Authorization": f"Bearer {auth_token}"},
        stream=True,
    )

    total_size_in_bytes = int(response.headers.get("Content-Length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(f"{SAFE_DOWNLOAD_PATH}/{product_name}.zip", "wb") as download:
        for data in response.iter_content(block_size):
            download.write(data)
            progress_bar.update(len(data))

    progress_bar.close()

    return


def unzip_downloaded_product(write_directory: str, product_name: str) -> None:
    """
    This function unzips the downloaded SAFE .zip archives to a given path.

    Parameters
    ----------
    write_directory: Path to write the unzipped SAFE directory to.
    product_name: Product name

    Returns
    -------

    """

    LOGGER.info(f"Attempting to unzip image: {product_name}")

    try:
        zip_path = os.path.join(write_directory, product_name + ".zip")
        zip_ref = zipfile.ZipFile(zip_path, "r")
        zip_ref.extractall(write_directory)
        zip_ref.close()
        os.remove(zip_path)
        LOGGER.info(f"Unzipped {zip_path} to {write_directory}")
    except UnboundLocalError:
        LOGGER.info(f"Unzip for {product_name} failed.")

    return


def check_product_exists(write_directory: str, product_name: str) -> bool:
    """
    This function checks whether a product is already downloaded and currently
    exists on the filesystem. If it does, the function returns True, and allows
    for the product download to be skipped within the main image acquisition pipeline.

    Parameters
    ----------
    write_directory: Directory products are downloaded to
    product_name: Name of the product

    Returns
    -------
    Boolean - True if product exists, False otherwise
    """

    if os.path.exists(f"{write_directory}/{product_name}"):
        LOGGER.info(
            f"Product {product_name} at {write_directory} already exists. Skipping."
        )
        return True
    else:
        return False
