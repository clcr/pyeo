import logging
import time
import traceback
from datetime import datetime

from sentinelsat import SentinelAPI
from sentinelsat.exceptions import ServerError, LTAError

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename=f"pyeo_run_{datetime.now().strftime('%d-%b-%Y(%H:%M:%S.%f)')}.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("pyeo")

global download_queue
download_queue = []


async def download_from_scihub(
    api: SentinelAPI, product_uuid: str, write_path: str
) -> None:
    """
    This function downloads a given product from Scihub using the SentinelSat library.

    Parameters
    ----------
    api: SentinelAPI instance
    product_uuid: UUID of the product to download
    write_path: Path the downloaded product is to be stored

    Returns
    -------

    """
    product_info = {}
    try:
        product_info = api.get_product_odata(product_uuid)
    except ServerError:
        LOGGER.info(f"ServerError when requesting product info for {product_uuid}")

    if not product_info and product_info["Online"]:
        try:
            api.download_all([product_uuid], directory_path=write_path)
        except ServerError:
            LOGGER.info(f"ServerError when downloading product: {product_uuid}")
    else:
        try:
            download_queue.append(product_uuid)
            LOGGER.info(f"Product {product_uuid} is not online. Adding to queue.")
        except LTAError:
            LOGGER.info(f"LTAError when product: {product_uuid} was requested.")

    return


async def activate_queued_imagery(api: SentinelAPI) -> None:
    """
    This function requests a given product UUID from the product queue to be activated
    for download from the Scihub Long Term Archive.

    Parameters
    ----------
    api: SentinelAPI instance

    Returns
    -------

    """
    for product in download_queue:
        LOGGER.info(f"Activating product: {product}.")
        try:
            api.trigger_offline_retrieval(product)
            LOGGER.info(f"Product {product} is not online. Retrieving from LTA.")
        except (ServerError, LTAError):
            LOGGER.info(f"Error occurred when activating product: {product}.")
            LOGGER.info(f"Stack trace: {traceback.format_exc()}")

    return


async def download_queued_imagery(api: SentinelAPI, write_path: str) -> int:
    """
    This function downloads the product UUIDs from the global download queue, once the
    imagery has become available for download.

    Parameters
    ----------
    api: SentinelAPI instance
    write_path: Path to write the imagery to

    Returns
    -------
    n_processed: Number of images processed

    """
    retry_counter = 0
    max_retry = 3
    n_processed = 0
    for product_uuid in download_queue:
        if api.is_online(product_uuid):
            try:
                api.download_all([product_uuid], directory_path=write_path)
                download_queue.remove(product_uuid)
                n_processed += 1
                LOGGER.info(f"Downloaded product: {product_uuid}")
            except ServerError:
                LOGGER.info(f"ServerError when downloading product: {product_uuid}")
                retry_counter += 1
                if retry_counter == max_retry:
                    retry_counter = 0
                    download_queue.remove(product_uuid)
                    n_processed += 1
                    continue
        else:
            time.sleep(300)
            LOGGER.info("Product is not online, sleeping for another 5 minutes")

    return n_processed
