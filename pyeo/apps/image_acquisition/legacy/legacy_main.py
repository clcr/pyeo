# Legacy API version script - Only for reference use. Copernicus Scihub API is deprecated!
import asyncio
import logging
import os
import time
from datetime import datetime

import pandas as pd
import toml
from sentinelsat import SentinelAPI

import product_processing
from image_acquisition import (
    download_from_scihub,
    download_queued_imagery,
    download_queue,
    activate_queued_imagery,
)

CONFIG = toml.load(os.path.abspath("config/image_acquisition_config.toml"))

API_URL = "https://scihub.copernicus.eu/dhus/"
REST_URL = "https://apihub.copernicus.eu/apihub/search"

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    filename=f"pyeo_run_{datetime.now().strftime('%d-%b-%Y(%H:%M:%S.%f)')}.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("pyeo")


async def main() -> None:
    user = CONFIG["authentication"]["username"]
    password = CONFIG["authentication"]["password"]

    api = SentinelAPI(user, password, timeout=600)
    api.api_url = API_URL

    available_imagery = product_processing.get_available_s2_imagery(
        use_config_roi=False
    )
    available_imagery = product_processing.filter_valid_s2_products(
        available_imagery=available_imagery
    )

    pd.set_option("max_colwidth", 100)

    l1c_imagery, l2a_imagery = product_processing.filter_preprocessing_level(
        filtered_products=available_imagery
    )
    l1c_imagery = (
        product_processing.stratify_by_relative_orbit_number(
            filtered_products=available_imagery
        )[["uuid", "title"]]
        .drop_duplicates(subset="uuid")
        .reset_index(drop=True)
    )

    l2a_imagery = (
        product_processing.stratify_by_relative_orbit_number(
            filtered_products=available_imagery
        )[["uuid", "title"]]
        .drop_duplicates(subset="uuid")
        .reset_index(drop=True)
    )

    available_imagery = pd.concat([l1c_imagery, l2a_imagery])

    queue_counter = 0
    n_downloaded = 0
    n_images_available = len(set(available_imagery["uuid"]))
    LOGGER.info(f"Duplicates removed, {n_images_available} will be downloaded.")
    while queue_counter <= 20:
        for product_uuid, product_title in available_imagery.itertuples(index=False):
            if product_processing.check_file_exists(
                write_path=CONFIG["directories"]["l2a_safes"],
                product_title=product_title,
            ):
                continue
            else:
                await download_from_scihub(
                    api=api,
                    product_uuid=product_uuid,
                    write_path=CONFIG["directories"]["l2a_safes"],
                )
                queue_counter = len(download_queue)
            if queue_counter == 20 or (n_images_available - n_downloaded) < 20:
                await activate_queued_imagery(api=api)
                LOGGER.info("Sleeping for 45 minutes, awaiting imagery coming online.")
                time.sleep(2700)
                LOGGER.info(f"Attempting to download product: {product_uuid}")
                if product_processing.check_file_exists(
                    write_path=CONFIG["directories"]["l2a_safes"],
                    product_title=product_title,
                ):
                    continue
                else:
                    n_downloaded = await download_queued_imagery(
                        api=api, write_path=CONFIG["directories"]["l2a_safes"]
                    )
                    queue_counter = 0
                    LOGGER.info(
                        f"Images downloaded: {n_downloaded}/{n_images_available}"
                    )

        if n_downloaded == n_images_available:
            break

    for product_title in set(available_imagery["title"]):
        product_processing.unzip_downloaded_products(
            write_path=CONFIG["directories"]["l2a_safes"], product=product_title
        )

    return


if __name__ == "__main__":
    asyncio.run(main())
