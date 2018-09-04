from sentinelsat.sentinel import SentinelAPI


def sen2_download(products, conf):
    # TODO: Specify download location, file resuming.
    api = SentinelAPI(conf["sen2"]["user"], conf["sen2"]["pass"], 'https://scihub.copernicus.eu/dhus')
    api.download_all(products, conf["data"]["out_folder"])