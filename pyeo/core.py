try:
    from google.cloud import storage
    from google.cloud.exceptions import ServiceUnavailable
except ModuleNotFoundError:
    print("google-cloud-storage required for Google downloads. Try pip install google-cloud-storage")

try:
    import tenacity
    from planet import api as planet_api
    from multiprocessing.dummy import Pool
except ModuleNotFoundError:
    print("Tenacity, Planet and Multiprocessing are required for Planet data downloading")


