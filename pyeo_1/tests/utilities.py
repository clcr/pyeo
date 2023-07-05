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
pytest utilities.py --log-cli-level DEBUG   (runs all non-slow tests with log output printed to stdout)
pytest utilities.py --log-cli-level DEBUG -k composite   (runs all non-slow tests with 'composite' in the function
name)
pytest utilities.py --log-cli-level DEBUG --runslow  (runs all tests)
"""
import os



import configparser
from osgeo import gdal

from pyeo.filesystem_utilities import init_log

gdal.UseExceptions()


def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_log("test_log.log")


def load_test_conf():
    test_conf = configparser.ConfigParser()
    test_conf.read("test_data/test_creds.ini.ignoreme")
    return test_conf



