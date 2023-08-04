from distutils.core import setup
import os
import stat

setup(
    name="pyeo",
    version="1.0.0",
    author="University of Leicester",
    author_email="hb91@le.ac.uk",
    packages=["pyeo", "pyeo.tests"],
    url="http://pypi.python.org/pypi/Pyeo/",
    license="LICENSE",
    description="Forest alerts from Sentinel-2 images",
    install_requires=[
        "boto3",
        "botocore",
        "gdal",
        "joblib",
        "matplotlib",
        "pip",
        "pytest",
        "requests",
        "setuptools",
        "numpy",
        "scikit-learn",
        "scipy",
        "geojson",
        "sentinelhub",
        "sentinelsat",
        "tenacity",
        "tqdm",
        "pysolar",
    ],
)
