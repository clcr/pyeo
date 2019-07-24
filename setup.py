from distutils.core import setup

setup(
    name='Pyeo',
    version='0.2.0',
    author='John Roberts',
    author_email='jfr10@le.ac.uk',
    packages=['pyeo', 'pyeo.tests'],
    url='http://pypi.python.org/pypi/Pyeo/',
    license='LICENSE',
    description='Modular processing chain from download to ard',
    install_requires=[
        "boto3",
        "botocore",
        "gdal",
        "joblib",
        "matplotlib",
        "pip",
        "pytest",
        "python",
        "requests",
        "scikit-image",
        "setuptools",
        "numpy",
        "sklearn=0.19.2",
        "scipy",
        "geojson",
        "google - cloud",
        "sentinelhub",
        "sentinelsat",
        "tenacity",
        "tqdm"
    ],
)
