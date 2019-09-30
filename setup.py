from distutils.core import setup
import os
import stat

if 'PYEO' not in os.environ:
    this_directory = os.path.abspath(os.path.dirname(__file__))
    os.environ['PYEO'] = this_directory
    os.environ['PATH'] = os.environ['PATH']+r":$PYEO/bin"
    permissions = stat.S_IEXEC | stat.S_IREAD
    os.chmod(this_directory+"/bin/*", permissions)


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
        "requests",
        "setuptools",
        "numpy",
        "scikit-learn",
        "scipy",
        "geojson",
        "sentinelhub",
        "sentinelsat",
        "tenacity",
        "tqdm"
    ],
)
