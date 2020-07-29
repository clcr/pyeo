[![DOI](https://zenodo.org/badge/126246599.svg)](https://zenodo.org/badge/latestdoi/126246599)

# pyeo
Python for Earth Observation processing chain

This is designed to provide a set of portable, extensible and modular Python scripts for earth observation and machine learning,
including downloading, preprocessing, creation of base layers and classification.

## Requirements
Package management is performed by Conda: https://docs.conda.io/en/latest/

For downloading, you will need a Scihub account: https://scihub.copernicus.eu/

For Sentinel 2 processing, you may need Sen2Cor installed: http://step.esa.int/main/third-party-plugins-2/sen2cor/

For AWS downloading, you will need credentials set up on your machine.

## To install
To install Pyeo, put the following commands into Bash (Linux), Terminal (Mac) or the **Anaconda Prompt** (Windows)

```bash
git clone https://github.com/clcr/pyeo.git
cd pyeo
conda env create --file environment.yml --name pyeo_env
conda activate pyeo_env
python -m pip install -e .
```
If you want access to the Pyeo command line functions, add the following to your .bashrc

```bash
export PYEO=/path/to/pyeo
export PATH=$PATH:$PYEO/bin
```

If you do not want to edit Pyeo, replace the pip install line with

```bash
python -m pip install . -vv
```

You can test your installation with
`import pyeo.classifier`

## Example script

Once credentials.ini is filled in with your Scihub username and password, the following script will download every sentinel 2 image in Kinangop betweem Christmas and New Year, and preprocess and classify one of them.

```python
import pyeo.queries_and_downloads
import pyeo.classification
import pyeo.filesystem_utilities
import pyeo.raster_manipulation

pyeo.filesystem_utilities.init_log("training_log.log")

import configparser
conf = configparser.ConfigParser()
conf.read("credentials.ini")
conf['sent_2']['user']

query_results = pyeo.queries_and_downloads.check_for_s2_data_by_date(
    "kinangop_rough.json",
    "20181225",
    "20190101",
    conf,
    "30"
)

pyeo.queries_and_downloads.download_s2_data(
    query_results,
    l1_dir = "level_1",
    l2_dir = "level_2",
    source='scihub',
    user=conf["sent_2"]["user"],
    passwd=conf["sent_2"]["pass"]
)

pyeo.raster_manipulation.stack_sentinel_2_bands(
    "level_2/S2A_MSIL2A_20181230T074321_N0211_R092_T36MZE_20181230T100827.SAFE",
    "merged.tif",
    out_resolution=60
)

pyeo.classification.classify_image(
    "merged.tif",
    "my_model.pkl",
    "my_classified_image.tif"
)
```

Full documentation at https://clcr.github.io/pyeo/build/html/index.html
