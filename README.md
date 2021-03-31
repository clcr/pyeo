[![DOI](https://zenodo.org/badge/126246599.svg)](https://zenodo.org/badge/latestdoi/126246599)

# pyeo
Python for Earth Observation

This is designed to provide a set of portable, extensible and modular Python scripts for machine learning in earth observation and GIS,
including downloading, preprocessing, creation of base layers, classification and validation.

Full documentation at https://clcr.github.io/pyeo/build/html/index.html

Example notebooks are at https://github.com/clcr/pyeo_training_materials

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

This presumes a set of training data exists, you have signed up to Scihub and the folders `s2_l1`, `s2_l2`, `preprocessed` and `classified` have been created.

```python
from pyeo import raster_manipulation as ras
from pyeo import queries_and_downloads as dl
from pyeo import classification as cls

# train_model.py
cls.extract_features_to_csv("training_raster.tif",
                            "training_shape.shp",
                            "features.csv")
cls.create_model_from_signatures("features.csv", "model.pkl")


# classify_area.py
username = "scihub_user"
password = "scihub_pass"
conf = {'sent_2':{'user':username, 'pass':password}}
data = dl.check_for_s2_data_by_date("aoi.shp",
                                    "20200101",
                                    "20200201",
                                    "conf")
dl.download_s2_data(data, "s2_l1", "s2_l2", username, password)
ras.preprocess_sen2_images("s2_l2", "preprocessed", "s2_l1")
cls.classify_directory("preprocessed",
                       "model.pkl",
                       "classified",
                       apply_mask=True)
```
(This is a toy script; keeping your username and password in your script is not recommended in the real world).
