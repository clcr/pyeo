<p align="center">
  <img src="./pyeo/assets/pyeo_logo.png" alt="Image alt text">
</p>

[![DOI](https://zenodo.org/badge/126246599.svg)](https://zenodo.org/badge/latestdoi/126246599)

# Python for Earth Observation (PyEO)

PyEO is designed to provide a set of portable, extensible and modular Python scripts for machine learning in earth observation and GIS,
including downloading, preprocessing, creation of baseline composites, classification and validation.

Full documentation of the functions provided are available at https://clcr.github.io/pyeo/build/html/index.html

Additionally, users on the cloud-processing platform, [SEPAL](https://sepal.io), can follow the [PyEO sepal user guide](./sepal_guide/SEPAL_User_Guide_PyEO_Forest_Alert_System.pdf): 

Example notebooks are available at:
- SEPAL specific notebooks
  - [Orientation to SEPAL](./notebooks/PyEO_sepal_orientation.ipynb)
  - [Training a PyEO ML Model on SEPAL](./notebooks/PyEO_sepal_model_training.ipynb)
  - [Running PyEO deforestation monitoring on SEPAL](./notebooks/PyEO_sepal_pipeline_training.ipynb)
- PyEO Local Materials
  - https://github.com/clcr/pyeo_training_materials

## Requirements
Python library requirements are categorised by Platform (Operating System - OS). For use in the Cloud Processing platform SEPAL - pyeo is already installed in a virtual environment. <!-- This is in anticipation of pyeo SEPAL-wide venv being created -->
SEPAL is a cloud computing platform for geospatial data which offers remote Linux Instances that are customised for performing geospatial analysis in R or Python. More information can be found here: https://github.com/openforis/sepal <br> 

Package management is performed by Conda, for instructions on how to install Conda, please refer to: https://docs.conda.io/en/latest/.  
*Note: Conda can be installed as part of Anaconda https://www.anaconda.com/*  
<br>

For installation locally on an OS of your choice, see the sections below.  

To install `pyeo`, put the following commands into **Bash** (Linux), **Terminal** (Mac) or the **Anaconda Prompt** (Windows) <br>

### Ubuntu or MacOS
```bash
conda install -c conda-forge git
git clone https://github.com/clcr/pyeo.git
cd pyeo
conda env create --file environment.yml --name pyeo_env
conda activate pyeo_env
python -m pip install -e .
```
If you do not want to edit `pyeo`, replace `python -m pip install -e .` line with

```bash
python -m pip install -vv .
```

### Windows
```bash
conda install -c conda-forge git
git clone https://github.com/clcr/pyeo.git
cd pyeo
conda env create --file environment_windows.yml --name pyeo_env
conda activate pyeo_env
python -m pip install -e .
```

If you do not want to edit `pyeo`, replace `python -m pip install -e .` line with

```bash
python -m pip install -vv .
```
<br>  

#### A Note on `.ini` file encoding on Windows
If the OS that pyeo is running on is Windows, we have noticed that `pyeo_windows.ini` may need to be saved with `ANSI` encoding instead of the usual `UTF-8`. See [this webpage](https://stackoverflow.com/questions/13282189/missingsectionheadererror-file-contains-no-section-headers) for more details.

## Satellite Imagery Providers
From July 2023, Scihub will be deprecated in favour of the Copernicus Data Space Ecosystem (CDSE). In the meantime, if you wish to download from Scihub, you will need a Scihub account: https://scihub.copernicus.eu/

To use the CDSE, you will need a separate account: https://dataspace.copernicus.eu

Once you have created your account, you will need to enter your email address and password into the `credentials_dummy.ini` file in the folder `credentials`, like this:  

```
[sent_2]
user=replace_this_with_your_email_address
pass=replace_this_with_your_password

[dataspace]
user=replace_this_with_your_email_address
pass=replace_this_with_your_password
```

Where `user` and `pass` under `[sent_2]` correspond to your `scihub` account details, and `user` and `pass` under `[dataspace]` correspond to your `dataspace` account details. <br>

To process Sentinel-2 L1Cs, you will also need Sen2Cor installed: http://step.esa.int/main/third-party-plugins-2/sen2cor/. This installation process is covered in the PyEO_I_Setup.ipynb notebook, available from the notebooks folder.  
<br>

<!-- ## Installation on SEPAL

If you want to use `pyeo` on SEPAL, you can follow these customised instructions below:

1. Register for a SEPAL account at https://docs.sepal.io/en/latest/setup/register.html
1. Request processing credits from the SEPAL Team by providing your use case: https://docs.sepal.io/en/latest/setup/register.html#request-additional-sepal-resources
1. Once approved, from the process screen: https://sepal.io/process, follow the steps below.

Press the terminal `>_` tab to open a Linux terminal

Create a pyeo_home directory in your file system:
```bash
mkdir pyeo_home
```

Move into the pyeo_home directory with: 
```bash
cd pyeo_home
```
Check that `git` is installed on your machine by entering in your terminal:
```bash
git -v
```
If installed it will report its version

1. Because SEPAL already provides git, you can skip the git installation step.
    1. If not, install git by following the install instructions on https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
1. Once git is installed, clone a copy of `pyeo` into your pyeo_home directory:
```bash
git clone https://github.com/clcr/pyeo.git
```

1. Press the spanner shaped tab and click to open JupyterLab
2. When JupyterLab is running navigate to your pyeo_home directory using the panel on the left hand side and then open the 'notebooks' subdirectory -->
<br> 
<!--
1. SEPAL uses `venv` as the package manager for building python libraries, so first create a venv:
```bash
python3 -m venv pyeo_venv
```
2. Then activate the venv:
```bash
source pyeo_venv/bin/activate
```
3. Install the packages that `pyeo` requires into `pyeo_venv`:
```bash
pip install -r pyeo/requirements.txt
```
4. Move into the `pyeo` folder that you cloned from Git:
```bash
cd pyeo
```
5. Install `pyeo` into `pyeo_venv`, be sure to include the `.` at the end of the command!:
```bash
python -m pip install -e .
```
6. Finally, test that `pyeo` was installed correctly by importing a module:
```bash
python
from pyeo import classification
```
7. Now, proceed to the Section below - How to Run PyEO.

<!-- For Linux users, you can optionally access the `pyeo` command line functions, by adding the following to your .bashrc

```bash
export pyeo=/path/to/pyeo
export PATH=$PATH:$pyeo/bin
``` -->
<br>  

## Installation Test Steps

You can test your installation with by typing the following in Bash/Terminal/Anaconda Prompt:
```bash
python
```
```python
import pyeo.classification
```

or, by running the same import command above, after having started a jupyter notebook via:

```bash
jupyter notebook
```

*Please note, if you are using SEPAL, jupyter notebooks have to be started via a GUI method instead of from Bash, see*: https://user-images.githubusercontent.com/149204/132491851-5ac0303f-1064-4e12-9627-f34e3f78d880.png  
<br>  

## How PyEO works
PyEO operates across two stages:  
1. Across Sentinel-2 (S2) tiles
2. Within individual S2 tiles
<br>  


1. Takes a Region of Interest (ROI) and calculates which Sentinel-2 (S2) tiles overlap with the ROI
2. Builds a Baseline Composite to compare land cover changes against, by downloading S2 images and calculating the median of these images.
3. Downloads images over the Change Period
4. Classifies the Composite and the Change images using a classifier in `./models/`
5. Calculates the change between the **from classes** and the **to classes**, for each classified image. This could be changes from forest to bare soil.
6. Creates a Change Report describing the consistency of the class changes, highlighting the changes that `PyEO` is confident.
7. Vectorises the Change Report and removes any changes outside of the ROI

## How to Run PyEO
PyEO can be run interactively in the Jupyter Notebooks provided in the Tutorials, but the pipeline method can be run via the **Terminal**.  This process is automated and is suited to the advanced python user. <br> 
Both the terminal and notebook methods rely on an a configuration file (e.g. `pyeo_linux.ini`, `pyeo_windows.ini`, `pyeo_sepal.ini`) to make processing decisions.  <br>
The below example references `pyeo_sepal.ini`, but this can be switched for the Linux or Windows equivalent. <br>
<!-- add ini file examples here -->

1. First, move to where PyEO is installed:
```bash
cd pyeo
```
2. Now, the pipeline method runs like this. Here we are telling the terminal that we want to invoke `python` to run the script `run_acd_national.py` within the folder `pyeo`, then we pass the absolute path to the initialisation file for your OS. The script `run_acd_national.py` requires this path as all the processing parameters are stored in the initialisation file. See below:
```bash
python pyeo/run_acd_national.py <insert_your_absolute_path_to>/pyeo_sepal.ini

```

The pipeline uses arguments specified in the `.ini` file (short for initialisation), to decide what processes to run.
Here, we will go through the sections of the `ini` file and what arguments do what.

```
[forest_sentinel]
# Acquisition dates for Images in the Change Period, in the form yyyymmdd
start_date=20230101
end_date=20230611

# Acquisition dates for Images for the Baseline Composite, in the form yyyymmdd
composite_start=20220101
composite_end=20221231

# EPSG code, for example - Kenya. This epsg is for areas North of equator and East of 36°E is EPSG:21097
# See https://epsg.io/21097 and https://spatialreference.org/ref/epsg/21097/
epsg=21097

# Cloud cover threshold for imagery to download
cloud_cover=25

# Certainty value above which a pixel is considered a cloud from sen2cor
cloud_certainty_threshold=0

# path to the trained machine learning model for land cover in Kenya
model= ./models/model_36MYE_Unoptimised_20230505_no_haze.pkl
```

## Automated Pipeline Execution
To enable parallel processing of the raster and vector processing pipelines with the `do_parallel = True` option enabled in `pyeo_sepal.ini`, make the following file an executable by issuing this command:
```bash
cd pyeo/apps/automation/
chmod u+x automate_launch.sh
```
<br>  

<!-- ## Further Setup Information
A slightly more verbose setup tutorial for `pyeo` can be found in the notebooks directory, at PyEO_I_Setup_on_SEPAL.ipynb
<br>  -->

## Tutorials
Once installation of `pyeo` is complete, you can follow the tutorial notebooks, which demonstrate the utility of `pyeo`.

How to Train Your Classifier: https://github.com/clcr/pyeo/blob/main/notebooks/PyEO_I_Model_Training.ipynb

Downloading Sentinel-2 Imagery, Creating a Baseline Composite, Performing Automatic Change Detection: https://github.com/clcr/pyeo/blob/main/notebooks/PyEO_I_Master_Tutorials.ipynb

## How to cite this software

Please use the following references when using pyeo:

Roberts, J.F., Mwangi, R., Mukabi, F., Njui, J., Nzioka, K., Ndambiri, J.K., Bispo, P.C., Espirito-Santo, F.D.B., Gou, Y., Johnson, S.C.M. and Louis, V., 2022. Pyeo: A Python package for near-real-time forest cover change detection from Earth observation using machine learning. Computers & Geosciences, 167, p.105192.

Roberts, J., Balzter, H., Gou, Y., Louis, V., Robb, C., 2020. Pyeo: Automated satellite imagery processing. https://doi.org/10.5281/zenodo.3689674

Pacheco-Pascagaza, A.M., Gou, Y., Louis, V., Roberts, J.F., Rodríguez-Veiga, P., da Conceição Bispo, P., Espírito-Santo, F.D., Robb, C., Upton, C., Galindo, G. and Cabrera, E., 2022. Near real-time change detection system using Sentinel-2 and machine learning: A test for Mexican and Colombian forests. Remote Sensing, 14(3), p.707.
