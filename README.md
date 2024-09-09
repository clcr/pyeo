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
  - [SEPAL Orientation](./notebooks/PyEO_sepal_orientation.ipynb)
  - [Training a PyEO ML Model on SEPAL](./notebooks/PyEO_sepal_model_training.ipynb)
  - [Running PyEO deforestation monitoring on SEPAL](./notebooks/PyEO_sepal_pipeline_training.ipynb)

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

### Google Colab (Linux)
```
# from a Jupyter Notebook in Colab, do this:
# clone and install a github repo on Colab: https://github.com/clcr/pyeo
!pwd
!git clone https://github.com/clcr/pyeo.git
!pip install git+https://github.com/clcr/pyeo
```

### Windows
```bash
conda install -c conda-forge git
git clone https://github.com/clcr/pyeo.git
cd pyeo
conda env create --file environment_windows_w1.yml --name pyeo_env
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

To process Sentinel-2 L1Cs, you will also need Sen2Cor installed: http://step.esa.int/main/third-party-plugins-2/sen2cor/.

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

To learn about how to install PyEO, the easiest way is to apply for a free account on [SEPAL](https://sepal.io), and to follow the SEPAL tutorial notebooks indicated at the top of this file.

You can test your installation by typing the following in Bash/Terminal/Anaconda Prompt:
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
As a broad overview, PyEO implements the following steps:

1. Takes a Region of Interest (ROI) and calculates which Sentinel-2 (S2) tiles overlap with the ROI.
2. Builds a Baseline Composite to compare land cover changes against, by downloading S2 images and calculating the median of these images.
3. Downloads Change Images over the Change Period to be monitored.
4. Classifies the Composite and the Change Images using a Random Forest Classifier.
5. Calculates the Class Changes between sets of **from classes** and **to classes**, for each classified image. For example, this could be changes **from** forest **to** bare soil.
6. Creates a Change Report describing the consistency of the Class Changes.
7. Vectorises the Change Report and filters out any Class Changes outside of the ROI.

## How to Run PyEO as an Automated Pipeline
More information on how to run PyEO as a pipeline from the command line can be found in the [SEPAL User Guide](./sepal_guide/SEPAL_User_Guide_PyEO_Forest_Alert_System.pdf).

## How to cite this software

Please use the following references when using pyeo:

Roberts, J.F., Mwangi, R., Mukabi, F., Njui, J., Nzioka, K., Ndambiri, J.K., Bispo, P.C., Espirito-Santo, F.D.B., Gou, Y., Johnson, S.C.M. and Louis, V., 2022. Pyeo: A Python package for near-real-time forest cover change detection from Earth observation using machine learning. Computers & Geosciences, 167, p.105192.

Roberts, J., Balzter, H., Gou, Y., Louis, V., Robb, C., 2020. Pyeo: Automated satellite imagery processing. https://doi.org/10.5281/zenodo.3689674

Pacheco-Pascagaza, A.M., Gou, Y., Louis, V., Roberts, J.F., Rodríguez-Veiga, P., da Conceição Bispo, P., Espírito-Santo, F.D., Robb, C., Upton, C., Galindo, G. and Cabrera, E., 2022. Near real-time change detection system using Sentinel-2 and machine learning: A test for Mexican and Colombian forests. Remote Sensing, 14(3), p.707.
