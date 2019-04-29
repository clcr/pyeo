# pyeo
Python for Earth Observation processing chain

This is designed to provide a set of portable, extensible and modular Python scripts for earth observation and machine learning,
including downloading, preprocessing, creation of base layers and classification.

## Requirements
Package management is performed by Conda: https://docs.conda.io/en/latest/

For downloading, you will need a Scihub account: https://scihub.copernicus.eu/

For Sentinel 2 processing, you will need Sen2Cor installed: http://step.esa.int/main/third-party-plugins-2/sen2cor/

For AWS downloading, you will need credentials set up on your machine.

## To install

With Git installed, `cd` to an install location then run the following lines

```bash
git clone https://github.com/clcr/pyeo.git
cd pyeo
conda env create --file environment.yml --name pyeo_env
source activate pyeo_env
```

At present, Pyeo does not insert itself into your Python path. Instead, add the following lines to the start of your programs:

```python
import sys
sys.path.append("/path/to/pyeo/")
```

To verify the installation, open a Python prompt and type

```python
>>> import pyeo.core
```

You should get no errors.


Full documentation at https://clcr.github.io/pyeo/build/html/index.html
