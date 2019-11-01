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
conda activate pyeo_env
python -m pip install -e . -vv
```
If you want access to the Pyeo command line functions, add the following to your .bashrc

```bash
export PYEO=/path/to/pyeo
export PATH=$PATH:$PYEO/bin
```
You can test this by restarting your shell and then running a set 

If you do not want to edit Pyeo, replace the pip install line with

```bash
python -m pip install . -vv
```

You can test your installation with
`import pyeo.classifier`

Full documentation at https://clcr.github.io/pyeo/build/html/index.html
