PyEO Troubleshooting Guide

This guide covers common issues encountered when installing and using PyEO, along with their solutions.

Table of Contents

Installation Issues
Windows-Specific Issues
Data Download Issues
Processing Errors
General Tips

Installation Issues

Issue: conda: command not found

Solution:

Install Anaconda or Miniconda from https://docs.conda.io/en/latest/
Restart your terminal after installation
Verify installation: conda --version
Issue: Environment creation fails with dependency conflicts

Solution:

# Try creating with a fresh conda cache

conda clean --all

conda env create --file environment.yml --name pyeo_env

 

# If still failing, try updating conda first

conda update -n base conda

Issue: ModuleNotFoundError: No module named 'pyeo'

Solution: Make sure you've activated the environment and installed pyeo:

conda activate pyeo_env

python -m pip install -e .


Windows-Specific Issues

Issue: MissingSectionHeaderError when reading config files

Problem: Configuration files (like pyeo_windows.ini) have encoding issues on Windows.

Solution:

Open the config file in Notepad
Go to File → Save As
At the bottom, change encoding from "UTF-8" to "ANSI"
Save and try again
Issue: Sen2Cor not found on Windows

Solution:

Download Sen2Cor from http://step.esa.int/main/third-party-plugins-2/sen2cor/
Install it and note the installation path (e.g., C:\Sen2Cor-02.11.00)
Add Sen2Cor to your PATH:
Search for "Environment Variables" in Windows
Edit PATH and add the Sen2Cor directory
Restart your terminal
Issue: Long file paths causing errors

Solution: Windows has a 260-character path limit. To fix:

Use shorter folder names
Move your working directory closer to C:\ (e.g., C:\pyeo)
Enable long paths in Windows 10/11:
·       Run as Administrator: Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1


Data Download Issues

Issue: Authentication failed when downloading Sentinel-2 data

Solution:

Check that you've created accounts:
Copernicus Data Space: https://dataspace.copernicus.eu
(Optional) Scihub: https://scihub.copernicus.eu
Verify your credentials in credentials/credentials_dummy.ini:
3.  [sent_2]user=your_email@example.compass=your_password[dataspace]user=your_email@example.compass=your_password

Make sure there are no extra spaces in the file
Try resetting your password if login still fails
Issue: Downloads are very slow or timing out

Solution:

Check your internet connection
The Copernicus servers can be slow during peak hours (try different times)
Reduce the date range or area of interest to download less data
Consider using SEPAL (https://sepal.io) which has faster access
Issue: "No images found for the specified date range"

Solution:

Sentinel-2 satellites don't cover every location every day
Expand your date range (try at least 1-2 months)
Check cloud coverage limits (increase if needed)
Verify your area of interest coordinates are correct

Processing Errors

Issue: Out of memory errors during processing

Solution:

Process smaller areas at a time
Reduce the number of images in your composite
Close other applications to free up RAM
If using SEPAL, request a larger instance
Issue: Classification fails with "No valid training data"

Solution:

Check that your training data file exists and is in the correct format
Ensure training data covers all classes you want to classify
Verify training data coordinates overlap with your study area
Training data should have at least 50-100 samples per class
Issue: All pixels classified as "No Data" or "Cloud"

Solution:

Your cloud masking might be too aggressive
Check the cloud probability threshold in your config
Verify that Sen2Cor is properly installed for atmospheric correction
Try using a different baseline period with less cloud cover

General Tips

Best Practices

Start small: Test on a small area before processing large regions
Check your data: Always visualize downloaded images before processing
Keep backups: Save your composite and trained models
Document your workflow: Keep notes on parameters and settings used
Getting Help

Check the full documentation: https://clcr.github.io/pyeo/build/html/index.html
Read the SEPAL User Guide (even if not using SEPAL): [Link in repo]
Search existing GitHub issues: https://github.com/clcr/pyeo/issues
Create a new issue with:
Your operating system
Full error message
Steps to reproduce
Config file settings (remove passwords!)
Useful Commands for Debugging

# Check your environment

conda list

 

# Verify pyeo installation

python -c "import pyeo; print(pyeo.__version__)"

 

# Test imports

python -c "import pyeo.classification"

python -c "import pyeo.raster_manipulation"

 

# Check GDAL installation (important for raster processing)

gdalinfo --version


Still Having Issues?

If you've tried the solutions above and still have problems:

Search existing issues: https://github.com/clcr/pyeo/issues
Create a new issue with:
Operating system and Python version
Complete error message
Steps to reproduce the problem
What you've already tried
Consider using SEPAL: If local installation is problematic, SEPAL (https://sepal.io) has PyEO pre-installed

Last Updated: December 2024
Contributions Welcome: Please submit a pull request if you find solutions to other common issues!

 