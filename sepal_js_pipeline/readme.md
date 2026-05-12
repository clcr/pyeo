# SEPAL JS Pipeline

This directory hosts the JavaScript pipeline for working with SEPAL, which works with the Google Earth Engine JavaScript API for processing on their servers, but without the use of visual aspects as this is not available without the in-browser code editor.

## Node.js Environment Configuration

```{bash}
# download the node version manager (nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# reload bash with updated profile
source ~/.bashrc

# install Node.js and nvm
nvm install --lts

# check installed correctly
node --version
npm --version

# initialise a new node environment
npm init -y

# install dotenv for credential management
npm install dotenv
```