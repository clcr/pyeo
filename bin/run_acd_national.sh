#!/bin/bash
set -o errexit
set -o nounset

python "$pyeo"/pyeo/run_acd_national.py "$pyeo"/pyeo/pyeo_linux.ini
