import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')))
import pyeo.core as pyeo
import argparse
import configparser
import boto3

if __name__ == "__main__":
    parser = configparser.ConfigParser()
