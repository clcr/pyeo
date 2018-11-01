"""Downloads the prerequisite bits from S3"""
import boto3
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulls a config file, an AOI and a model from S3")
    parser.add_argument("config_file", help="The S3 path to the config file for this instance")
    parser.add_argument("aoi", help="The S3 path to the AOI for this instance")
    parser.add_argument("model", help="The S3 path to the model for this instance")
    args = parser.parse_args()

    s3_service = boto3.resource("s3")
    s3_service.meta.client.download_file("forestsentinelconfig", args.config_file, "/home/ubuntu/config/cd_config.ini")
    s3_service.meta.client.download_file("forestsentinelconfig", args.model, "/home/ubuntu/config/model.pkl")
    s3_service.meta.client.download_file("forestsentinelconfig", args.aoi, "/home/ubuntu/config/aoi.json")
