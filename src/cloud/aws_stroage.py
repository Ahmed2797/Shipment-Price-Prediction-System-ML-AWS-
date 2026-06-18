"""
s3_storage_service.py

Production-ready AWS S3 storage service for ML pipelines.
Supports model loading, CSV read/write, file uploads, and prefix management.

Author      : Efrot
Version     : 1.0
Use Case    : Data ingestion, model registry, prediction services
"""

import os
import sys
import pickle
import joblib
from io import StringIO
from typing import Optional

import pandas as pd
from pandas import DataFrame
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket, Object

from src.configuration.aws_connection import S3Client
from src.logger import logging
from src.exception import CustomException


class S3StorageService:
    """
    Centralized service for all AWS S3 operations.

    This class provides a clean abstraction layer over boto3 for:
    - Reading and writing CSV files
    - Uploading files and DataFrames
    - Loading ML models from S3
    - Managing S3 prefixes (folders)

    Designed for production ML pipelines.
    """

    def __init__(self) -> None:
        """
        Initialize S3 resource and client.
        """
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.s3_resource
            self.s3_client = s3_client.s3_client
            logging.info("S3StorageService initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Retrieve an S3 bucket resource.

        Args:
            bucket_name (str): Name of the S3 bucket

        Returns:
            Bucket: boto3 Bucket object
        """
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise CustomException(e, sys)

    def s3_key_exists(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check whether an S3 key exists.

        Args:
            bucket_name (str): S3 bucket name
            s3_key (str): Object key or prefix

        Returns:
            bool: True if key exists, else False
        """
        try:
            bucket = self.get_bucket(bucket_name)
            return any(True for _ in bucket.objects.filter(Prefix=s3_key))
        except Exception as e:
            raise CustomException(e, sys)

    def get_object(self, bucket_name: str, key: str) -> Object:
        """
        Fetch a single S3 object.

        Args:
            bucket_name (str): S3 bucket name
            key (str): Object key

        Returns:
            Object: boto3 S3 Object

        Raises:
            FileNotFoundError: If object does not exist
        """
        try:
            bucket = self.get_bucket(bucket_name)
            objects = list(bucket.objects.filter(Prefix=key))

            if not objects:
                raise FileNotFoundError(f"{key} not found in bucket {bucket_name}")

            if len(objects) > 1:
                logging.warning(
                    f"Multiple objects found for key '{key}', using first match"
                )

            return objects[0]

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_object(
        s3_object: Object,
        decode: bool = True,
        make_readable: bool = False,
    ):
        """
        Read the contents of an S3 object.

        Args:
            s3_object (Object): boto3 S3 Object
            decode (bool): Decode bytes to string
            make_readable (bool): Return StringIO for pandas compatibility

        Returns:
            bytes | str | StringIO: Object content
        """
        try:
            data = s3_object.get()["Body"].read()

            if decode:
                data = data.decode("utf-8")

            return StringIO(data) if make_readable else data

        except Exception as e:
            raise CustomException(e, sys)

    def load_model(
        self,
        bucket_name: str,
        model_name: str,
        model_dir: Optional[str] = None,
    ) -> object:
        """
        Load a pickled ML model from S3.

        Args:
            bucket_name (str): S3 bucket name
            model_name (str): Model file name
            model_dir (Optional[str]): Directory containing model

        Returns:
            object: Deserialized ML model

        Note:
            Assumes the S3 bucket is trusted.
        """
        try:
            key = f"{model_dir}/{model_name}" if model_dir else model_name
            s3_object = self.get_object(bucket_name, key)
            binary_data = self.read_object(s3_object, decode=False)
            return joblib.loads(binary_data)
        except Exception as e:
            raise CustomException(e, sys)

    def upload_file(
        self,
        local_path: str,
        bucket_name: str,
        s3_key: str,
        remove_local: bool = True,
    ) -> None:
        """
        Upload a local file to S3.

        Args:
            local_path (str): Path to local file
            bucket_name (str): S3 bucket name
            s3_key (str): Destination S3 key
            remove_local (bool): Remove local file after upload
        """
        try:
            self.s3_client.upload_file(local_path, bucket_name, s3_key)
            logging.info(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")

            if remove_local:
                os.remove(local_path)
        except Exception as e:
            raise CustomException(e, sys)

    def upload_df_as_csv(
        self,
        df: DataFrame,
        bucket_name: str,
        s3_key: str,
    ) -> None:
        """
        Upload a pandas DataFrame to S3 as a CSV file.

        Args:
            df (DataFrame): Pandas DataFrame
            bucket_name (str): S3 bucket name
            s3_key (str): Destination S3 key
        """
        try:
            buffer = StringIO()
            df.to_csv(buffer, index=False)

            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
            )

            logging.info(f"DataFrame uploaded to s3://{bucket_name}/{s3_key}")
        except Exception as e:
            raise CustomException(e, sys)

    def read_csv(self, bucket_name: str, s3_key: str) -> DataFrame:
        """
        Read a CSV file from S3 into a DataFrame.

        Args:
            bucket_name (str): S3 bucket name
            s3_key (str): CSV file key

        Returns:
            DataFrame: Loaded pandas DataFrame
        """
        try:
            s3_object = self.get_object(bucket_name, s3_key)
            content = self.read_object(s3_object, make_readable=True)
            return pd.read_csv(content)
        except Exception as e:
            raise CustomException(e, sys)

    def create_prefix(self, bucket_name: str, prefix: str) -> None:
        """
        Create an S3 prefix (folder-like structure).

        Args:
            bucket_name (str): S3 bucket name
            prefix (str): Prefix path
        """
        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=f"{prefix.rstrip('/')}/",
            )
            logging.info(f"Created prefix s3://{bucket_name}/{prefix}")
        except ClientError as e:
            raise CustomException(e, sys)
