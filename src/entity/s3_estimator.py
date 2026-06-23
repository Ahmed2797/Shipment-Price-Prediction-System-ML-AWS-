"""
aws_estimator.py

Production-ready AWS estimator for model persistence and inference.
"""

import sys
from typing import Optional
from pandas import DataFrame

from src.cloud.aws_stroage import S3StorageService
from src.exception import CustomException
from src.entity.estimator import ShipmentPriceEstimator


class AWSEstimator:
    """
    Handles model storage, retrieval, and prediction using AWS S3.
    """

    def __init__(self, bucket_name: str, model_key: str):
        """
        Initialize AWS estimator.

        Args:
            bucket_name (str): S3 bucket name
            model_key (str): S3 key where model is stored
        """
        self.bucket_name = bucket_name
        self.model_key = model_key
        self.s3 = S3StorageService()
        self._model: Optional[ShipmentPriceEstimator] = None

    def is_model_present(self) -> bool:
        """
        Check if model exists in S3.

        Returns:
            bool: True if model exists
        """
        try:
            return self.s3.s3_key_exists(
                bucket_name=self.bucket_name,
                s3_key=self.model_key,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self) -> ShipmentPriceEstimator:
        """
        Load model from S3 (cached after first load).

        Returns:
            ProjectModel: Loaded model
        """
        try:
            if self._model is None:
                self._model = self.s3.load_model(
                    bucket_name=self.bucket_name,
                    model_name=self.model_key,
                )
            return self._model
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, local_model_path: str, remove_local: bool = False) -> None:
        """
        Upload model file to S3.

        Args:
            local_model_path (str): Local path to model file
            remove_local (bool): Remove local file after upload
        """
        try:
            self.s3.upload_file(
                local_path=local_model_path,
                bucket_name=self.bucket_name,
                s3_key=self.model_key,
                remove_local=remove_local,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Perform prediction using loaded model.

        Args:
            dataframe (DataFrame): Input data

        Returns:
            Prediction output
        """
        try:
            model = self.load_model()
            return model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(e, sys)
