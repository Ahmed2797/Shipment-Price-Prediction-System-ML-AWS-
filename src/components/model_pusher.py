import sys
from typing import Optional
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import (
    Model_Pusher_Artifact,
    Data_Transformation_Artifact,
    Model_Evaluation_Artifact
)
from src.entity.config_entity import Model_Pusher_Config
from src.entity.s3_estimator import AWSEstimator


class Model_Pusher:
    """
    Handles pushing validated, trained models and preprocessing pipelines 
    to AWS S3 for production deployment.
    """

    def __init__(
        self, 
        model_pusher_config: Model_Pusher_Config,
        data_transformation_artifact: Data_Transformation_Artifact,
        model_evaluation_artifact: Model_Evaluation_Artifact
    ):
        """
        Initialize Model Pusher with config and upstream pipeline artifacts.
        """
        self.model_pusher_config = model_pusher_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        
        # Initialize your estimators/uploaders
        # Note: If your AWSEstimator takes a directory prefix rather than a file key, 
        # ensure your s3_model_key_path reflects a folder structure.
        self.model_estimator = AWSEstimator(
            bucket_name=self.model_pusher_config.bucket_name,
            model_key=self.model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> Model_Pusher_Artifact:
        """
        Pushes both the preprocessing pipeline and the trained model artifact to S3.

        Returns:
            Model_Pusher_Artifact: Structural data containing cloud locations of pushed assets.
        """
        logging.info("Initiating model pusher process...")

        try:
            # 1. Validate local files exist before attempting expensive cloud operations
            local_preprocessing_path = self.data_transformation_artifact.preprocessing_pkl
            local_model_path = self.model_evaluation_artifact.trained_model_path

            logging.info(f"Uploading preprocessing object from: {local_preprocessing_path}")
            # ASSUMPTION: If save_model requires a target file path on S3, 
            # ensure your AWSEstimator handles key concatenation internally, 
            # otherwise pass unique target keys for each file.
            self.model_estimator.save_model(
                local_model_path=local_preprocessing_path,
                remove_local=False
            )

            logging.info(f"Uploading trained model artifact from: {local_model_path}")
            self.model_estimator.save_model(
                local_model_path=local_model_path,
                remove_local=False
            )

            # 2. Build the output metadata artifact
            model_pusher_artifact = Model_Pusher_Artifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info(f"Model assets successfully pushed to S3. Artifact created: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            logging.error("Exception occurred during model pusher stage.")
            raise CustomException(e, sys) from e