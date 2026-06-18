import sys
from src.cloud.aws_stroage import S3StorageService
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import Model_Pusher_Artifact
from src.entity.config_entity import Model_Pusher_Config
from src.entity.s3_estimator import AWSEstimator


class Model_Pusher:
    """
    Handles pushing trained models to S3 for production.
    """

    def __init__(self, model_evaluation_artifact, model_pusher_config: Model_Pusher_Config):
        """
        Initialize Model Pusher.

        Args:
            model_evaluation_artifact: Output reference of model evaluation stage
            model_pusher_config (Model_Pusher_Config): Configuration for model pusher
        """
        self.s3 = S3StorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.model_estimator = AWSEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_key=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> Model_Pusher_Artifact:
        """
        Push the trained model to S3.

        Returns:
            Model_Pusher_Artifact: Information about the uploaded model in S3

        Raises:
            CustomException: If any error occurs during upload
        """
        logging.info("Entered initiate_model_pusher method of Model_Pusher class")

        try:
            logging.info("Uploading trained model to S3 bucket")
            self.model_estimator.save_model(
                local_model_path=self.model_evaluation_artifact.trained_model_path,
                remove_local=False
            )

            model_pusher_artifact = Model_Pusher_Artifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Uploaded trained model to S3 bucket successfully")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of Model_Pusher class")
            
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
