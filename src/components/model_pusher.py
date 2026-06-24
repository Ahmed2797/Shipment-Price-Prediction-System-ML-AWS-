import sys
from typing import Optional
from pathlib import Path
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
        
        # # Initialize your estimators/uploaders
        # # Note: If your AWSEstimator takes a directory prefix rather than a file key, 
        # # ensure your s3_model_key_path reflects a folder structure.
        # self.model_estimator = AWSEstimator(
        #     bucket_name=self.model_pusher_config.bucket_name,
        #     model_key=self.model_pusher_config.s3_model_key_path
        # )

    def initiate_model_pusher(self) -> Model_Pusher_Artifact:
        """
        Pushes both the preprocessing pipeline and the trained model artifact to S3.

        Returns:
            Model_Pusher_Artifact: Structural data containing cloud locations of pushed assets.
        """
        logging.info("Initiating model pusher process...")

        try:
            # from pathlib import Path

            # 1. Cast your string paths to Path objects
            local_preprocessing_path = Path(self.data_transformation_artifact.preprocessing_pkl)
            local_model_path = Path(self.model_evaluation_artifact.trained_model_path)

            # 2. Extract only the filenames using '.name' (replaces os.path.basename)
            preprocessing_filename = local_preprocessing_path.name
            model_filename = local_model_path.name

            # 3. Combine with S3 configurations using Path, but convert to POSIX string for AWS S3
            s3_preprocessor_key = str(
                Path(self.model_pusher_config.s3_preprocessing_obj_path) / preprocessing_filename
            ).replace("\\", "/")

            s3_model_key = str(
                Path(self.model_pusher_config.s3_model_key_path) / model_filename
            ).replace("\\", "/")

            # 4. Execute uploads using the clean strings
            logging.info(f"Uploading preprocessing object to S3: {s3_preprocessor_key}")
            preprocessing_estimator = AWSEstimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_key=s3_preprocessor_key
            )
            preprocessing_estimator.save_model(
                local_model_path=str(local_preprocessing_path),  # Some older S3 wrappers require strings
                remove_local=False
            )

            logging.info(f"Uploading trained model artifact to S3: {s3_model_key}")
            model_estimator = AWSEstimator(
                bucket_name=self.model_pusher_config.bucket_name,
                model_key=s3_model_key
            )
            model_estimator.save_model(
                local_model_path=str(local_model_path),
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