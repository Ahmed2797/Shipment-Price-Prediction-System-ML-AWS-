from src.components.data_ingestion import Data_Ingestion
from src.components.data_validation import Data_Validation
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer

from src.entity.config_entity import (
    Data_Ingestion_Config,
    Data_Validation_Config,
    Data_Transformation_Config,
    Model_Trainer_Config
)

from src.entity.artifact_entity import (
    Data_Ingestion_Artifact,
    Data_Validation_Artifact,
    Data_Transformation_Artifact,
    Model_Trainer_Artifact
)

from src.exception import CustomException
from src.logger import logging
import sys


class Training_Pipeline:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()
        self.data_validation_config = Data_Validation_Config()
        self.data_transformation_config = Data_Transformation_Config()
        self.model_trainer_config = Model_Trainer_Config()



    def get_started_data_ingestion(self) -> Data_Ingestion_Artifact:
        try:
            logging.info(">>>>>>>>>>>  Data Ingestion Started  >>>>>>>>>>>>")

            data_ingestion = Data_Ingestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.init_data_ingestion()

            logging.info(">>>>>>>>>>>  Data Ingestion Completed  >>>>>>>>>>>>")
            logging.info(data_ingestion_artifact)

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_started_data_validation(self, data_ingestion_artifact: Data_Ingestion_Artifact) -> Data_Validation_Artifact:
        try:
            logging.info(">>>>>>>>>>>  Data Validation Started  >>>>>>>>>>>>")

            data_validation = Data_Validation(self.data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.init_data_validation()

            logging.info(">>>>>>>>>>>  Data Validation Completed  >>>>>>>>>>>>")
            logging.info(data_validation_artifact)

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def get_started_data_transformation(
        self,
        data_ingestion_artifact: Data_Ingestion_Artifact,
        data_validation_artifact: Data_Validation_Artifact
    ) -> Data_Transformation_Artifact:
        try:
            logging.info(">>>>>>>>>>>  Data Transformation Started  >>>>>>>>>>>>")

            data_transformation = Data_Transformation(
                self.data_transformation_config,
                data_ingestion_artifact,
                data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info(">>>>>>>>>>>  Data Transformation Completed  >>>>>>>>>>>>")
            logging.info(data_transformation_artifact)

            return data_transformation_artifact
                
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_model_trainer(self,
        data_transformation_artifact:Data_Transformation_Artifact
    ) -> Model_Trainer_Artifact:
        try:
            logging.info(">>>>>>>>>>>  Model Training Started  >>>>>>>>>>>>")

            model_train = Model_Trainer(
                data_transformation_artifact,self.model_trainer_config,
            )
            model_trainer_artifact = model_train.init_model()

            logging.info(">>>>>>>>>>>  Model Training Completed  >>>>>>>>>>>>")
            logging.info(model_trainer_artifact)

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def run_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("======= Training Pipeline Execution Started =======")

            data_ingestion_artifact = self.get_started_data_ingestion()
            data_validation_artifact = self.get_started_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.get_started_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )
            model_trainer_artifact = self.get_model_trainer(data_transformation_artifact)


            
            return None

        except Exception as e:
            raise CustomException(e, sys)
