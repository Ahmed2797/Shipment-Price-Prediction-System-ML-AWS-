from src.components.data_ingestion import Data_Ingestion 

from src.entity.config_entity import (
    Data_Ingestion_Config
)

from src.entity.artifact_entity import (
    Data_Ingestion_Artifact
)

from src.exception import CustomException
from src.logger import logging
import sys


class Training_Pipeline:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()


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

        
        
    def run_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("======= Training Pipeline Execution Started =======")

            data_ingestion_artifact = self.get_started_data_ingestion()
            
            
            return None

        except Exception as e:
            raise CustomException(e, sys)
