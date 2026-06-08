from src.logger import logging                         
from src.exception import CustomException 
from src.cloud.data_load import Data_Extract 
from src.utils import read_yaml 
from src.constants import COLUMN_YAML_FILE_PATH
from src.entity.config_entity import Data_Ingestion_Config 
from src.entity.artifact_entity import Data_Ingestion_Artifact
from sklearn.model_selection import train_test_split
import pandas as pd 
import sys 
import os



class Data_Ingestion:

    '''
    Class to handle data ingestion from (MongoDB, save feature store,
    split into train/test sets, and create an artifact object)

    '''


    def __init__(self, ingestion_config: Data_Ingestion_Config):
        try:
            self.ingestion_config = ingestion_config
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_stored(self) -> pd.DataFrame:
        try:
            logging.info("Starting data extraction from MongoDB...")
            extract_data = Data_Extract()
            dataframe = extract_data.get_dataframe(self.ingestion_config.data_ingestion_collection_name)

            feature_store = self.ingestion_config.raw_path
            os.makedirs(os.path.dirname(feature_store), exist_ok=True)
            dataframe.to_csv(feature_store, index=False, header=True)
            logging.info('Data extracted and saved successfully in feature store')

            #print(dataframe.head())
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, dataframe: pd.DataFrame):
        try:
            # logging.info('Drop_Unnessary_Column')
            # dataframe = dataframe.drop(self._column_schema['drop_columns'], axis=1, errors='ignore')
            
            logging.info('Splitting data into train and test sets...')
            train_data, test_data = train_test_split(
                dataframe, test_size=self.ingestion_config.split_ratio, random_state=42
            )

            train_file_path = self.ingestion_config.train_path
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_data.to_csv(train_file_path, index=False, header=True)

            test_file_path = self.ingestion_config.test_path
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            test_data.to_csv(test_file_path, index=False, header=True)

            logging.info('Train and test data saved successfully')
            return train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)

    def init_data_ingestion(self):
        try:
            dataframe = self.get_feature_stored()
            self.split_data(dataframe=dataframe)

            artifact = Data_Ingestion_Artifact(
                train_file_path=self.ingestion_config.train_path,
                test_file_path=self.ingestion_config.test_path
            )

            logging.info('Data ingestion process completed successfully')
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
        
