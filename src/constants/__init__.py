import os
from datetime import date

Current_Year = date.today().year

# MongoDB
Data_Base_Name = 'ShipmentDB'
Collection_name = 'Ship-connection'
MONGODB_URL_KEY = 'MONGODB_URL' 

# Artifacts 
Artifact = 'artifact'
Pipeline_dir = 'pipeline'
final_model = 'final_model'

# Data
Raw_Data = 'raw.csv'
Train_Data = 'train.csv'
Test_Data = 'test.csv' 

# Target_column
Target_Column = 'Cost'


# yaml_file
COLUMN_YAML_FILE_PATH = os.path.join('config','schema.yaml')
PARAM_YAML_FILE = os.path.join('yaml_file','model.yaml')

# DataIngestion
DATA_INGESTION_DIR:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORED_DIR:str = Raw_Data #'feature.csv'
DATA_INGESTION_INGESTED_DIR:str = 'ingested'
DATA_INGESTION_SPLIT_RATIO:float = 0.2
Collection_name:str = 'Ship-connection'
