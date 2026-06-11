import os
from datetime import date
from os import environ
import numpy as np
from dotenv import load_dotenv
load_dotenv()

Current_Year = date.today().year

# MongoDB
Data_Base_Name = 'ShipmentDB'
Collection_name = 'Ship-connection'
MONGODB_URL_KEY = 'MONGODB_URL'
# MONGODB_URL_KEY = environ['MONGODB_URL']

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
PARAM_YAML_FILE = os.path.join('config','model.yaml')

# DataIngestion
DATA_INGESTION_DIR:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORED_DIR:str = Raw_Data #'feature.csv'
DATA_INGESTION_INGESTED_DIR:str = 'ingested'
DATA_INGESTION_SPLIT_RATIO:float = 0.2
Collection_name:str = 'Ship-connection'

# Data_validation 
DATA_VALIDATION_DIR:str = 'data_validation'
DATA_VALIDATION_REPORT_DIR:str = 'drift_report'
DATA_VALIDATION_REPORT_STATUS:str = 'report.yaml'

# Data_Transformation
DATA_TRANSFORMATION_DIR = "data_transform"
TRANSFORM_FILE = "transform"
TRANSFORM_OBJECT = "transform_obj"
PREPROCESSING_OBJECT = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "n_neighbors": 3,
    "weights": "uniform",
    "missing_values": np.nan
}
