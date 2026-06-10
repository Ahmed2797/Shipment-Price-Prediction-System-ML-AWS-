from dataclasses import dataclass 
from src.constants import * 
from datetime import datetime 
import os


Timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

# @dataclass 
# class Project_Configuration:
#     artifact: str = os.path.join(Artifact, Timestamp)
#     pipeline: str = Pipeline_dir
#     timestamp: str = Timestamp
#     model_dir: str = os.path.join(final_model, Timestamp)


@dataclass 
class Project_Configuration:
    artifact: str = Artifact  # just Artifact, no timestamp
    pipeline: str = Pipeline_dir
    timestamp: str = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    model_dir: str = final_model  


project_config = Project_Configuration()


# ================================================================
# DATA INGESTION CONFIG 
# Fix: train_path & test_path must NOT duplicate parent dir
# ================================================================
@dataclass 
class Data_Ingestion_Config:
    data_ingestion_dir = os.path.join(project_config.artifact, DATA_INGESTION_DIR)
    # Keep this path as the raw feature-store file for backward compatibility.
    data_ingestion_feature_stored_dir = os.path.join(data_ingestion_dir, Raw_Data)
    data_ingestion_ingested_dir = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)

    # Canonical ingestion artifact files
    raw_path = os.path.join(data_ingestion_dir, Raw_Data)
    train_path = os.path.join(data_ingestion_dir, Train_Data)
    test_path = os.path.join(data_ingestion_dir, Test_Data)

    split_ratio = DATA_INGESTION_SPLIT_RATIO 
    data_ingestion_collection_name = Collection_name 


# ================================================================
# DATA VALIDATION CONFIG
# ================================================================
@dataclass 
class Data_Validation_Config:
    data_validation_dir = os.path.join(project_config.artifact, DATA_VALIDATION_DIR)
    report_dir = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_DIR)
    report_status = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_STATUS)

