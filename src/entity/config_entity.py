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


# ================================================================
# DATA TRANSFORMATION CONFIG
# ================================================================
@dataclass
class Data_Transformation_Config:
    data_transformation_dir: str = os.path.join(project_config.artifact, DATA_TRANSFORMATION_DIR)
    
    transform_train_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_FILE, 
        Train_Data.replace('csv', 'npy')
    )
    transform_test_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_FILE, 
        Test_Data.replace('csv', 'npy')
    )
    transform_object_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_OBJECT, 
        PREPROCESSING_OBJECT
    )

    final_model_path: str = os.path.join(project_config.model_dir, PREPROCESSING_OBJECT)



# ================================================================
# MODEL TRAINER CONFIG
# Fix: best_model_object path corrected
# ================================================================
@dataclass 
class Model_Trainer_Config:
    model_train_dir: str = os.path.join(project_config.artifact, MODEL_TRAINER_DIR)
    # model_train_file_path: str = os.path.join(model_train_dir, MODEL_TRAINER_FILE_PATH)

    best_model_object: str = os.path.join(model_train_dir, BEST_MODEL_OBJECT)
    # model_train_file_path: str = best_model_object

    excepted_score: float = EXCEPTED_SCORE 
    param_yaml = PARAM_YAML_FILE

    mlflow_tracking_uri: str = 'https://dagshub.com/Ahmed2797/Shipment-Price-Prediction-System-ML-AWS-.mlflow'
    mlflow_experiment_name: str = 'Shipment-Cost-V1'

    # final_model_path: str = os.path.join(project_config.model_dir, BEST_MODEL_OBJECT)
    final_model_path: str = os.path.join(project_config.model_dir,"prediction_model",PREDICTION_BEST_MODEL_OBJECT)
    model_train_file_path: str = final_model_path
    

@dataclass
class Model_Evaluation_Config:
    """
    Configuration class for evaluating machine learning models.

    Attributes:
        performance has changed significantly and needs retraining or redeployment.
        bucket_name (str): Name of the S3 bucket where the model or evaluation artifacts are stored.
        s3_model_key_path (str): S3 key path (file name) of the model to be evaluated.
    """
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str =  MODEL_PUSHER_S3_MODEL_PATH ## BEST_MODEL_OBJECT
    changed_threshold_score: float = THRESHOLD


@dataclass
class Model_Pusher_Config:
    """
    Configuration class for pushing machine learning models to production storage.

    Attributes:
        bucket_name (str): Name of the S3 bucket where the model should be uploaded.
        s3_model_key_path (str): S3 key path (file name) for saving the model in the bucket.
    """
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_PUSHER_S3_MODEL_PATH ## BEST_MODEL_OBJECT
    s3_preprocessing_obj_path:str = MODEL_PUSHER_S3_PREPROCESSING_PATH