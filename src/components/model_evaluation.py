import sys
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from src.entity.config_entity import Data_Transformation_Config, Model_Evaluation_Config
from src.entity.artifact_entity import (
    Data_Ingestion_Artifact,
    Model_Evaluation_Artifact,
    Model_Trainer_Artifact,
)
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, read_yaml
from src.constants import COLUMN_YAML_FILE_PATH
from src.entity.s3_estimator import AWSEstimator


@dataclass
class Evaluate_Model_Response:
    trained_model_r2: float
    trained_model_mae: float
    best_model_r2: Optional[float]
    is_model_accepted: bool
    difference: float


class Model_Evaluation:
    def __init__(
        self,
        model_eval_config: Model_Evaluation_Config,
        data_ingestion_artifact: Data_Ingestion_Artifact,
        model_trainer_artifact: Model_Trainer_Artifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_best_model(self, trained_model_path: Optional[str] = None):
        """
        Load local baseline model (AWS mode intentionally disabled).
        Set `model_eval_config.s3_model_key_path` to a local model path.
        """
        try:
            local_model_path = self.model_eval_config.s3_model_key_path
            trained_abs = os.path.abspath(trained_model_path) if trained_model_path else None
            basename = os.path.basename(local_model_path) if local_model_path else None

            candidate_paths = []
            if local_model_path:
                candidate_paths.append(local_model_path)
            if basename:
                candidate_paths.extend(
                    [
                        os.path.join("artifact", "model_trainer", basename),
                        os.path.join("final_model", "prediction_model", basename),
                    ]
                )

            seen = set()
            unique_paths = []
            for path in candidate_paths:
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)

            for path in unique_paths:
                path_abs = os.path.abspath(path)
                if trained_abs and path_abs == trained_abs:
                    continue
                if os.path.exists(path):
                    logging.info(f"Using local baseline model: {path}")
                    return load_object(file_path=path), path

            logging.info(
                f"Local baseline model not found at '{local_model_path}'. "
                "Evaluation will accept trained model by default."
            )


            # AWS production mode (enable later):
            bucket_name = self.model_eval_config.bucket_name
            model_key = self.model_eval_config.s3_model_key_path
            model_estimator = AWSEstimator(bucket_name=bucket_name, model_key=model_key)
            if model_estimator.is_model_present():
                # return actual loaded model and identifier
                loaded = model_estimator.load_model()
                return loaded, model_key

            return None
            
        except Exception as e:
            raise CustomException(e, sys)


    def apply_binary_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binary mapping from schema (Yes/No, Male/Female, etc.).
        """
        df_copy = df.copy()
        binary_cols = self._column_schema.get("binary_categorical_columns", {})

        for col, details in binary_cols.items():
            if col in df_copy.columns:
                mapping = details.get("mapping", {})
                df_copy[col] = df_copy[col].replace(mapping)

        return df_copy

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transform for configured columns.
        """
        df_copy = df.copy()
        log_cols = self._column_schema.get("log_transform_col", [])

        for col in log_cols:
            if col in df_copy.columns:
                df_copy[col] = np.log1p(df_copy[col].clip(lower=0))

        return df_copy

    def _transform_for_raw_model(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Build transformed features for raw sklearn model inference.
        """
        transform_cfg = Data_Transformation_Config()
        preprocessor = load_object(file_path=transform_cfg.transform_object_path)

        x_trans = preprocessor.transform(x)
        feature_names = preprocessor.get_feature_names_out()
        x_trans = pd.DataFrame(x_trans, columns=feature_names, index=x.index)
        # Raw sklearn models in this src are trained from numpy arrays.
        # Return numpy here to keep fit/predict input style consistent.
        return x_trans.to_numpy()

    def _prepare_target(self, y: pd.Series) -> np.ndarray:
        """
        Prepare target for evaluation using same transform as training if available.
        Falls back to log1p(abs(y)).
        """
        try:
            transform_cfg = Data_Transformation_Config()
            transform_obj_dir = os.path.dirname(transform_cfg.transform_object_path)
            target_transformer_path = os.path.join(transform_obj_dir, "target_transformer.pkl")
            if os.path.exists(target_transformer_path):
                target_transformer = load_object(file_path=target_transformer_path)
                return target_transformer.transform(y)
            return np.log1p(np.abs(y.values))
        except Exception as e:
            raise CustomException(e, sys)


    def _predict_with_any_model(self, model_obj: Any, x_raw: pd.DataFrame) -> np.ndarray:
        """
        Predict robustly for both:
        - ProjectModel (raw features)
        - raw sklearn model (transformed features)
        """
        if hasattr(model_obj, "transform_object"):
            return self._to_1d_predictions(model_obj.predict(x_raw))
        x_trans = self._transform_for_raw_model(x_raw)
        return self._to_1d_predictions(model_obj.predict(x_trans))

    def _to_1d_predictions(self, preds: Any) -> np.ndarray:
        """
        Normalize predictions to 1D numpy array.
        Accepts DataFrame (with one column), Series, list-like, or numpy array.
        """
        if isinstance(preds, pd.DataFrame):
            # If DataFrame contains a single column with named prediction, extract it
            if "Predicted_Shipment_Cost" in preds.columns:
                arr = preds["Predicted_Shipment_Cost"].to_numpy()
            else:
                # take first column
                arr = preds.iloc[:, 0].to_numpy()
            return arr.ravel()

        if isinstance(preds, pd.Series):
            return preds.to_numpy().ravel()

        # numpy array or list-like
        arr = np.asarray(preds)
        return arr.ravel()

    def evaluate_model(self) -> Evaluate_Model_Response:
        """
        Evaluate trained model against local baseline model (if available).
        """
        try:
            x, y_true = self._prepare_features_and_target()

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)

            # Ensure y_true is prepared using same transformation as training
            if isinstance(y_true, np.ndarray):
                y_true_prep = y_true
            else:
                y_true_prep = self._prepare_target(y_true)

            y_hat_trained = self._predict_with_any_model(trained_model, x)

            trained_r2 = r2_score(y_true_prep, y_hat_trained)
            trained_mae = mean_absolute_error(y_true_prep, y_hat_trained)
            logging.info(f"Trained model R2 on evaluation set: {trained_r2:.4f}")
            logging.info(f"Trained model MAE on evaluation set: {trained_mae:.4f}")

            best_model_r2 = None
            best_candidate = self.get_best_model(
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )
            if best_candidate is not None:
                best_model_obj, best_model_path = best_candidate
                y_hat_best_model = self._predict_with_any_model(best_model_obj, x)
                best_model_r2 = r2_score(y_true_prep, y_hat_best_model)
                logging.info(f"Baseline model R2 on evaluation set: {best_model_r2:.4f} (path={best_model_path})")
            else:
                logging.info("No local baseline model found. Accepting trained model by default.")

            tmp_best_model_score = 0.0 if best_model_r2 is None else best_model_r2
            difference = trained_r2 - tmp_best_model_score
            threshold = float(self.model_eval_config.changed_threshold_score)
            is_model_accepted = best_model_r2 is None or difference > threshold

            result = Evaluate_Model_Response(
                trained_model_r2=trained_r2,
                trained_model_mae=trained_mae,
                best_model_r2=best_model_r2,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> Model_Evaluation_Artifact:
        """
        Initiate model evaluation and return artifact.
        """
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = Model_Evaluation_Artifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
