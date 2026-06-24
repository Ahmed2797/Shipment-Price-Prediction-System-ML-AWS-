import io
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from src.entity.artifact_entity import (
    Data_Ingestion_Artifact,
    Data_Transformation_Artifact,
    Model_Evaluation_Artifact,
    Model_Trainer_Artifact,
    RegressionMetricArtifact,
)
from src.entity.config_entity import Model_Evaluation_Config
from src.entity.s3_estimator import AWSEstimator
from src.exception import CustomException
from src.logger import logging
from src.utils import load_numpy_array, load_object


@dataclass
class Evaluate_Model_Response:
    trained_model_score: float
    best_model_score: Optional[float]
    is_model_accepted: bool
    difference: float


class Model_Evaluation:
    """
    Compare the newly trained model with the current production model.

    The trainer saves a raw sklearn-compatible model and trains it on the
    transformed numpy arrays, so evaluation uses the transformed test array too.
    """

    def __init__(
        self,
        model_eval_config: Model_Evaluation_Config,
        data_ingestion_artifact: Data_Ingestion_Artifact,
        data_transformation_artifact: Data_Transformation_Artifact,
        model_trainer_artifact: Model_Trainer_Artifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def _is_regression_model(self) -> bool:
        metric_artifact = self.model_trainer_artifact.metric_artifact
        return isinstance(metric_artifact, RegressionMetricArtifact) or hasattr(
            metric_artifact, "r2_score"
        )

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            test_arr = load_numpy_array(self.data_transformation_artifact.transform_test_path)
            if test_arr.ndim != 2 or test_arr.shape[1] < 2:
                raise ValueError(
                    "Transformed test array must contain feature columns and target column."
                )

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            return x_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _predict(model: Any, x_test: np.ndarray) -> np.ndarray:
        prediction = model.predict(x_test)
        return np.asarray(prediction).ravel()

    def _model_score(self, model: Any, x_test: np.ndarray, y_test: np.ndarray) -> float:
        try:
            y_pred = self._predict(model, x_test)

            if self._is_regression_model():
                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                logging.info(
                    f"Regression evaluation | R2={score:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
                )
                return float(score)

            score = accuracy_score(y_test, y_pred)
            logging.info(f"Classification evaluation | Accuracy={score:.4f}")
            return float(score)
        except Exception as e:
            raise CustomException(e, sys)

    def _local_baseline_candidates(self) -> list[str]:
        model_key_or_path = self.model_eval_config.s3_model_key_path
        trained_model_path = self.model_trainer_artifact.trained_model_file_path
        trained_abs_path = os.path.abspath(trained_model_path)

        candidates = []
        if model_key_or_path:
            candidates.append(model_key_or_path)

            basename = os.path.basename(model_key_or_path.rstrip(os.sep))
            if basename:
                candidates.extend(
                    [
                        os.path.join("artifact", "model_trainer", basename),
                        os.path.join("final_model", "prediction_model", basename),
                    ]
                )

        unique_candidates = []
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if os.path.abspath(path) == trained_abs_path:
                continue
            unique_candidates.append(path)

        return unique_candidates

    def _load_local_baseline_model(self) -> Optional[Tuple[Any, str]]:
        for model_path in self._local_baseline_candidates():
            if os.path.isfile(model_path):
                logging.info(f"Loading local baseline model from: {model_path}")
                return load_object(model_path), model_path

        return None

    def _load_s3_baseline_model(self) -> Optional[Tuple[Any, str]]:
        bucket_name = self.model_eval_config.bucket_name
        model_key = self.model_eval_config.s3_model_key_path

        if not bucket_name or not model_key:
            return None

        try:
            estimator = AWSEstimator(bucket_name=bucket_name, model_key=model_key)
            if not estimator.is_model_present():
                logging.info(
                    f"No baseline model found at s3://{bucket_name}/{model_key}"
                )
                return None

            s3_object = estimator.s3.get_object(bucket_name=bucket_name, key=model_key)
            binary_data = estimator.s3.read_object(s3_object, decode=False)
            model = joblib.load(io.BytesIO(binary_data))
            s3_path = f"s3://{bucket_name}/{s3_object.key}"
            logging.info(f"Loaded baseline model from: {s3_path}")
            return model, s3_path
        except Exception as e:
            logging.warning(
                "AWS baseline model check/load failed. Continuing without baseline."
            )
            logging.warning(str(e))
            return None

    def get_best_model(self) -> Optional[Tuple[Any, str]]:
        """
        Load the current production/baseline model from local disk or S3.
        """
        try:
            local_model = self._load_local_baseline_model()
            if local_model is not None:
                return local_model

            return self._load_s3_baseline_model()
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self) -> Evaluate_Model_Response:
        try:
            x_test, y_test = self._load_test_data()
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            trained_model_score = self._model_score(trained_model, x_test, y_test)

            best_model_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                best_model_obj, best_model_path = best_model
                logging.info(f"Evaluating baseline model: {best_model_path}")
                best_model_score = self._model_score(best_model_obj, x_test, y_test)
            else:
                logging.info("No baseline model found. Accepting trained model by default.")

            baseline_score = 0.0 if best_model_score is None else best_model_score
            difference = trained_model_score - baseline_score
            threshold = float(self.model_eval_config.changed_threshold_score)
            is_model_accepted = best_model_score is None or difference >= threshold

            response = Evaluate_Model_Response(
                trained_model_score=trained_model_score,
                best_model_score=best_model_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )
            logging.info(f"Model evaluation response: {response}")
            return response
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> Model_Evaluation_Artifact:
        try:
            evaluation_response = self.evaluate_model()
            model_evaluation_artifact = Model_Evaluation_Artifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                changed_accuracy=evaluation_response.difference,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
