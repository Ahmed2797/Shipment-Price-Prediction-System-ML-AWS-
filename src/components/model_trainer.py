import sys
import os
import importlib
import optuna
import numpy as np
from optuna.pruners import MedianPruner
from typing import Tuple, List, Any, Optional, Dict

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import (
    Data_Transformation_Artifact,
    Model_Trainer_Artifact,
    ClassificationMetricArtifact,
    RegressionMetricArtifact
)
from src.entity.config_entity import Model_Trainer_Config
# from src.pipeline.prediction_pipeline import ShipmentPricePredictor
from src.constants import THRESHOLD
from src.utils import load_numpy_array, load_object, save_object, read_yaml

import mlflow
import mlflow.sklearn

import dagshub

# dagshub.init(repo_owner='Ahmed2797', repo_name='Shipment-Price-Prediction-System-ML-AWS-', mlflow=True)

class Model_Trainer:
    """
    Model_Trainer handles training, evaluating, selecting, saving, and logging ML models.

    Attributes:
        data_transformation_artifact (Data_Transformation_Artifact): Paths/objects from the data transformation stage.
        model_trainer_config (Model_Trainer_Config): Configuration for training, saving, and MLflow logging.
    """

    def __init__(self, 
                 data_transformation_artifact: Data_Transformation_Artifact,
                 model_trainer_config: Model_Trainer_Config):
        """
        Initialize Model_Trainer with required artifacts and configuration.

        Args:
            data_transformation_artifact (Data_Transformation_Artifact): Outputs from data transformation.
            model_trainer_config (Model_Trainer_Config): Configuration for training, saving, and logging.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def _build_metrics(model_name: str, model_obj, y_true, y_pred, is_regressor: bool = False) -> dict:
        """Build metrics for either regression or classification models."""
        if is_regressor:
            # Regression metrics
            return {
                "Model": model_name,
                "ModelObject": model_obj,
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "F1": r2_score(y_true, y_pred)  # Use R2 as F1 equivalent for model selection
            }
        else:
            # Classification metrics
            return {
                "Model": model_name,
                "ModelObject": model_obj,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "ClassificationReport": classification_report(y_true, y_pred, zero_division=0)
            }

    @staticmethod
    def _suggest_param(trial: optuna.trial.Trial, param_name: str, param_space: Any):
        """
        Convert YAML param space into Optuna suggestion.
        """
        if isinstance(param_space, list):
            if len(param_space) == 1:
                return param_space[0]
            return trial.suggest_categorical(param_name, param_space)

        if isinstance(param_space, dict):
            # Backward-compatible parsing:
            # 1) explicit `type`
            # 2) infer from low/high if `type` is missing
            # 3) categorical via `choices`
            if "type" in param_space:
                param_type = param_space.get("type")
            elif "choices" in param_space:
                param_type = "categorical"
            elif "low" in param_space and "high" in param_space:
                low, high = param_space["low"], param_space["high"]
                is_int_range = isinstance(low, int) and isinstance(high, int)
                param_type = "int" if is_int_range else "float"
            else:
                param_type = "categorical"

            if param_type == "int":
                return trial.suggest_int(
                    param_name,
                    int(param_space["low"]),
                    int(param_space["high"]),
                    step=int(param_space.get("step", 1)),
                    log=bool(param_space.get("log", False))
                )
            if param_type == "float":
                return trial.suggest_float(
                    param_name,
                    float(param_space["low"]),
                    float(param_space["high"]),
                    step=param_space.get("step"),
                    log=bool(param_space.get("log", False))
                )
            if param_type == "categorical":
                if "choices" not in param_space:
                    raise ValueError(
                        f"Missing 'choices' for categorical param '{param_name}'. "
                        "Provide `choices` or use `low`/`high`."
                    )
                return trial.suggest_categorical(param_name, param_space["choices"])
            raise ValueError(f"Unsupported Optuna param type '{param_type}' for '{param_name}'")

        return param_space

    def _tune_model_from_yaml(
        self,
        model_key: str,
        model_cfg: dict,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int,
        cv: int,
        scoring: str,
        direction: str
    ):
        model_module = importlib.import_module(model_cfg["module"])
        model_class = getattr(model_module, model_cfg["class"])
        fixed_params = model_cfg.get("params", {})
        search_space = model_cfg.get("search_param_grid", {})

        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

        def objective(trial: optuna.trial.Trial):
            trial_params = {
                param_name: self._suggest_param(trial, param_name, param_space)
                for param_name, param_space in search_space.items()
            }
            model_params = {**fixed_params, **trial_params}
            model = model_class(**model_params)
            scores = cross_val_score(
                model,
                x_train,
                y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1
            )
            return float(np.mean(scores))


        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = {**fixed_params, **study.best_params}
        best_model = model_class(**best_params)
        best_model.fit(x_train, y_train)
        logging.info(
            f"{model_key} ({model_cfg['class']}) tuned with Optuna | "
            f"best_cv_{scoring}: {study.best_value:.4f}"
        )
        return best_model, best_params, study.best_value

    def track_mlflow(
        self,
        model_path: str,
        model_obj,
        metrics: dict,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None
    ):
        """
        Log metrics and model to MLflow.

        Args:
            model_path (str): Path of the trained model file.
            model_obj: Trained model object to be logged with mlflow.sklearn.
            metrics (dict): Dictionary of numeric metrics to log.
            params (Optional[Dict[str, Any]]): Hyperparameters (e.g., Optuna best params).
            tags (Optional[Dict[str, Any]]): Run metadata for traceability.
            run_name (Optional[str]): Human-readable run name.

        Raises:
            CustomException: If MLflow logging fails.
        """
        try:
            # Use config values if present, otherwise skip setting URI/experiment
            tracking_uri = getattr(self.model_trainer_config, "mlflow_tracking_uri", None)
            experiment_name = getattr(self.model_trainer_config, "mlflow_experiment_name", None)

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=run_name) as run:
                # Log model hyperparameters (best params) in a MLflow-safe format
                if params:
                    safe_params = {}
                    for key, value in params.items():
                        if isinstance(value, (str, int, float, bool)):
                            safe_params[key] = value
                        else:
                            safe_params[key] = str(value)
                    mlflow.log_params(safe_params)

                # Log run tags for reproducibility/traceability
                if tags:
                    safe_tags = {str(k): str(v) for k, v in tags.items()}
                    mlflow.set_tags(safe_tags)

                # Log only numeric metrics
                for key, value in metrics.items():
                    try:
                        mlflow.log_metric(key, float(value))
                    except Exception:
                        logging.warning(f"Skipping non-numeric metric '{key}': {value}")

                # Log model artifact file and model object
                if model_path and os.path.exists(model_path):
                    mlflow.log_artifact(model_path)
                else:
                    logging.warning(f"Model artifact path does not exist: {model_path}")

                mlflow.sklearn.log_model(model_obj, artifact_path="model")
                logging.info(f"MLflow run logged successfully. run_id={run.info.run_id}")

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _predict_labels(model_obj, x_test: np.ndarray) -> np.ndarray:
        try:
            return model_obj.predict(x_test)
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_models_metrics(self, x_train, y_train, x_test, y_test) -> Tuple[List[dict], dict]:
        """
        Train all models from configuration, evaluate them, and select the best based on metric score.

        Args:
            x_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            x_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            Tuple[List[dict], dict]:
                - List of metrics for all trained models.
                - Best model result dictionary.

        Raises:
            CustomException: If training or evaluation fails.
        """
        try:
            logging.info("Loading model definitions from param.yaml and tuning with Optuna...")
            param_config = read_yaml(self.model_trainer_config.param_yaml)
            model_configs = param_config.get("model_selection", {})
            optuna_config = param_config.get("optuna", {})

            if not model_configs:
                raise CustomException("No models found under 'model_selection' in param.yaml", sys)

            n_trials = int(optuna_config.get("n_trials", 20))
            cv = int(optuna_config.get("cv", 3))
            scoring = optuna_config.get("scoring", "f1_weighted")
            direction = optuna_config.get("direction", "maximize")

            # Determine if this is a regression or classification task
            regression_metrics = {'r2', 'mse', 'mae', 'rmse', 'neg_mean_squared_error', 'neg_mean_absolute_error'}
            is_regressor = scoring.lower() in regression_metrics

            results = []
            errors = []

            for model_key, model_cfg in model_configs.items():
                try:
                    logging.info(f"Training model: {model_key} ({model_cfg.get('class', 'Unknown')})")
                    model, best_params, best_cv_score = self._tune_model_from_yaml(
                        model_key=model_key,
                        model_cfg=model_cfg,
                        x_train=x_train,
                        y_train=y_train,
                        n_trials=n_trials,
                        cv=cv,
                        scoring=scoring,
                        direction=direction
                    )
                    model_name = f"{model_cfg['class']}_Optuna"
                    y_pred = model.predict(x_test)
                    metrics = self._build_metrics(model_name, model, y_test, y_pred, is_regressor=is_regressor)
                    metrics["BestParams"] = best_params
                    metrics["BestCVScore"] = best_cv_score
                    metrics["Scoring"] = scoring
                    metrics["CV"] = cv
                    metrics["NTrials"] = n_trials
                    metrics["Direction"] = direction
                    results.append(metrics)
                    logging.info(f"✓ {model_key} trained successfully | Best CV Score: {best_cv_score:.4f}")
                except Exception as tune_error:
                    error_msg = f"Optuna tuning failed for {model_key}: {str(tune_error)}"
                    logging.error(error_msg)
                    errors.append((model_key, str(tune_error)))

            if not results:
                error_summary = "\n".join([f"  - {model}: {err}" for model, err in errors])
                raise CustomException(
                    f"Optuna failed for all models defined in param.yaml\nErrors:\n{error_summary}", 
                    sys
                )

            logging.info(f"✓ Optuna tuning completed for {len(results)}/{len(model_configs)} model(s).")

            # Select best model by F1 score (which equals R2 for regression models)
            best_result = max(results, key=lambda x: x["F1"])
            logging.info(f"Best Model: {best_result['Model']} | Score: {best_result['F1']:.4f}")
            return results, best_result

        except Exception as e:
            raise CustomException(e, sys)

    def init_model(self) -> Model_Trainer_Artifact:
        """
        Main method to:
            1. Train models
            2. Evaluate and pick the best
            3. Save the model and pipeline
            4. Optionally log metrics to MLflow

        Returns:
            Model_Trainer_Artifact: Contains paths and metrics of the trained model.

        Raises:
            CustomException: If training, evaluation, saving, or logging fails.
        """
        try:
            logging.info("Starting model training process...")

            # Load train/test arrays
            train_arr = load_numpy_array(self.data_transformation_artifact.transform_train_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transform_test_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train models & get metrics
            _, best_result = self.get_all_models_metrics(
                x_train, y_train, x_test, y_test
            )

            best_model_obj = best_result["ModelObject"]
            
            # Determine if this is regression or classification based on available metrics
            is_regression = "R2" in best_result  # Regression models have R2 metric
            
            if is_regression:
                best_model_metrics = {
                    "R2": best_result.get("R2"),
                    "MSE": best_result.get("MSE"),
                    "RMSE": best_result.get("RMSE"),
                    "MAE": best_result.get("MAE")
                }
            else:
                best_model_metrics = {
                    "Accuracy": best_result.get("Accuracy"),
                    "Precision": best_result.get("Precision"),
                    "Recall": best_result.get("Recall"),
                    "F1": best_result.get("F1")
                }
            
            logging.info(f"Best model metrics: {best_model_metrics}")

            # # Load preprocessing object
            # preprocessor_obj = load_object(self.data_transformation_artifact.preprocessing_pkl)

            # # Wrap model with preprocessing pipeline
            # prediction_model = ShipmentPricePredictor(
            #     transform_object=preprocessor_obj,
            #     trained_model=best_model_obj
            # )
            # logging.info("Prediction pipeline object created successfully.")
            
            # # Ensure directory for final model exists
            # os.makedirs(os.path.dirname(self.model_trainer_config.final_model_path), exist_ok=True)

            # # Save final pipeline and model
            # save_object(self.model_trainer_config.final_model_path, prediction_model)

            # Save raw model object separately
            best_model_path = os.path.join(self.model_trainer_config.model_train_dir, "best_model.pkl")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            save_object(best_model_path, best_model_obj)

            # Create metric artifact based on model type
            y_pred_best = self._predict_labels(best_model_obj, x_test)

            if is_regression:
                metrics_artifact = RegressionMetricArtifact(
                    r2_score=r2_score(y_test, y_pred_best),
                    mse=mean_squared_error(y_test, y_pred_best),
                    rmse=np.sqrt(mean_squared_error(y_test, y_pred_best)),
                    mae=mean_absolute_error(y_test, y_pred_best)
                )
                logging.info(f"Final metrics (Regression): R2={metrics_artifact.r2_score:.4f}, RMSE={metrics_artifact.rmse:.4f}")
                
                mlflow_metrics = {
                    "R2": metrics_artifact.r2_score,
                    "MSE": metrics_artifact.mse,
                    "RMSE": metrics_artifact.rmse,
                    "MAE": metrics_artifact.mae,
                    "BestCVScore": float(best_result.get("BestCVScore", 0.0))
                }
            else:
                metrics_artifact = ClassificationMetricArtifact(
                    accuracy_score=accuracy_score(y_test, y_pred_best),
                    f1_score=f1_score(y_test, y_pred_best, average='weighted', zero_division=0),
                    precision_score=precision_score(y_test, y_pred_best, average='weighted', zero_division=0),
                    recall_score=recall_score(y_test, y_pred_best, average='weighted', zero_division=0)
                )
                logging.info(f"Final metrics (Classification): Accuracy={metrics_artifact.accuracy_score:.4f}, F1={metrics_artifact.f1_score:.4f}")
                
                mlflow_metrics = {
                    "Accuracy": metrics_artifact.accuracy_score,
                    "Precision": metrics_artifact.precision_score,
                    "Recall": metrics_artifact.recall_score,
                    "F1": metrics_artifact.f1_score,
                    "BestCVScore": float(best_result.get("BestCVScore", 0.0))
                }

            # Track final model in MLflow with best hyperparameters and CV score
            # try:
            #     self.track_mlflow(
            #         model_path=best_model_path,
            #         model_obj=best_model_obj,
            #         metrics=mlflow_metrics,
            #         params=best_result.get("BestParams", {}),
            #         tags={
            #             "best_model_name": best_result.get("Model", "unknown"),
            #             "model_type": "regression" if is_regression else "classification",
            #             "cv_folds": str(best_result.get("CV", "unknown")),
            #             "optuna_scoring": best_result.get("Scoring", "unknown"),
            #             "optuna_trials": str(best_result.get("NTrials", "unknown")),
            #             "optuna_direction": best_result.get("Direction", "unknown")
            #         },
            #         run_name=f"model_trainer_{best_result.get('Model', 'best_model')}"
            #     )
            # except Exception as mlflow_error:
            #     logging.warning(f"MLflow tracking failed, but training artifacts are still saved: {mlflow_error}")
            
            logging.info("✓ Model training and saving completed successfully.")

            return Model_Trainer_Artifact(
                trained_model_file_path=self.model_trainer_config.final_model_path,
                metric_artifact=metrics_artifact
            )

        except Exception as e:
            raise CustomException(e, sys)
