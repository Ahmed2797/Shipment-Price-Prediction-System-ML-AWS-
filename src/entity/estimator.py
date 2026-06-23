import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.constants import COLUMN_YAML_FILE_PATH, Target_Column
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


class ShipmentPriceEstimator:
    """
    Production-ready execution pipeline wrapping ingestion, cleaning, 
    feature extraction, transformation, and model execution.
    """

    def __init__(self, transform_object, trained_model: BaseEstimator):
        self.transform_object = transform_object
        self.trained_model = trained_model
        try:
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
        except Exception as e:
            raise CustomException(
                f"Unable to load schema configuration for runtime inference: {e}",
                sys
            )

    @staticmethod
    def _to_dataframe(data) -> pd.DataFrame:
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, pd.DataFrame):
            return data.copy()
        raise ValueError("Input data payload must be a python dict or pandas DataFrame")

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        object_cols = df.select_dtypes(include=["object"]).columns

        # Vectorized string cleaning and standardization of missing indicators
        for col in object_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {"": np.nan, "na": np.nan, "nan": np.nan, "null": np.nan, "none": np.nan}
            )
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Convert temporal features safely to datetime objects
        if "Scheduled Date" in df.columns:
            df["Scheduled Date"] = pd.to_datetime(df["Scheduled Date"], errors="coerce")
            df["Scheduled_Month"] = df["Scheduled Date"].dt.month

        if "Delivery Date" in df.columns:
            df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")

        # 2. FIXED: Delivery Days calculation must be (End Date - Start Date)
        if "Scheduled Date" in df.columns and "Delivery Date" in df.columns:
            df["Delivery_Days"] = (df["Delivery Date"] - df["Scheduled Date"]).dt.days
            
            # Fill eventual negative dates or NaNs from broken data payloads with a baseline fallback
            df["Delivery_Days"] = df["Delivery_Days"].fillna(0).clip(lower=0)

        return df

    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self._column_schema.get("drop_columns", [])
        return df.drop(columns=drop_cols, errors="ignore")

    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_cols = self._column_schema.get("log_transform_col", [])
        for col in log_cols:
            if col in df.columns:
                # clip(lower=0) protects against negative values throwing RuntimeWarnings
                df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
        return df

    def _apply_power_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        power_cols = self._column_schema.get("power_transform_col", [])
        for col in power_cols:
            if col in df.columns:
                df[col] = np.power(pd.to_numeric(df[col], errors="coerce").clip(lower=0), 0.5)
        return df

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Construct exact operational schema matching what the transform_object expects
        expected_columns = []
        expected_columns.extend(self._column_schema.get("numerical_columns", []))
        expected_columns.extend(self._column_schema.get("multi_categorical_columns", []))

        df = df.copy()
        for col in expected_columns:
            if col not in df.columns:
                logging.warning(f"Expected column '{col}' missing from inference payload. Imputing NaN.")
                df[col] = np.nan

        # Enforce exact column ordering
        return df[expected_columns]

    def _prepare_input(self, data) -> pd.DataFrame:
        df = self._to_dataframe(data)
        df = self._clean_dataframe(df)
        df = self._feature_engineering(df)
        df = self._drop_unused_columns(df)
        df = self._apply_log_transform(df)
        df = self._apply_power_transform(df)

        if Target_Column in df.columns:
            df = df.drop(columns=[Target_Column])

        return self._ensure_required_columns(df)

    def predict(self, data) -> pd.DataFrame:
        try:
            logging.info("Starting shipment cost execution pipeline.")
            df = self._prepare_input(data)

            # Pass processed dataframe through Scikit-Learn pipeline object
            transformed_features = self.transform_object.transform(df)
            predictions = self.trained_model.predict(transformed_features)

            logging.info("Prediction vector generated successfully.")
            return pd.DataFrame({"Predicted_Shipment_Cost": predictions})

        except Exception as e:
            logging.error(f"Error encountered during prediction pipeline execution: {e}")
            raise CustomException(e, sys) from e

    def predict_single(self, data) -> float:
        """
        Runs inference over a singular dictionary record and returns a direct float value.
        """
        prediction_df = self.predict(data)
        return float(prediction_df["Predicted_Shipment_Cost"].iloc[0])