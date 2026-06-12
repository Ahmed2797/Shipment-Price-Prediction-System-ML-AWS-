import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.constants import COLUMN_YAML_FILE_PATH, Target_Column
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


class ShipmentPricePredictor:
    """
    Production-ready prediction class for Shipment Price Prediction.

    Features:
    ----------
    - Accept dict or DataFrame
    - Automatic data cleaning
    - Feature engineering
    - Uses saved preprocessing object
    - Uses trained model
    - Returns clean prediction DataFrame
    """

    def __init__(
        self,
        transform_object,
        trained_model: BaseEstimator
    ):
        self.transform_object = transform_object
        self.trained_model = trained_model
        try:
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
        except Exception as e:
            raise CustomException(
                f"Unable to load schema for prediction preprocessing: {e}",
                sys
            )

    @staticmethod
    def _to_dataframe(data):

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, pd.DataFrame):
            return data.copy()

        raise ValueError(
            "Input must be dict or pandas DataFrame"
        )

    @staticmethod
    def _clean_dataframe(df):

        df = df.copy()

        object_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in object_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

            df[col] = df[col].replace(
                {
                    "": np.nan,
                    "na": np.nan,
                    "nan": np.nan,
                    "null": np.nan,
                    "none": np.nan
                }
            )

        return df

    @staticmethod
    def _feature_engineering(df):

        df = df.copy()

        if "Scheduled Date" in df.columns:
            df["Scheduled Date"] = pd.to_datetime(
                df["Scheduled Date"],
                errors="coerce"
            )

        if "Delivery Date" in df.columns:
            df["Delivery Date"] = pd.to_datetime(
                df["Delivery Date"],
                errors="coerce"
            )

        if "Scheduled Date" in df.columns:

            df["Scheduled_Month"] = (
                df["Scheduled Date"]
                .dt.month
            )

        if (
            "Scheduled Date" in df.columns
            and
            "Delivery Date" in df.columns
        ):

            df["Delivery_Days"] = (
                df["Scheduled Date"]
                -
                df["Delivery Date"]
            ).dt.days

        return df

    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self._column_schema.get("drop_columns", [])
        return df.drop(columns=drop_cols, errors="ignore")

    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        log_cols = self._column_schema.get("log_transform_col", [])
        for col in log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def _apply_power_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        power_cols = self._column_schema.get("power_transform_col", [])
        for col in power_cols:
            if col in df.columns:
                df[col] = np.power(df[col].clip(lower=0).astype(float), 0.5)
        return df

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = []
        expected_columns.extend(self._column_schema.get("numerical_columns", []))
        expected_columns.extend(self._column_schema.get("multi_categorical_columns", []))

        df = df.copy()
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df[expected_columns]

    def _prepare_input(self, data):
        df = self._to_dataframe(data)
        df = self._clean_dataframe(df)
        df = self._feature_engineering(df)
        df = self._drop_unused_columns(df)
        df = self._apply_log_transform(df)
        df = self._apply_power_transform(df)

        if Target_Column in df.columns:
            df = df.drop(columns=[Target_Column])

        df = self._ensure_required_columns(df)
        return df

    def predict(self, data):

        try:

            logging.info(
                "Starting shipment cost prediction"
            )

            df = self._prepare_input(data)

            transformed_features = (
                self.transform_object.transform(df)
            )

            predictions = (
                self.trained_model.predict(
                    transformed_features
                )
            )

            prediction_df = pd.DataFrame(
                {
                    "Predicted_Shipment_Cost":
                    predictions
                }
            )

            logging.info(
                "Prediction completed successfully"
            )

            return prediction_df

        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, data):

        prediction = self.predict(data)

        return float(
            prediction[
                "Predicted_Shipment_Cost"
            ].iloc[0]
        )