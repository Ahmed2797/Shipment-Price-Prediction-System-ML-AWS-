import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.constants import COLUMN_YAML_FILE_PATH, Target_Column
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml


class ShipmentPricePredictor:
    """A pipeline class to preprocess shipment data and predict shipment costs.

    This class handles the end-to-end inference pipeline, including data type 
    validation, cleaning, feature engineering, mathematical transformations, 
    and model prediction.

    Attributes:
        transform_object: A trained Sklearn-compliant transformer pipeline 
            (e.g., ColumnTransformer).
        trained_model (BaseEstimator): A trained Scikit-Learn estimator model.
        column_schema (dict): Schema configurations loaded from a YAML file.
    """

    def __init__(self, transform_object, trained_model: BaseEstimator):
        """Initializes the predictor with transformers, models, and schema configs.

        Args:
            transform_object: The pre-fitted transformation object/pipeline.
            trained_model (BaseEstimator): The pre-trained ML model object.

        Raises:
            CustomException: If any error occurs during initialization or 
                YAML reading.
        """
        try:
            self.transform_object = transform_object
            self.trained_model = trained_model
            self.column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def to_dataframe(data) -> pd.DataFrame:
        """Converts the input data into a pandas DataFrame format.

        Args:
            data (dict or pd.DataFrame): Input data to be converted.

        Returns:
            pd.DataFrame: A shallow copy of the dataframe or a new single-row dataframe.

        Raises:
            ValueError: If the input data is neither a dictionary nor a DataFrame.
        """
        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, pd.DataFrame):
            return data.copy()

        raise ValueError("Input must be a dict or a pandas DataFrame")

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans string columns by stripping whitespace and normalizing missing value placeholders.

        Args:
            df (pd.DataFrame): The input DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame with standard np.nan values for empty fields.
        """
        df = df.copy()
        object_columns = df.select_dtypes(include=["object"]).columns

        for col in object_columns:
            df[col] = df[col].astype(str).str.strip()
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

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives time-based features from 'Scheduled Date' and 'Delivery Date'.

        Extracts the scheduled month and calculates the total delivery days delta.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the newly engineered features.
        """
        df = df.copy()

        if "Scheduled Date" in df.columns:
            df["Scheduled Date"] = pd.to_datetime(df["Scheduled Date"], errors="coerce")
            df["Scheduled_Month"] = df["Scheduled Date"].dt.month

        if "Delivery Date" in df.columns:
            df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")

        if "Scheduled Date" in df.columns and "Delivery Date" in df.columns:
            df["Delivery_Days"] = (df["Delivery Date"] - df["Scheduled Date"]).dt.days

        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns specified in the column schema.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with designated columns dropped.
        """
        drop_cols = self.column_schema.get("drop_columns", [])
        return df.drop(columns=drop_cols, errors="ignore")

    def log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log1p transformation to specified highly skewed columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with log-transformed features.
        """
        df = df.copy()
        cols = self.column_schema.get("log_transform_col", [])

        for col in cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def power_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies a square root transformation to specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with square-root transformed features.
        """
        df = df.copy()
        cols = self.column_schema.get("power_transform_col", [])

        for col in cols:
            if col in df.columns:
                df[col] = np.sqrt(df[col].clip(lower=0))
        return df

    def prepare_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required columns exist in the DataFrame before transformation.

        Missing columns are initialized with np.nan to avoid structural errors.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame matching schema requirements.
        """
        df = df.copy()
        required_columns = []
        required_columns.extend(self.column_schema.get("numerical_columns", []))
        required_columns.extend(self.column_schema.get("multi_categorical_columns", []))

        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def preprocess(self, data) -> pd.DataFrame:
        """Executes the sequence of cleaning, engineering, and structural prep steps.

        Args:
            data (dict or pd.DataFrame): Raw payload data.

        Returns:
            pd.DataFrame: Completely processed DataFrame ready for transform object.
        """
        df = self.to_dataframe(data)
        df = self.clean_dataframe(df)
        df = self.feature_engineering(df)
        df = self.drop_columns(df)
        df = self.log_transform(df)
        df = self.power_transform(df)

        if Target_Column in df.columns:
            df = df.drop(columns=[Target_Column])

        df = self.prepare_columns(df)
        return df

    def predict(self, data) -> pd.DataFrame:
        """Generates cost predictions for bulk or batch input records.

        Preprocesses data, transforms features via the pipeline, generates 
        predictions, and converts the target log-scale output back to currency space.

        Args:
            data (dict or pd.DataFrame): Input batch records or single record.

        Returns:
            pd.DataFrame: DataFrame containing a "Predicted_Shipment_Cost" column.

        Raises:
            CustomException: Wraps any processing or model evaluation errors.
        """
        try:
            logging.info("Prediction started")
            df = self.preprocess(data)

            transformed_data = self.transform_object.transform(df)
            prediction = self.trained_model.predict(transformed_data)
            # prediction = np.array(prediction).flatten()
            
            # Convert back from log scale log1p -> expm1
            prediction = np.expm1(prediction)
            
            result = pd.DataFrame({"Predicted_Shipment_Cost": prediction})
            logging.info("Prediction completed")
            print("Prediction completed",result)
            return result

        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, data) -> float:
        """Generates a scalar cost prediction for a single instance payload.

        Args:
            data (dict or pd.DataFrame): A single record input data.

        Returns:
            float: The predicted shipment cost value.
        """
        # Call base predict method which transforms data and corrects log-scaling
        result = self.predict(data)
        
        # Extracted directly from DataFrame as a float
        return float(result.iloc[0, 0])