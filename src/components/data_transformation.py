import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

from src.entity.config_entity import Data_Transformation_Config
from src.entity.artifact_entity import (
    Data_Ingestion_Artifact,
    Data_Validation_Artifact,
    Data_Transformation_Artifact
)
from src.constants import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    COLUMN_YAML_FILE_PATH,
    Target_Column
)
from src.utils import read_yaml, save_object, save_numpy_array
from src.exception import CustomException
from src.logger import logging


class TargetTransformer:
    """
    Transforms target variable using log1p and absolute value.
    Saves inverse transformation for predictions.
    
    Prevents data leakage by fitting ONLY on training data.
    """
    
    def __init__(self):
        self.is_fit = False
        
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """
        Fit and transform target: apply abs then log1p
        
        Parameters
        ----------
        y : pd.Series
            Target variable from training data
            
        Returns
        -------
        np.ndarray
            Transformed target values
        """
        # Apply transformations
        y_transformed = np.abs(y.values)
        y_transformed = np.log1p(y_transformed)
        self.is_fit = True
        return y_transformed
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """
        Transform target (test/validation data)
        
        Parameters
        ----------
        y : pd.Series
            Target variable from test/validation data
            
        Returns
        -------
        np.ndarray
            Transformed target values
        """
        if not self.is_fit:
            raise ValueError("TargetTransformer must be fit before transform")
        
        y_transformed = np.abs(y.values)
        y_transformed = np.log1p(y_transformed)
        return y_transformed
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale
        
        Parameters
        ----------
        y_transformed : np.ndarray
            Transformed predictions
            
        Returns
        -------
        np.ndarray
            Predictions in original scale
        """
        if not self.is_fit:
            raise ValueError("TargetTransformer must be fit before inverse_transform")
        
        # Reverse: expm1(log1p) = original, abs doesn't need reversal
        y_original = np.expm1(y_transformed)
        return y_original


class Data_Transformation:
    """
    Comprehensive data transformation pipeline for shipment price prediction.
    
    Features:
    - Prevents data leakage (fit preprocessor only on train)
    - Feature engineering with domain-specific features
    - Proper handling of numerical and categorical features
    - Target transformation with inverse capability
    - Saves all preprocessing objects for inference
    
    Parameters
    ----------
    data_transformation_config : Data_Transformation_Config
        Configuration object with artifact paths
    data_ingestion_artifact : Data_Ingestion_Artifact
        Paths to train/test CSV files
    data_validation_artifact : Data_Validation_Artifact
        Validation results
    """

    def __init__(self,
                 data_transformation_config: Data_Transformation_Config,
                 data_ingestion_artifact: Data_Ingestion_Artifact,
                 data_validation_artifact: Data_Validation_Artifact):
        """Initialize transformation pipeline."""
        try:
            self.transformation_config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
            self.validation_artifact = data_validation_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
            self.power_transformer = None
            self.target_transformer = TargetTransformer()
            
            logging.info("Data_Transformation initialized successfully")
            
        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================================
    # SECTION 1: DATA CLEANING & NORMALIZATION
    # =====================================================================
    
    def empty_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert empty strings to NaN for proper imputation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        """
        df_copy = df.copy()
        object_cols = df_copy.select_dtypes(include=["object"]).columns
        
        for col in object_cols:
            df_copy[col] = df_copy[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            df_copy[col] = df_copy[col].replace("", np.nan)
        
        return df_copy

    # =====================================================================
    # SECTION 2: FEATURE ENGINEERING
    # =====================================================================
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific engineered features.
        
        Features created:
        - Scheduled_Month: Month from scheduled date
        - Delivery_Days: Days between scheduled and delivery
        - Size: Height × Width (surface area proxy)
        - Price_per_Weight: Price normalized by weight
        - Price_per_Size: Price normalized by size
        - Weight_to_Size: Weight density ratio
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features
        """
        df = df.copy()
        
        # ---- Date features ----
        if "Scheduled Date" in df.columns:
            df["Scheduled Date"] = pd.to_datetime(df["Scheduled Date"], errors="coerce")
        if "Delivery Date" in df.columns:
            df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")
        
        if "Scheduled Date" in df.columns:
            df["Scheduled_Month"] = df["Scheduled Date"].dt.month
            df["Scheduled_Month"] = df["Scheduled_Month"].fillna(df["Scheduled_Month"].median())
        
        if "Scheduled Date" in df.columns and "Delivery Date" in df.columns:
            df["Delivery_Days"] = (df["Scheduled Date"] - df["Delivery Date"]).dt.days
            df["Delivery_Days"] = df["Delivery_Days"].fillna(df["Delivery_Days"].median())
            df["Delivery_Days"] = np.abs(df["Delivery_Days"])
        
        # # ---- Size features ----
        # if "Height" in df.columns and "Width" in df.columns:
        #     # Create size feature (surface area proxy)
        #     df["Size"] = df["Height"] * df["Width"]
        #     df["Size"] = df["Size"].clip(lower=0)  # Ensure non-negative
        #     df["Size"] = df["Size"].fillna(df["Size"].median())
        
        # # ---- Price ratio features ----
        # if "Price Of Sculpture" in df.columns and "Weight" in df.columns:
        #     # Avoid division by zero by adding 1
        #     df["Price_per_Weight"] = df["Price Of Sculpture"] / (df["Weight"].abs() + 1)
        #     df["Price_per_Weight"] = df["Price_per_Weight"].fillna(df["Price_per_Weight"].median())
        
        # if "Price Of Sculpture" in df.columns and "Size" in df.columns:
        #     df["Price_per_Size"] = df["Price Of Sculpture"] / (df["Size"] + 1)
        #     df["Price_per_Size"] = df["Price_per_Size"].fillna(df["Price_per_Size"].median())
        
        # # ---- Weight density features ----
        # if "Weight" in df.columns and "Size" in df.columns:
        #     df["Weight_to_Size"] = df["Weight"].abs() / (df["Size"] + 1)
        #     df["Weight_to_Size"] = df["Weight_to_Size"].fillna(df["Weight_to_Size"].median())
        
        logging.info(f"Feature engineering completed. New features: "
                    f"Scheduled_Month, Delivery_Days")
        
        return df

    # =====================================================================
    # SECTION 3: FEATURE TRANSFORMATIONS
    # =====================================================================
    
    def apply_log_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply log1p transformation to specified columns.
        Helps normalize skewed distributions.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : pd.DataFrame, optional
            Test dataframe
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Transformed train and test dataframes
        """
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
        
        log_cols = self._column_schema.get("log_transform_col", [])
        
        for col in log_cols:
            if col in train_df.columns:
                # Clip to avoid log of negative numbers
                train_df[col] = np.log1p(train_df[col].clip(lower=0))
                
                if test_df is not None and col in test_df.columns:
                    test_df[col] = np.log1p(test_df[col].clip(lower=0))
        
        return (train_df, test_df) if test_df is not None else (train_df, None)

    def apply_power_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply power transformation (Yeo-Johnson) to normalize distributions.
        FIT ONLY on train data, then TRANSFORM test data.
        Prevents data leakage.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : pd.DataFrame
            Test dataframe
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Power-transformed train and test dataframes
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        power_cols = self._column_schema.get("power_transform_col", [])
        
        # Only use columns that exist
        valid_cols = [col for col in power_cols if col in train_df.columns]

        if not valid_cols:
            logging.info("No columns to power transform")
            return train_df, test_df

        # ✅ FIT ONLY on training data (prevents leakage)
        self.power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        
        train_df[valid_cols] = self.power_transformer.fit_transform(train_df[valid_cols])
        
        # ✅ TRANSFORM test using fitted transformer
        test_df[valid_cols] = self.power_transformer.transform(test_df[valid_cols])

        logging.info(f"Power transformation applied to columns: {valid_cols}")
        return train_df, test_df

    # =====================================================================
    # SECTION 4: PREPROCESSING PIPELINE
    # =====================================================================
    
    def get_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline using ColumnTransformer.
        
        Pipeline:
        - Numerical: KNN imputation → StandardScaling
        - Categorical: Frequency imputation → OneHotEncoding
        
        Data leakage prevention:
        - Fit ONLY on training data
        - Transform test data using fitted parameters
        
        Returns
        -------
        ColumnTransformer
            Fitted preprocessing pipeline
        """
        numerical_cols = self._column_schema.get("numerical_columns", [])
        multi_cat_cols = self._column_schema.get("multi_categorical_columns", [])
        power_cols = self._column_schema.get("power_transform_col", [])

        # Numerical pipeline: KNN imputation + scaling
        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline: frequency imputation + one-hot encoding
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
        ])

        power_pipeline = Pipeline([
            ("power_transformer",PowerTransformer(method="yeo-johnson", standardize=False))
        ])



        # Combine into ColumnTransformer
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numerical_cols),
            # ("power", power_pipeline, power_cols),
            ("cat", categorical_pipeline, multi_cat_cols)
        ])

        logging.info(f"Preprocessor created with {len(numerical_cols)} numerical "
                    f"and {len(multi_cat_cols)} categorical columns and {len(power_cols)} power transformed columns.")
        
        return preprocessor

    # =====================================================================
    # SECTION 5: MAIN TRANSFORMATION ORCHESTRATION
    # =====================================================================
    
    def initiate_data_transformation(self) -> Data_Transformation_Artifact:
        """
        Execute complete data transformation pipeline with best practices:
        
        1. Load raw data
        2. Data cleaning (empty strings → NaN)
        3. Feature engineering
        4. Drop unnecessary columns
        5. Apply log transformation
        6. Apply power transformation (fit on train only)
        7. Separate target variable
        8. Create preprocessor and fit on train only
        9. Transform both train and test
        10. Save all artifacts
        
        Returns
        -------
        Data_Transformation_Artifact
            Contains paths to transformed arrays and preprocessor objects
        """
        try:
            logging.info("=" * 80)
            logging.info("STARTING DATA TRANSFORMATION PIPELINE")
            logging.info("=" * 80)
            
            # ---- STEP 1: Load raw data ----
            logging.info("Step 1: Loading raw data...")
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # ---- STEP 2: Clean empty strings ----
            logging.info("Step 2: Cleaning empty string columns...")
            train_df = self.empty_string_columns(train_df)
            test_df = self.empty_string_columns(test_df)

            # ---- STEP 3: Feature engineering ----
            logging.info("Step 3: Creating engineered features...")
            train_df = self.apply_feature_engineering(train_df)
            test_df = self.apply_feature_engineering(test_df)
            print(train_df.head())
            logging.info(f"Columns: {train_df.columns}")

            # ---- STEP 4: Drop unnecessary columns ----
            logging.info("Step 4: Dropping unnecessary columns...")
            drop_cols = self._column_schema.get("drop_columns", [])
            train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
            test_df.drop(columns=drop_cols, inplace=True, errors="ignore")
            logging.info(f"Dropped columns: {drop_cols}")

            # ---- STEP 5: Apply log transformation ----
            logging.info("Step 5: Applying log transformation...")
            train_df, test_df = self.apply_log_transform(train_df, test_df)

            # ---- STEP 6: Apply power transformation (FIT on train only) ----
            logging.info("Step 6: Applying power transformation (fit on train)...")
            train_df, test_df = self.apply_power_transform(train_df, test_df)

            # ---- STEP 7: Separate target variable ----
            logging.info("Step 7: Separating target variable...")
            X_train = train_df.drop(columns=[Target_Column])
            y_train = train_df[Target_Column]
            
            X_test = test_df.drop(columns=[Target_Column])
            y_test = test_df[Target_Column]

            # Verify train and test have same features
            assert set(X_train.columns) == set(X_test.columns), \
                "Train and test features don't match!"
            
            logging.info(f"Features shape - Train: {X_train.shape}, Test: {X_test.shape}")
            logging.info(f"Target shape - Train: {y_train.shape}, Test: {y_test.shape}")

            # ---- STEP 8: Transform target variable ----
            logging.info("Step 8: Transforming target (fit on train only)...")
            y_train_transformed = self.target_transformer.fit_transform(y_train)
            y_test_transformed = self.target_transformer.transform(y_test)
            
            logging.info(f"Target - Original range: [{y_train.min():.2f}, {y_train.max():.2f}]")
            logging.info(f"Target - Transformed range: [{y_train_transformed.min():.2f}, {y_train_transformed.max():.2f}]")

            # ---- STEP 9: Create and fit preprocessor (on train only) ----
            logging.info("Step 9: Creating and fitting preprocessor (on train only)...")

            # Use only columns that actually exist in X_train to avoid sklearn errors
            numerical_cols = [
                c for c in self._column_schema.get("numerical_columns", []) if c in X_train.columns
            ]
            multi_cat_cols = [
                c for c in self._column_schema.get("multi_categorical_columns", []) if c in X_train.columns
            ]

            transformers = []
            if numerical_cols:
                numeric_pipeline = Pipeline([
                    ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                    ("scaler", StandardScaler())
                ])
                transformers.append(("num", numeric_pipeline, numerical_cols))

            if multi_cat_cols:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
                ])
                transformers.append(("cat", categorical_pipeline, multi_cat_cols))

            if not transformers:
                raise CustomException("No valid numeric or categorical columns found for preprocessing", sys)

            preprocessor_local = ColumnTransformer(transformers, remainder="drop")

            preprocessor = self.get_preprocessor()

            # ✅ FIT ONLY on train data (prevents leakage)
            X_train_transformed = preprocessor.fit_transform(X_train)

            # ✅ TRANSFORM test using fitted preprocessor
            X_test_transformed = preprocessor.transform(X_test)

            # Convert to DataFrame for better tracking
            feature_names = preprocessor.get_feature_names_out()
            X_train_transformed = pd.DataFrame(
                X_train_transformed,
                columns=feature_names,
                index=X_train.index
            )
            X_test_transformed = pd.DataFrame(
                X_test_transformed,
                columns=feature_names,
                index=X_test.index
            )
            
            logging.info(f"Preprocessing complete - Features: {X_train_transformed.shape[1]}")
            logging.info(f"Missing values - Train: {X_train_transformed.isnull().sum().sum()}, "
                        f"Test: {X_test_transformed.isnull().sum().sum()}")

            # ---- STEP 10: Combine features and target ----
            logging.info("Step 10: Combining features and transformed target...")
            train_arr = np.c_[X_train_transformed, y_train_transformed]
            test_arr = np.c_[X_test_transformed, y_test_transformed]

            # ---- STEP 11: Save all artifacts ----
            logging.info("Step 11: Saving artifacts...")
            
            # Save transformed arrays
            save_numpy_array(self.transformation_config.transform_train_path, train_arr)
            save_numpy_array(self.transformation_config.transform_test_path, test_arr)
            
            # Save preprocessor
            save_object(self.transformation_config.transform_object_path, preprocessor)
            
            # Save target transformer (for inverse transformation during inference)
            target_transformer_path = os.path.join(
                os.path.dirname(self.transformation_config.transform_object_path),
                "target_transformer.pkl"
            )
            save_object(target_transformer_path, self.target_transformer)
            
            # # Save power transformer (if exists)
            # if self.power_transformer is not None:
            #     power_transformer_path = os.path.join(
            #         os.path.dirname(self.transformation_config.transform_object_path),
            #         "power_transformer.pkl"
            #     )
            #     save_object(power_transformer_path, self.power_transformer)

            logging.info("=" * 80)
            logging.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            logging.info(f"Train array: {train_arr.shape}")
            logging.info(f"Test array: {test_arr.shape}")
            logging.info(f"Artifacts saved to: {self.transformation_config.transform_object_path}")

            return Data_Transformation_Artifact(
                transform_train_path=self.transformation_config.transform_train_path,
                transform_test_path=self.transformation_config.transform_test_path,
                preprocessing_pkl=self.transformation_config.transform_object_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(f"Error in initiate_data_transformation: {e}", sys)
