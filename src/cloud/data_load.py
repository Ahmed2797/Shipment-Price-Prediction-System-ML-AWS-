from src.configuration.mongodb import MongoDBClient    
from src.logger import logging                         
from src.exception import CustomException               
from typing import Optional
import pandas as pd
import numpy as np
import sys


class Data_Extract:
    """
    This class is responsible for extracting data from MongoDB collections
    and converting them into Pandas DataFrames for further processing.
    """

    def __init__(self):
        """
        Initialize the Data_Extract class and connect to MongoDB using MongoDBClient.
        """
        try:
            # Initialize MongoDB client connection
            self.mongo_client = MongoDBClient()
            logging.info("MongoDBClient initialized successfully in Data_Extract.")

        except Exception as e:
            logging.error("Error initializing MongoDBClient in Data_Extract.")
            raise CustomException(e, sys)


    def get_dataframe(self, collection: str, database: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts data from a MongoDB collection and converts it into a Pandas DataFrame.

        Args:
            collection (str): The name of the MongoDB collection.
            database (Optional[str]): The database name (optional). 
                                      If not provided, the default database from MongoDBClient is used.

        Returns:
            pd.DataFrame: A cleaned DataFrame containing data from the MongoDB collection.
        """
        try:
            # Select database and collection
            if database:
                # If a specific database is provided
                collection = self.mongo_client[database][collection]
                logging.info(f"Extracting data from specified database: {database}, collection: {collection}.")
            else:
                # If no database name is passed, use the default one
                collection = self.mongo_client.database[collection]
                logging.info(f"Extracting data from default database, collection: {collection}.")

            # Convert MongoDB documents to DataFrame
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"{len(df)} records fetched successfully from MongoDB.")

            # Drop '_id' column as it’s not useful for analysis
            if '_id' in df.columns.to_list():
                df.drop('_id', axis=1, inplace=True)
                logging.info("Removed '_id' column from DataFrame.")

            # Replace 'na' strings with actual NaN values
            df.replace('na', np.nan, inplace=True)

            logging.info("Data extraction and DataFrame creation successful.")
            print(df.head())
            return df

        except Exception as e:
            logging.error("Error while extracting data from MongoDB or converting to DataFrame.")
            raise CustomException(e, sys)
        



