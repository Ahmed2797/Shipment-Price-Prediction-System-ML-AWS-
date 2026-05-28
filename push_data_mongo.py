from src.exception import CustomException  
from src.logger import logging  
from src.constants import MONGODB_URL_KEY,Data_Base_Name,Collection_name 
from dotenv import load_dotenv  
from pathlib import Path
import pandas as pd  
import pymongo  
import certifi  
import json  
import sys  
import os  

# Load environment variables
load_dotenv()

ca = certifi.where()
MONGODB_URL = os.getenv(MONGODB_URL_KEY)

class Data_Insert_Mongo:

    def __init__(self):
        try:
            self.mongo_client = None
            logging.info("Data_Insert_Mongo class initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def csv_to_json(self, file_path):
        """Convert CSV file to JSON records."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError("CSV file not found at given path")

            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = json.loads(data.to_json(orient="records"))

            logging.info(f"Successfully converted {len(records)} records from CSV to JSON")
            return records

        except Exception as e:
            logging.error("Error converting CSV to JSON")
            raise CustomException(e, sys)

    def insert_data_into_mongodb(self, records, database, collection):
        """Insert JSON records into MongoDB."""
        try:
            if not records:
                logging.warning("Records list is empty")
                return 0

            if not database or not collection:
                raise ValueError("Database and Collection names cannot be empty")

            self.mongo_client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=ca)
            db = self.mongo_client[database]
            coll = db[collection]

            result = coll.insert_many(records)
            inserted_count = len(result.inserted_ids)

            logging.info(f"Successfully inserted {inserted_count} records into {database}.{collection}")
            return inserted_count

        except Exception as e:
            logging.error("Error inserting data into MongoDB")
            raise CustomException(e, sys)

        finally:
            if self.mongo_client:
                self.mongo_client.close()
                logging.info("MongoDB connection closed.")


if __name__=='__main__':
    try:
        data_path = Path.cwd() / "Data" / "Shipment-data.csv"
        # data_path = os.path.join(os.getcwd(), 'Data', 'Shipment-data.csv')
        database_name = Data_Base_Name
        collection =  Collection_name   

        data_obj = Data_Insert_Mongo()  
        records = data_obj.csv_to_json(data_path)
        print(len(records))
        record_count = data_obj.insert_data_into_mongodb(records,database_name,collection)
        print(f'Inserted no_of_the_records {record_count} in MongoDB')


    except Exception as e:
        raise CustomException (e,sys)  
          

## python push_data_mongo.py