from dataclasses import dataclass 
from typing import List
from src.constants import *


@dataclass 
class Data_Ingestion_Artifact:
    train_file_path:str 
    test_file_path:str