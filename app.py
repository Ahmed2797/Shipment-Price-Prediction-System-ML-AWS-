
from src.pipeline.training_pipeline import Training_Pipeline 
from src.exception import CustomException 
from src.logger import logging
import sys

if __name__ == '__main__':
    try:
        logging.info('Started Training_Pipeline.......')
        pipeline = Training_Pipeline()
        pipeline.run_pipeline()
        logging.info('Training_Pipeline Completed')

    except Exception as e:
        raise CustomException (e,sys)


