import psycopg2
import pandas as pd
from src.logger import get_logger
import os
from src.custom_exception import CustomException
import sys
from sklearn.model_selection import train_test_split

from config.paths_config import *
from config.database_config import *

logger = get_logger(__name__)

class DataIngestion:
    
    def __init__(self,db_params , output_dir):
        self.db_params = db_params
        self.output_dir = output_dir

        os.makedirs(self.output_dir ,exist_ok=True)
    
    def connect_to_db(self):
        try:
            conn = psycopg2.connect(
                **self.db_params
            )

            logger.info("Database connection established sucessfully ..")
            return conn

        except Exception as e:
            logger.error(f"Error during DB connection {e}")
            raise CustomException("Error during DB connection",e)
    
    def extract_data(self):
        try:
            conn = self.connect_to_db()
            query = "SELECT * FROM public.titanic"
            df = pd.read_sql_query(query,conn)
            conn.close()
            logger.info("Data extracted sucessfully")
            return df
        
        except Exception as e:
            logger.error(f"Error during data loading {e}")
            raise CustomException("Error during data loading",e)
    
    def save_data(self,df):
        try:
            train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)
            train_df.to_csv(TRAIN_PATH,index=False)
            test_df.to_csv(TEST_PATH,index=False)

            logger.info("Data Spliting and saving completed")
        
        except Exception as e:
            logger.error(f"Error during data spliting and saving {e}")
            raise CustomException("Error during data spliting and saving ",e)
    
    def run(self):
        try:
            logger.info("Data Ingestion Pipeline Started...")
            df = self.extract_data()
            self.save_data(df)
            logger.info("Data Ingestion Pipeline completed")
        except Exception as e:
            logger.error(f"Error during data ingestion pipeline {e}")
            raise CustomException("Error during data ingestion pipeline ",e)
    
if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
    data_ingestion.run()