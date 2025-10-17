import os
import sys
from src.exception import CustomException
from src.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

class DataIngestionConfig:
    train_data_path = os.path.join("artifact", "train.csv")
    test_data_path = os.path.join("artifact", "test.csv")
    raw_data_path = os.path.join("artifact", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initiating DataIngestion...")
        try:
            df = pd.read_csv("C://Users//YASH//Desktop//DataScienceProjects//ML_Proj1//notebook//data//stud.csv")
            logger.info("Data read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Train-Test Split Initiated...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("Data Ingestion Completed...")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()