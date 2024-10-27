import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join(r"C:\C++\ML_P","artifacts","train.csv")
    test_data_path=os.path.join(r"C:\C++\ML_P","artifacts","test.csv")
    raw_data_path=os.path.join(r"C:\C++\ML_P","artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'C:\C++\ML_p\notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            #make three directories one for raw data , train data and test data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Split the train and test data")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)


if __name__ =='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    #print(train_data,test_data)
    data_trans_obj=DataTransformation()
    train_arr,test_arr,_=data_trans_obj.initiate_data_tranformation(train_data,test_data)
    model_trainer_obj=ModelTrainer()
    r_score=model_trainer_obj.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)


