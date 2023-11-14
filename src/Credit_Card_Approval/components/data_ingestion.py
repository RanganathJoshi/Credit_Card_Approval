import pandas as pd
import numpy as np
from src.Credit_Card_Approval.logger import logging
from src.Credit_Card_Approval.exception import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
    data_path:str=os.path.join("artifacts",'raw.csv')
    train_path:str=os.path.join("artifacts",'train.csv')
    valid_path:str=os.path.join("artifacts",'valid.csv')


class Ingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion Initiated")
        try:
            data=pd.read_csv(Path(os.path.join('notebooks/data','data.csv')))
            data.drop(columns='Unnamed: 0',inplace=True)
            column_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','target']
            data.columns=column_names
            logging.info("Data Fetched")
            data['target']=data['target'].map({'+':1,'-':0})

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.data_path,index=False)

            logging.info("Uploaded data to the artifacts folder")
            logging.info("Splitting the dataset")
            train_data,valid_data=train_test_split(data,test_size=0.1,random_state=2)

            train_data.to_csv(self.ingestion_config.train_path,index=False)
            valid_data.to_csv(self.ingestion_config.valid_path,index=False)


            logging.info("Data Ingestion part completed")

            return (self.ingestion_config.train_path,self.ingestion_config.valid_path)

        except Exception as e:
            logging.info("Exception occured during data ingestion")
            raise customexception(e,sys)