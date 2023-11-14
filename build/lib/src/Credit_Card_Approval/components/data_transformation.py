import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.Credit_Card_Approval.exception import customexception
from src.Credit_Card_Approval.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.Credit_Card_Approval.utils.utils import save_object

class DataTransformConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts",'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()

    def prepare_data_transformation(self):
        try:
            logging.info("Starting data transform")
            num_cols=['A', 'B', 'E', 'H', 'M', 'N']
            cat_cols=['C', 'D', 'F', 'G', 'I', 'J', 'K', 'L', 'O']
            categories=[['g', 's', 'p'],['f', 't'],['t', 'f'],['t', 'f'],['v', 'h', 'bb', 'ff', 'j', 'z', 'o', 'dd', 'n'],['w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff', 'j'],['g', 'p', 'gg'],['u', 'y', 'l'],['b', 'a']]
            logging.info("Initiating Pipeline")

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('standardscaling',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=categories)),
                    ('scalar',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
                ('numericalpipeline',num_pipeline,num_cols),
                ('categorical columns',cat_pipeline,cat_cols)
            ],remainder='passthrough')


            return preprocessor
        
        except Exception as e:
            logging.info("Error occured while data transformation")
            raise customexception(e,sys)
        
    def initialize_data_transformation(self,train_path,valid_path):
        try:
            train_df=pd.read_csv(train_path)
            valid_df=pd.read_csv(valid_path)

            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head()}')
            logging.info(f'Valid Dataframe Head : \n{valid_df.head()}')

            processor_obj=self.prepare_data_transformation()

            input_feature_train_df=train_df.drop(columns='target',axis=1)
            target_feature_train_df=train_df['target']

            input_feature_valid_df=valid_df.drop(columns='target',axis=1)
            target_feature_valid_df=valid_df['target']

            input_feature_train_arr=processor_obj.fit_transform(input_feature_train_df)

            input_feature_valid_arr=processor_obj.transform(input_feature_valid_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            valid_arr=np.c_[input_feature_valid_arr,np.array(target_feature_valid_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processor_obj
            )

            logging.info(train_arr[:5,:])
            logging.info(valid_arr[:5,:])

            return (train_arr,valid_arr)
        
        except Exception as e:
            logging.info("error occured while data transformation")
            raise customexception(e,sys)