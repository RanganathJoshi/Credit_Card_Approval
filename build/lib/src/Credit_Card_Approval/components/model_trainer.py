import pandas as pd
import numpy as np
import os
import sys
from src.Credit_Card_Approval.logger import logging
from src.Credit_Card_Approval.exception import customexception
from dataclasses import dataclass
from src.Credit_Card_Approval.utils.utils import save_object
from src.Credit_Card_Approval.utils.utils import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_path:str=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dataset into independent and dependent features")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={
                'LogisticRegression':LogisticRegression(),
                'KNN':KNeighborsClassifier(),
                'SVM':SVC(),
            }
            model_report=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print("\n=========================================================================")
            logging.info(f"Model Report : {model_report}")
            best_model_score=max(model_report.values())
            best_model_name= best_model_name=list(model_report.keys())[np.argmax(list(model_report.values()))]
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            best_model=models[best_model_name]   
            save_object(file_path=self.model_trainer.trained_model_path,
                        obj=best_model)

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise customexception(e,sys)   

