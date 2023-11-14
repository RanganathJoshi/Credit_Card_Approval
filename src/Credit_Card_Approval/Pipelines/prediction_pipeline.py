import os
import sys
import pandas as pd
from src.Credit_Card_Approval.exception import customexception
from src.Credit_Card_Approval.logger import logging
from src.Credit_Card_Approval.utils.utils import load_object

class predictpipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            preprocessor_path=os.path.join("artifacts",'preprocessor.pkl')
            model_path=os.path.join("artifacts",'model.pkl')
            processor=load_object(preprocessor_path)
            model=load_object(model_path)
            trans_data=processor.transform(feature)
            pred=model.predict(trans_data)

            return pred

        except Exception as e:
            logging.info("Error occured while predicting")
        


class customdata:
    def __init__(self,A:str,B:str,C:str,D:str,E:str,F:str,G:str,H:str,I:str,J:str,K:str,L:str,M:str,N:str,O:str):
        self.A=A,
        self.B=B,
        self.C=C,
        self.D=D,
        self.E=E,
        self.F=F,
        self.G=G,
        self.H=H,
        self.I=I,
        self.J=J,
        self.K=K,
        self.L=L,
        self.M=M,
        self.N=N,
        self.O=O

    def get_custom_data(self):
        
        custom_data={
            'A':self.A,
            'B':self.B,
            'C':self.C,
            'D':self.D,
            'E':self.E,
            'F':self.F,
            'G':self.G,
            'H':self.H,
            'I':self.I,
            'J':self.J,
            'K':self.K,
            'L':self.L,
            'M':self.M,
            'N':self.N,
            'O':self.O
        }
        df=pd.DataFrame(custom_data)
        logging.info("DataFrame created")
        return df
    