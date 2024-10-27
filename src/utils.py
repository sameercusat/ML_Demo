import sys
import numpy as np
import os
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}

        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            train_pred=model.predict(x_train)
            test_pred=model.predict(x_test)
            r2_train=r2_score(y_train,train_pred)
            r2_test=r2_score(y_test,test_pred)
            report[list(models.keys())[i]]=r2_test
        return report    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
         with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

    
