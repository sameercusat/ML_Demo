import os
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.metrics import r2_score
from src.utils import save_object,evaluate_models
import sys

@dataclass
class modelTrainerConfig:
    trained_model_path=os.path.join(r"C:\C++\ML_P","artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=modelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train array and test array")
            x_train,y_train,x_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])  
            models={
            "Random Forest":RandomForestRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Linear Regressor":LinearRegression(),
            "Ada Boost Regressor":AdaBoostRegressor(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            }
            model_reports:dict=evaluate_models(x_train,y_train,x_test,y_test,models)
            logging.info("Model Reports of all models are received")
            #print(model_reports)
            best_model_score=max(sorted(model_reports.values()))
            best_model_name=list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]
            best_model=models[best_model_name]
            if best_model_score <0.6:
                 logging.info("No best model found")
                 raise CustomException("No best model found")
            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)
            predicted_out=best_model.predict(x_test)
            score_r2=r2_score(y_test,predicted_out)
            logging.info(f"Best model {best_model_name} r2 score is {score_r2}")
            return "xxx"
            
        except Exception as e:
            raise CustomException(e,sys)

    