import numpy as np
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import os
#this class helpful in providing the necessary paths for the output pickle file which we produce as output of this program
@dataclass
class DataTransformmationConfig:
    preprocessor_obj_file_path=os.path.join(r"C:\C++\ML_P","artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformmationConfig()

    def get_data_tranformer_obj(self):
        ''' This function is used for data transormation '''
        try:
            numerical_features=[ 'reading_score', 'writing_score']
            categorical_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy='median')),
                                         ("scaler",StandardScaler())])
            categorical_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy='most_frequent')),("one_hot_encoder",OneHotEncoder()),("scaler",StandardScaler(with_mean=False))])
            logging.info("Categorical Features",categorical_features)
            logging.info("Numerical Features",numerical_features)
            preprocessor=ColumnTransformer([("num_transformer",num_pipeline,numerical_features),("categ_tranformer",categorical_pipeline,categorical_features)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            #print(train_df.head(10))
            test_df=pd.read_csv(test_path)
            #print(test_df.head(10))
            logging.info("Reading of test and train dataframe completed")
            preprocessing_obj=self.get_data_tranformer_obj()
            logging.info("Obtaining preprocessing object")
              
            target_column_name='math_score'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            #print(input_feature_train_df.head(10))
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applyning preprocessing object on train and test input features")

            input_features_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_features_train_arr,target_feature_train_df]
            test_arr=np.c_[input_features_test_arr,target_feature_test_df]

            logging.info("Saved preprocessing objects")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)


            