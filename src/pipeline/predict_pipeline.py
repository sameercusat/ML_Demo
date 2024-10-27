import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model=load_object(r'C:\C++\ML_p\artifacts\model.pkl')
            preprocessor=load_object(r'C:\C++\ML_p\artifacts\preprocessor.pkl')
            data_scaled=preprocessor.transform(feature)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

#Class for mapping data of FE and BE
class CustomData:
    def __init__(self,gender,race,parent_education,lunch,test_course,writing_score,reading_score):
        self.gender=gender
        self.race=race
        self.parent_education=parent_education
        self.lunch=lunch
        self.test_course=test_course
        self.writing_score=writing_score
        self.reading_score=reading_score

    def get_data_as_dataframe(self):
        try:
            create_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race],
                "parental_level_of_education":[self.parent_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_course],
                "writing_score":[self.writing_score],
                "reading_score":[self.reading_score]
            }
            return pd.DataFrame(create_dict)
        except Exception as e:
            raise CustomException(e,sys)    
    