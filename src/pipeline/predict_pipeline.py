import sys
import os
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            MODEL_PATH = os.path.join('artifact', 'model.pkl')
            PREPROCESSOR_PATH = os.path.join('artifact', 'preprocessor.pkl')

            model = load_object(MODEL_PATH)
            preprocessor = load_object(PREPROCESSOR_PATH)

            # Preprocess the input data
            processed_data = preprocessor.transform(features)

            # Make prediction
            prediction = model.predict(processed_data)[0]

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_DF(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info("Dataframe Gathered")
            return df

        except Exception as e:
            raise CustomException(e, sys)

