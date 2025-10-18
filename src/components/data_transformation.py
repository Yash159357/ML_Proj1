import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return the data transformer object with preprocessing steps.
        """
        try:
            logger.info("Creating data transformer object")
            
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            logger.info(f"Numerical features: {numerical_features}")
            logger.info(f"Categorical features: {categorical_features}")

            logger.info("Initializing numerical pipeline")
            numerical_pipeline = Pipeline(
                steps=[
                    ("median_imputation", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logger.info("Initializing categorical pipeline")
            categorical_pipeline = Pipeline(
                steps=[
                    ("mode_imputation", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),  # Changed to with_mean=False for sparse matrix
                ]
            )
            
            logger.info("Creating column transformer")
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", categorical_pipeline, categorical_features),
                ]
            )
            
            logger.info("Data transformer object created successfully")
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating data transformer object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation on training and test datasets.
        
        Args:
            train_path (str): Path to training data CSV file
            test_path (str): Path to test data CSV file
            
        Returns:
            tuple: (train_arr, test_arr, preprocessor_path)
        """
        try:
            logger.info("Starting data transformation process")
            
            # Load data
            logger.info(f"Loading training data from: {train_path}")
            train_df = pd.read_csv(train_path)
            logger.info(f"Training data loaded. Shape: {train_df.shape}")
            
            logger.info(f"Loading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            logger.info(f"Test data loaded. Shape: {test_df.shape}")
            
            # Get preprocessor
            logger.info("Getting data transformer object")
            preprocessor = self.get_data_transformer_object()
            
            target_column = "math_score"
            logger.info(f"Target column: {target_column}")
            
            # Split features and target
            logger.info("Splitting features and target")
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logger.info(f"Training features shape: {input_feature_train_df.shape}")
            logger.info(f"Test features shape: {input_feature_test_df.shape}")
            
            # Transform features
            logger.info("Fitting and transforming training data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logger.info("Transforming test data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            logger.info(f"Transformed training features shape: {input_feature_train_arr.shape}")
            logger.info(f"Transformed test features shape: {input_feature_test_arr.shape}")
            
            # Combine features and target
            logger.info("Combining features and target")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logger.info(f"Final training array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")
            
            # Save preprocessor
            save_path = self.data_transformation_config.preprocessor_obj_file_path
            logger.info(f"Saving preprocessor object to: {save_path}")
            save_object(
                file_path=save_path,
                obj=preprocessor
            )
            logger.info("Preprocessor saved successfully")
            
            logger.info("Data transformation completed successfully")
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logger.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)


