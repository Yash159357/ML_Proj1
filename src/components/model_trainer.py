import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    AdaBoostRegressor, 
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    trained_model_file_path: str = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    """Class for training and evaluating multiple regression models."""
    
    def __init__(self):
        """Initialize ModelTrainer with default configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, 
        train_arr: np.ndarray, 
        test_arr: np.ndarray, 
    ):
        """
        Train and evaluate multiple models, then save the best performing one.
        
        Args:
            train_arr: Training data array with features and target
            test_arr: Test data array with features and target
            preprocessor_path: Path to the saved preprocessor
            
        Returns:
            Tuple containing (best_model_name, best_test_score, best_train_score)
            
        Raises:
            CustomException: If any error occurs during model training
        """
        try:
            logger.info("Splitting training and test input data")
            x_train, x_test, y_train, y_test = (
                train_arr[:, :-1], 
                test_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, -1]
            )

            # Initialize models with random states for reproducibility
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(random_state=42, n_jobs=-1),
                "CatBoostRegressor": CatBoostRegressor(random_seed=42, verbose=0),
                "XGBRegressor": XGBRegressor(random_state=42, n_jobs=-1),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
                "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            }

            logger.info("Starting model evaluation...")
            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)
            
            # Log model performances
            logger.info("\n=== Model Evaluation Results ===")
            for model_name, (test_score, train_score) in model_report.items():
                logger.info(
                    f"{model_name:<25} | Test Score: {test_score:.4f} | "
                    f"Train Score: {train_score:.4f} | "
                    f"Diff: {abs(train_score - test_score):.4f}"
                )

            # Get best model based on test score
            best_model_name = max(model_report.items(), key=lambda x: x[1][0])[0]
            best_test_score, best_train_score = model_report[best_model_name]
            
            logger.info("\n=== Best Model ===")
            logger.info(f"Selected Model: {best_model_name}")
            logger.info(f"Test Score: {best_test_score:.4f}")
            logger.info(f"Train Score: {best_train_score:.4f}")
            logger.info(f"Train-Test Difference: {abs(best_train_score - best_test_score):.4f}")

            # Retrain best model on full training data
            logger.info("\nRetraining best model on full training data...")
            best_model = models[best_model_name]
            best_model.fit(
                np.vstack((x_train, x_test)),  # Combine train and test for final training
                np.concatenate((y_train, y_test))
            )
            
            # Save the trained model
            logger.info(f"Saving best model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logger.info("Model training and saving completed successfully!")

            pred = best_model.predict(x_test)

            r2_sqaure = r2_score(y_test, pred)


        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

