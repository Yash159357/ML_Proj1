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
        tune_hyperparams: bool = True
    ):
        """
        Train and evaluate multiple models with optional hyperparameter tuning,
        then save the best performing one.
        
        Args:
            train_arr: Training data array with features and target
            test_arr: Test data array with features and target
            tune_hyperparams: Whether to perform hyperparameter tuning (default: True)
            
        Returns:
            Tuple containing (best_model_name, best_test_score, best_train_score, best_params)
            
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
            MODEL_PARAMS = {
                'LinearRegression': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                'RandomForestRegressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'CatBoostRegressor': {
                    'iterations': [100, 200],
                    'depth': [4, 6, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5, 7]
                },
                'XGBRegressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4]
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1: manhattan, 2: euclidean
                },
                'DecisionTreeRegressor': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            }

            logger.info("Starting model evaluation with hyperparameter tuning...")
            model_report, best_params = evaluate_models(
                x_train, y_train, x_test, y_test, 
                models, tune_hyperparams=tune_hyperparams, MODEL_PARAMS=MODEL_PARAMS
            )
            
            # Log model performances
            logger.info("\n=== Model Evaluation Results ===")
            for model_name, (test_score, train_score) in model_report.items():
                logger.info(
                    f"{model_name:<25} | Test Score: {test_score:.4f} | "
                    f"Train Score: {train_score:.4f} | "
                    f"Diff: {abs(train_score - test_score):.4f}"
                )
                if tune_hyperparams and model_name in best_params:
                    logger.info(f"Best params for {model_name}: {best_params[model_name]}")

            # Get best model based on test score
            best_model_name = max(model_report.items(), key=lambda x: x[1][0])[0]
            best_test_score, best_train_score = model_report[best_model_name]
            
            logger.info("\n=== Best Model ===")
            logger.info(f"Selected Model: {best_model_name}")
            logger.info(f"Test Score: {best_test_score:.4f}")
            logger.info(f"Train Score: {best_train_score:.4f}")
            logger.info(f"Train-Test Difference: {abs(best_train_score - best_test_score):.4f}")
            
            if tune_hyperparams and best_model_name in best_params:
                logger.info(f"Best parameters used: {best_params[best_model_name]}")

            # Retrain best model on full training data with best parameters
            logger.info("\nRetraining best model on full training data...")
            best_model = models[best_model_name]
            
            # If we did hyperparameter tuning, we need to reinitialize the model with best params
            if tune_hyperparams and best_model_name in best_params:
                best_model = best_model.set_params(**best_params[best_model_name])
                
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
            
            # Save the best parameters
            params_file = os.path.join(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                'best_params.pkl'
            )
            save_object(params_file, best_params)
            
            logger.info("Model training and saving completed successfully!")

            # Calculate final R2 score on test set
            pred = best_model.predict(x_test)
            r2_square = r2_score(y_test, pred)
            logger.info(f"Final R2 score on test set: {r2_square:.4f}")
            
            return best_model_name, best_test_score, best_train_score, best_params

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)
    
    # Original implementation without hyperparameter tuning (kept for reference)
    """
    def initiate_model_trainer(self, train_arr: np.ndarray, test_arr: np.ndarray):
        try:
            logger.info("Splitting training and test input data")
            x_train, x_test, y_train, y_test = (
                train_arr[:, :-1], 
                test_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, -1]
            )

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
            model_report, _ = evaluate_models(x_train, y_train, x_test, y_test, models, tune_hyperparams=False)
            
            logger.info("\n=== Model Evaluation Results ===")
            for model_name, (test_score, train_score) in model_report.items():
                logger.info(
                    f"{model_name:<25} | Test Score: {test_score:.4f} | "
                    f"Train Score: {train_score:.4f} | "
                    f"Diff: {abs(train_score - test_score):.4f}"
                )

            best_model_name = max(model_report.items(), key=lambda x: x[1][0])[0]
            best_test_score, best_train_score = model_report[best_model_name]
            
            logger.info("\n=== Best Model ===")
            logger.info(f"Selected Model: {best_model_name}")
            logger.info(f"Test Score: {best_test_score:.4f}")
            logger.info(f"Train Score: {best_train_score:.4f}")
            logger.info(f"Train-Test Difference: {abs(best_train_score - best_test_score):.4f}")

            logger.info("\nRetraining best model on full training data...")
            best_model = models[best_model_name]
            best_model.fit(
                np.vstack((x_train, x_test)),
                np.concatenate((y_train, y_test))
            )
            
            logger.info(f"Saving best model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logger.info("Model training and saving completed successfully!")
            
            pred = best_model.predict(x_test)
            r2_square = r2_score(y_test, pred)
            
            return best_model_name, best_test_score, best_train_score, {}

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)
    """

