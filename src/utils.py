import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logger

# Parameter grids for hyperparameter tuning

def save_object(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)

def tune_hyperparameters(model, param_grid, X, y, cv=3, scoring='r2', n_jobs=-1):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        model: The model to tune
        param_grid: Dictionary of parameters to tune
        X: Training features
        y: Training target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        Best estimator and best parameters
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_models(x_train, y_train, x_test, y_test, models, MODEL_PARAMS, tune_hyperparams=False):
    """
    Evaluate multiple models and optionally perform hyperparameter tuning.
    
    Args:
        x_train: Training features
        y_train: Training target
        x_test: Test features
        y_test: Test target
        models: Dictionary of models to evaluate
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with model names as keys and (test_score, train_score) as values
    """
    try:
        report = {}
        best_params = {}
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            # Skip hyperparameter tuning if not requested or no parameters defined
            if not tune_hyperparams or model_name not in MODEL_PARAMS:
                model.fit(x_train, y_train)
                best_params[model_name] = "Default parameters"
            else:
                logger.info(f"Tuning hyperparameters for {model_name}...")
                best_model, params = tune_hyperparameters(
                    model=model,
                    param_grid=MODEL_PARAMS[model_name],
                    X=x_train,
                    y=y_train
                )
                model = best_model
                best_params[model_name] = params
                logger.info(f"Best parameters for {model_name}: {params}")
            
            # Make predictions and calculate scores
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = (test_model_score, train_model_score)
            
            logger.info(f"{model_name} - Test Score: {test_model_score:.4f}, "
                       f"Train Score: {train_model_score:.4f}")
        
        return report, best_params

    except Exception as e:
        raise CustomException(e, sys)


