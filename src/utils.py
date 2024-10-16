import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging  # Make sure to import logging if you want to use it
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        # Create directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file and save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved at {file_path}")  # Log success

    except Exception as e:
        logging.error(f"Error occurred while saving object: {str(e)}")  # Log the error
        raise CustomException(e, sys)
#Create Function And Evaluate Model
def evaluate_models(X_train, y_train, X_test, y_test, models , param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
             # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            # Set the best parameters from GridSearchCV
            model.set_params(**gs.best_params_)
            
            # Refit the model with the best parameters
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # R2 Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Storing the test model score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {str(e)}")  # Log the error
        raise CustomException(e, sys)