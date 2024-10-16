import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models  # Utility functions for saving model and evaluating models
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging system for tracking

# Create Data Class to store the path where the trained model will be saved
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")  # Path to save the trained model

# Model Trainer Class
class ModelTrainer:
    def __init__(self):
        # Initialize the model trainer with configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        This method initiates the model training process by splitting the dataset, 
        training various models, and evaluating them.
        """
        try:
            # Step 1: Log the process of splitting the training and test data
            logging.info("Splitting training and test input data")
            
            # Splitting the training and test arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, all columns except the last one (features)
                train_array[:, -1],   # All rows, only the last column (target)
                test_array[:, :-1],   # All rows, all columns except the last one (features)
                test_array[:, -1]     # All rows, only the last column (target)
            )

            # Step 2: Define the models to be trained and evaluated
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),  # Disable verbose output for CatBoost
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Linear Regression": {},
                "Lasso": {'alpha': np.logspace(-4, 4, 10), 'max_iter': [1000, 5000, 10000]},
                "Ridge": {'alpha': np.logspace(-4, 4, 10), 'max_iter': [1000, 5000, 10000]},
                "K-Neighbors Regressor": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
                "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['auto', 'sqrt', 'log2']},
                "Random Forest Regressor": {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['auto', 'sqrt', 'log2']},
                "XGBRegressor": {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 0.9, 1.0]},
                "CatBoosting Regressor": {'iterations': [500, 1000], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [6, 10, 12]},
                "AdaBoost Regressor": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
            }

            # Step 3: Evaluate the models using the evaluate_model function and return the results
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,  # Training data
                X_test= X_test, y_test=y_test,      # Test data
                models=models,                      # Dictionary of models to evaluate
                param = params
            )
            
            # Get best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # Get the name of the best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Raise custom exception if the best model score is below the threshold
            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found.")
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Step 4: Save the best-performing model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            # Make predictions using the best model
            predicted = best_model.predict(X_test)

            # Calculate R2 score for predictions
            model_r2_score = r2_score(y_test, predicted)
            
            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception with the error message