import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object  # Corrected import statement

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates the data transformer pipeline for both numerical and categorical features.
        """
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Pipeline for numerical columns (handling missing values and scaling)
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Handle missing values by median
                ("scaler", StandardScaler())  # Scale numerical values
            ])

            # Pipeline for categorical columns (handling missing values, one-hot encoding, scaling)
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values by most frequent
                ("one_hot_encoder", OneHotEncoder()),  # One-Hot encoding for categorical features
                ("scaler", StandardScaler(with_mean=False))  # Scaling after one-hot encoding (without centering)
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Combine the pipelines into a column transformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function handles data loading, transformation, and saving the preprocessing object.
        """
        try:
            # Load the training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data loaded successfully")

            # Getting the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column
            target_column_name = "math_score"

            # Split the data into features and target (for both train and test data)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessing transformations to the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with target values (train and test)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object to disk
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed and preprocessing object saved.")

            # Return transformed data arrays and the preprocessing object file path
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {str(e)}")
            raise CustomException(e, sys)
