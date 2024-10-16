import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the raw data from CSV file
            df = pd.read_csv("Student/stud.csv")
            logging.info("Data loaded successfully")

            # Create artifacts directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #Save the data into CSV FILES
            df.to_csv(self.ingestion_config.raw_data_path, index=False , header= True)

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException("Data ingestion failed", e)

# Example usage
if __name__ == "__main__":
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

#     """Imports:
# os: This module provides a way of using operating system-dependent functionality, like reading or writing to the file system.
# sys: Provides access to some variables used or maintained by the Python interpreter and functions that interact with the interpreter.
# CustomException: Custom error class defined in the src.exception module for handling exceptions.
# logging: A module from the src.logger that manages logging messages for tracking the application’s behavior.
# pandas: A powerful data manipulation and analysis library, used here for handling data in DataFrame format.
# train_test_split: A function from sklearn.model_selection used to split datasets into training and testing sets.
# dataclass: A decorator to create classes that are primarily used to store data, making the code cleaner and more maintainable."""


# #2"""DataIngestionConfig Class:
# This is a data class that stores configuration paths for data ingestion.
# Attributes:
# train_data_path: The path where the training data will be saved.
# test_data_path: The path where the testing data will be saved.
# raw_data_path: The path where the raw data (input CSV file) is located.
# Using os.path.join ensures that the paths are constructed correctly regardless of the operating system."""



# DataIngestion Class:
# This class handles the data ingestion process, including reading the raw data, splitting it into train and test sets, and saving those sets to specified paths.
# Constructor (__init__ method):
# Initializes an instance of DataIngestionConfig and assigns it to self.ingestion_config.


# Initiate Data Ingestion:
# This method logs that the data ingestion process has started.
# python

# Reading Data:
# A try block is used to handle any exceptions that might occur during the data ingestion process.
# The raw data is read from a CSV file (stud.csv) located in the Student directory using pandas.read_csv().
# A log message is generated to confirm that the data has been loaded successfully.


# Splitting Data:
# The loaded DataFrame (df) is split into training and testing sets using train_test_split().
# test_size=0.2 indicates that 20% of the data will be used for testing, while the remaining 80% will be used for training.
# random_state=42 ensures that the split is reproducible. The same data will be split the same way if this script is run multiple times.

# Saving Data:
# The training and testing sets are saved as CSV files using the paths specified in DataIngestionConfig.
# The index=False argument ensures that the DataFrame indices are not written to the CSV file.
# A log message indicates that the data has been saved successfully.
# python



# Returning File Paths:
# The method returns the paths to the training and testing datasets, which can be used later in the model training or evaluation processes.