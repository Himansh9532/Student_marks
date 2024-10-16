import os 
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging  # Make sure to import logging if you want to use it

def save_object(file_path, obj):
    try:
        # Create directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file and save the object using dill
        with open(file_path, "wb") as file_obj:  # Corrected the open function
            dill.dump(obj, file_obj)

        logging.info(f"Object saved at {file_path}")  # Log success

    except Exception as e:  # Fixed the typo from Except to Exception
        logging.error(f"Error occurred while saving object: {str(e)}")  # Log the error
        raise CustomException(e, sys)
