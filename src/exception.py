import sys
from src.logger import logging
def error_message_detail(error, error_detail: sys):
    """Generate a detailed error message."""
    _, _, exc_tb = error_detail.exc_info()  # Corrected the method name
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract the filename
    line_number = exc_tb.tb_lineno  # Extract the line number
    
    # Create the error message
    error_message = (
        f"Error occurred in Python script named [{file_name}] "
        f"at line number [{line_number}] with error message [{str(error)}]"
    )
    return error_message  # Return the formatted error message
class CustomException(Exception):
    """Custom exception class."""
    
    def __init__(self, error_message, error_detail: sys):
        """Initialize the custom exception with error details."""
        super().__init__(error_message)  # Call the base class constructor
        self.error_message = error_message_detail(error_message, error_detail)  # Store detailed error message

    def __str__(self):
        """Return the error message as a string."""
        return self.error_message
if __name__ == "__main__":
    try:
        # Simulate an error for demonstration
        x = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)  # Raise custom exception
