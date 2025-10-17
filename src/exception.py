import sys
from src.logger import logger

# ANSI escape codes for coloring in console
YELLOW_CONSOLE_MSG_CODE = '\033[93m'
RESET_CONSOLE_MSG_CODE = '\033[0m'

def error_msg_detail(error, error_detail: sys):
    """
    Extracts detailed error information: file, line number, message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = (
        f"{YELLOW_CONSOLE_MSG_CODE}ERROR: File - {file_name}; "
        f"Error Message - {str(error)}; Line - {exc_tb.tb_lineno}{RESET_CONSOLE_MSG_CODE}"
    )
    return error_msg

class CustomException(Exception):
    """
    Custom Exception class that logs the error automatically
    """
    def __init__(self, error_message, error_detail: sys):
        # Generate formatted message
        self.error_message = error_msg_detail(error_message, error_detail)
        # Log the error
        logger.error(self.error_message)
        # Call parent Exception class
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
