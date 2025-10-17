import sys
import logging
from src.logger import setup_logging

# Set up logging
# setup_logging()

# ANSI escape code for yellow color
YELLOW_CONSOLE_MSG_CODE = '\033[93m'
# ANSI escape code to reset color
RESET_CONSOLE_MSG_CODE = '\033[0m'

def error_msg_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"{YELLOW_CONSOLE_MSG_CODE}ERROR: File - {file_name}; Error Message - {str(error)}; Line - {exc_tb.tb_lineno}{RESET_CONSOLE_MSG_CODE}"
    
    return error_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
