import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message , errot_detail : sys):
        super().__init__()
        self.error_message = self.get_detailed_error_message(error_message, errot_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys) :
        
        _,_,exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        detailed_error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {error_message}"

        return detailed_error_message
        
    def __str__(self):
        return self.error_message