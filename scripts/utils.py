# utils.py
from datetime import datetime  
from audio_ul2_config import DEBUG

def debug_print(function_name=None, msg=None):  
    if DEBUG:  
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-5]  
        # print(f"[{timestamp}] [{function_name}] {msg}")  
