# audio_ul2_config.py

DEBUG = True

# Parameters for audio processing.
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 32
N_FFT = 512
DURATION = 3

# Tokenizer parameters
N_CLUSTERS = 512
DEFAULT_VOCAB_SIZE = N_CLUSTERS # will be updated in the tokenizer
IGNORE_INDEX = -100 # Standard ignore index for CrossEntropyLoss

audio_ul2_config = [  
    # Use prefix_language_modeling but adjust for audio continuation  
    {  
        "name": 'prefix_language_modeling',  
        "prefix_fraction": 0.5,  # Predict second half from first half  
        "paradigm_str": "<|PREFIX LM|>", 
    },  
    
    # Use infilling objective for audio repair  
    {  
        "name": 'infilling',  
        "paradigm_str": "<|INFILL|>", 
    },  
    
    # Additional infilling variant  
    {  
        "name": 'next_token_prediction',  
        "paradigm_str": "<|NTP|>", 
    }  
]  