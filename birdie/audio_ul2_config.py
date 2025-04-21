# Parameters for audio processing
SAMPLE_RATE = 22050
N_MELS = 32
HOP_LENGTH = 256
N_FFT = 256
DURATION = 5

# Tokenizer parameters
N_CLUSTERS = 1024

DEBUG = True

# AUDIO_SEQ_LENGTH = 256  # Typical sequence length for audio tokens  
# MASK_TOKEN_ID = N_CLUSTERS - 1  # Reserve the last cluster as a mask token  

audio_ul2_config = [  
    # Use prefix_language_modeling but adjust for audio continuation  
    {  
        "name": 'prefix_language_modeling',  
        "prob": 1.0,  
        "corruption_rate": 0.0,  # No corruption for continuation  
        "prefix_fraction": 0.5,  # Predict second half from first half  
        "paradigm_token": "[S]",  
        "mean_tokens_per_span": 4.0 * 8  
    },  
    
    # Use infilling objective for audio repair  
    {  
        "name": 'infilling',  
        "prob": 0.7,  
        "corruption_rate": 0.15,  
        "paradigm_token": "[R]",  
        "mean_tokens_per_span": 4.0 * 4    
    },  
    
    # Additional infilling variant  
    {  
        "name": 'infilling',  
        "prob": 0.3,  
        "corruption_rate": 0.3,  
        "paradigm_token": "[X]",   
        "mean_tokens_per_span": 4.0 * 16  
    }  
]  