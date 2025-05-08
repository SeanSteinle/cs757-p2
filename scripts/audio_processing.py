# audio_processing.py  

import sys  
import librosa  
import numpy as np  
import soundfile as sf  
import matplotlib.pyplot as plt  
import librosa.display  
import os 

# Add 'birdie' directory   
birdie_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'birdie')  
if birdie_path not in sys.path:  
    sys.path.insert(0, birdie_path) 
    
from audio_ul2_config import (  
    SAMPLE_RATE, N_MELS, HOP_LENGTH,  
    N_FFT, DURATION
    )  
from utils import debug_print

#%%
def load_audio_file(file_path):
    """  
    Loads an audio file and preprocesses the raw audio data.
    """  
    fn = load_audio_file.__name__
    try:
        debug_print(fn, f"Loading audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

        # Handle silent or too-short audio AFTER loading with duration limit
        # A very short segment might still result in some samples but no meaningful frames
        if len(y) < int(HOP_LENGTH): # Check if at least one full hop is possible
             return None

        # Pad/Truncate audio to exactly DURATION seconds worth of samples
        target_samples_process = int(DURATION * SAMPLE_RATE)
        if len(y) < target_samples_process:
            padding = np.zeros(target_samples_process - len(y), dtype=y.dtype)
            y = np.concatenate([y, padding])
            debug_print(fn, f"Padded audio to {target_samples_process} samples.")
        elif len(y) > target_samples_process:
            y = y[:target_samples_process]
            debug_print(fn, f"Truncated audio to {target_samples_process} samples.")

        # Normalize audio to prevent issues with amplitude
        y = librosa.util.normalize(y)
        if np.sum(np.abs(y)) < 1e-9: # Check for near-silent audio after normalization
             return None

        return y

    except Exception as e:
        return None

def convert_audio_to_mel(audio_data):
    fn = convert_audio_to_mel.__name__
    if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1:
         raise Exception(fn, "Input audio_data is not a 1D numpy array.")

    if audio_data.size == 0:
         raise Exception(fn, "Input audio_data is empty.")

    try:
        # Ensure the input audio data matches the expected duration for consistent spectrogram shape
        target_samples_process = int(DURATION * SAMPLE_RATE)
        if len(audio_data) != target_samples_process:
             if len(audio_data) < target_samples_process:
                 debug_print(fn, "Padding...")
                 padding = np.zeros(target_samples_process - len(audio_data), dtype=audio_data.dtype)
                 audio_data = np.concatenate([audio_data, padding])
             else:
                 debug_print(fn, "Truncating...")
                 audio_data = audio_data[:target_samples_process]

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=SAMPLE_RATE, # Use global sample rate from config
            n_fft=N_FFT,     # Use global N_FFT from config
            hop_length=HOP_LENGTH, # Use global HOP_LENGTH from config
            n_mels=N_MELS,   # Use global N_MELS from config
            power=2.0 # Using power spectrogram
            )

        # Convert to dB scale
        ref_value = np.max(mel_spec) if np.max(mel_spec) > 1e-10 else 1e-10
        mel_spec_db = librosa.power_to_db(mel_spec, ref=ref_value)

        # Ensure the spectrogram has the target number of frames after processing
        # This step accounts for potential floating-point differences in frame calculation
        expected_frames = int(np.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH))
        current_frames = mel_spec_db.shape[1]


        if current_frames < expected_frames:
            # Pad spectrogram with a small value in dB (e.g., -100 dB, representing silence)
            padding_spec = np.full((N_MELS, expected_frames - current_frames), -100.0)
            mel_spec_db = np.concatenate([mel_spec_db, padding_spec], axis=1)
        elif current_frames > expected_frames:
            # Truncate spectrogram
            mel_spec_db = mel_spec_db[:, :expected_frames]

        # Final check on shape
        if mel_spec_db.shape != (N_MELS, expected_frames):
             raise Exception(fn, "Final spectrogram shape is not matching")

        debug_print(fn, f"Success. Final mel_spec_db shape is {mel_spec_db.shape}")
        return mel_spec_db

    except Exception as e:
        raise Exception(fn, f"Error converting audio data to mel spectrogram: {str(e)}")
        


def mel_to_audio(mel_spectrogram, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, target_peak=0.9):  
    fn = mel_to_audio.__name__
     
    try:  
        mel_spectrogram_power = librosa.db_to_power(mel_spectrogram)  

        linear_spectrogram_power = librosa.feature.inverse.mel_to_stft(  
            mel_spectrogram_power,  
            sr=sr,  
            n_fft=n_fft  
            )  

        y_reconstructed = librosa.griffinlim(  
            linear_spectrogram_power,  
            n_fft=n_fft,  
            hop_length=hop_length  
            )  

        peak_value = np.max(np.abs(y_reconstructed))  

        if peak_value > 1e-8: # Avoid division by near zero  
            scaling_factor = target_peak / peak_value  
            y_reconstructed_normalized = y_reconstructed * scaling_factor  
            debug_print(fn, f"Applied peak normalization with target peak {target_peak}.")  
        else:  
            y_reconstructed_normalized = y_reconstructed  
            debug_print(fn, "Reconstructed audio is near silent, skipping normalization.")  

        return y_reconstructed_normalized.astype(np.float32)

    except Exception as e:  
        debug_print(fn, f"Error during mel spectrogram to audio conversion: {str(e)}")  
        return None  


def save_audio(audio_data, output_path, sr=SAMPLE_RATE):  
    fn = save_audio.__name__
    sf.write(output_path, audio_data.astype(np.float32), sr)
    return True  


def visualize_mel_spectrogram(mel_spectrogram_db, title="Mel Spectrogram", output_path=None, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):  
    fn = visualize_mel_spectrogram.__name__

     
    plt.figure(figsize=(12, 6))  
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')  
    plt.colorbar(format='%+2.0f dB')  
    plt.title(title)  
    plt.tight_layout()  

    if output_path:  
        plt.savefig(output_path)  
    else:  
        plt.show()  

# Example usage  
# audio -> mel -> audio (with final spectrogram from saved audio)  
if __name__ == "__main__":  
    
    fn = f"*{os.path.basename(__file__)}*"
    print("\nRunning Script",fn,"\n")
    
    import os  
    output_dir = "reconstruction_output"  
    os.makedirs(output_dir, exist_ok=True)  

    input_audio_path = 'blues.00005.wav'  
    output_reconstructed_audio_path = os.path.join(output_dir, "reconstructed_" + os.path.basename(input_audio_path))  
    original_spectrogram_path = os.path.join(output_dir, "original_mel_spectrogram.png")  
    reconstructed_spectrogram_path = os.path.join(output_dir, "reconstructed_mel_spectrogram.png")  
    
    # generate a dummy audio if file does not exist
    if not os.path.exists(input_audio_path):  
        debug_print(fn, f"Creating a dummy audio file at {input_audio_path}")  
        # Generate a simple sine wave  
        t = np.linspace(0., DURATION, int(DURATION * SAMPLE_RATE))  
        amplitude = np.iinfo(np.int16).max * 0.1 
        data = amplitude * np.sin(2. * np.pi * 440. * t)  
        sf.write(input_audio_path, data.astype(np.int16), SAMPLE_RATE)  
        debug_print(fn, "Dummy audio file created.")  

    # --- Load audio and convert to original mel spectrogram ---  
    y = load_audio_file(input_audio_path)  
    original_mel_spec_db = convert_audio_to_mel(y)
    
    # --- Visualize the original spectrogram ---  
    visualize_mel_spectrogram(  
        original_mel_spec_db,  
        title="Original Mel Spectrogram",  
        output_path=original_spectrogram_path,  
        sr=SAMPLE_RATE,  
        hop_length=HOP_LENGTH  
    )   

    # --- Convert the mel spectrogram back to audio to reconstruct  ---
    reconstructed_audio = mel_to_audio(original_mel_spec_db, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, target_peak=0.9)  
    
    # --- Save the reconstructed audio ---  
    save_success = save_audio(reconstructed_audio, output_reconstructed_audio_path, SAMPLE_RATE)  

    # --- Reload the saved audio and generate its spectrogram ---     
    saved_y_reconstruction = load_audio_file(output_reconstructed_audio_path)
    saved_reocnstruction_spec = convert_audio_to_mel(saved_y_reconstruction) 

    # --- Visualize the final reconstructed spectrogram ---  
    visualize_mel_spectrogram(  
        saved_reocnstruction_spec,  
        title="Reconstructed Mel Spectrogram",  
        output_path=reconstructed_spectrogram_path,  
        sr=SAMPLE_RATE,  
        hop_length=HOP_LENGTH  
    )