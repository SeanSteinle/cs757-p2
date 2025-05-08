# tokenizer.py

import sys
import numpy as np
import soundfile as sf
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import librosa
import ast 

# Add 'birdie' directory 
birdie_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'birdie')
if birdie_path not in sys.path:
    sys.path.insert(0, birdie_path)

from audio_ul2_config import (
    audio_ul2_config, 
    validation_objectives_config,
    SAMPLE_RATE, N_MELS, HOP_LENGTH,
    N_FFT, DURATION, N_CLUSTERS,
    IGNORE_INDEX 
)
from audio_processing import (
    load_audio_file, convert_audio_to_mel, mel_to_audio,
    save_audio, visualize_mel_spectrogram
)
from utils import debug_print 

#%%
class SpectrogramTokenizer:
    def __init__(self, n_clusters=N_CLUSTERS):
        fn = "SpectrogramTokenizer.__init__"
        debug_print(fn, "Initializing SpectrogramTokenizer.")
        self.n_clusters = n_clusters

        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=23, algorithm='lloyd')
        self.cluster_centers = None
        self.frame_mean = None 
        self.frame_std = None 

        # Special tokens map: token_string -> token_id
        self.special_token_map = {}
        # Inverse map: token_id -> token_string
        self.id_to_token = {}

        # Assign PAD token ID to IGNORE_INDEX from config.
        self.pad_token = "[PAD]"
        self.ignore_index = IGNORE_INDEX
        self.pad_token_id = self.ignore_index

        if 0 < self.pad_token_id < n_clusters:
             raise ValueError(f"IGNORE_INDEX ({self.ignore_index}) from config overlaps with audio token range")

        self._add_special_token(self.pad_token, self.pad_token_id) 

        # Collect and add paradigm tokens from objective configs
        all_paradigm_strings = set()
        for obj_config_list in [audio_ul2_config, validation_objectives_config]:
             for obj in obj_config_list:
                  paradigm_str = obj.get('paradigm_str') or obj.get('paradigm_token')
                  if paradigm_str is not None and isinstance(paradigm_str, str):
                       all_paradigm_strings.add(paradigm_str)

        for paradigm_str in sorted(list(all_paradigm_strings)):
             if paradigm_str not in self.special_token_map:
                  debug_print(fn, f"Adding paradigm token '{paradigm_str}'.")
                  self._add_special_token(paradigm_str)
             else:
                  debug_print(fn, f"Paradigm token '{paradigm_str}' already added (likely as PAD). Skipping.")

        # Add other common special tokens 
        common_special_tokens_other = ["[MASK]", 
                                       "[UNK]",
                                       "<|START|>",
                                       "<|END|>"
                                       "\n<|assistant|>\n"
                                       ]
        
        for token_str in sorted(common_special_tokens_other):
             if token_str not in self.special_token_map:
                  self._add_special_token(token_str) 
             else:
                  debug_print(fn, f"Common special token '{token_str}' already added. Skipping.")

        self.vocab_size = self.n_clusters + len(self.special_token_map)
        debug_print(fn, f"Special token map: {self.special_token_map}")
        debug_print(fn, f"ID to token map: {self.id_to_token}")


    def _add_special_token(self, token_str, token_id=None):
        fn = "SpectrogramTokenizer._add_special_token"
        if token_id is None:
            token_id = self.n_clusters + len(self.special_token_map)
            if token_id == self.pad_token_id and token_str != self.pad_token:
                 token_id += 1
                 while token_id in self.id_to_token:
                      token_id += 1

        # Check for existing mappings to avoid conflicts
        if token_id in self.id_to_token:
             if self.id_to_token[token_id] != token_str:
                  raise ValueError(f"Token ID {token_id} is already assigned .")
             return

        if token_str in self.special_token_map:
             if self.special_token_map[token_str] != token_id:
                  raise ValueError(f"Token string '{token_str}' is already assigned")
             return

        debug_print(fn, f"Adding mapping: '{token_str}' -> {token_id}")
        self.special_token_map[token_str] = token_id
        self.id_to_token[token_id] = token_str

    def fit(self, spectrogram_iterator):
        fn = "SpectrogramTokenizer.fit"

        all_frames_list = []

        for spec in tqdm(spectrogram_iterator, desc="Collecting spectrogram for K-Means "):
            if spec is not None and spec.shape[1] > 0:
                if spec.shape[0] != N_MELS:
                    raise ValueError(f"{fn}: Found Spectrogram with incorrect feature shape")
                frames = spec.T # (time_frames, N_MELS)
                all_frames_list.append(frames)

        all_frames = np.vstack(all_frames_list)

        if all_frames.shape[0] < self.n_clusters:
             if all_frames.shape[0] == 0:
                  raise ValueError(f"{fn}: No frames collected for fitting KMeans.")
             else:
                  if all_frames.shape[0] < 10 * self.n_clusters: 
                       debug_print(fn, "Warning: Very low number of frames compared to clusters.")

        # Perform normalization
        self.frame_mean = np.mean(all_frames, axis=0, keepdims=True).astype(np.float32)
        self.frame_std = np.std(all_frames, axis=0, keepdims=True).astype(np.float32) + 1e-8  
        normalized_frames = (all_frames - self.frame_mean) / self.frame_std


        try:
             self.kmeans.fit(normalized_frames)
             self.cluster_centers = self.kmeans.cluster_centers_.astype(np.float32)
        except Exception as e:
             raise RuntimeError(f"{fn}: K-Means fitting failed. error: {str(e)}")


    def _tokenize_spectrogram(self, spectrogram):
        fn = "SpectrogramTokenizer._tokenize_spectrogram"

        if not isinstance(spectrogram, np.ndarray) or spectrogram.ndim != 2 or spectrogram.shape[0] != N_MELS:
            raise ValueError(f"{fn}: Expected ({N_MELS}, T) for spectrogram, got {getattr(spectrogram, 'shape', 'N_A')}")

        if self.cluster_centers is None or self.frame_mean is None or self.frame_std is None:
             raise RuntimeError(f"{fn}: Tokenizer not fitted.")

        if spectrogram.shape[1] == 0:
             debug_print(fn, "Input spectrogram is empty. Returning empty token sequence.")
             return np.array([], dtype=np.int32)

        frames = spectrogram.T  # (time_frames, N_MELS)

        normalized_frames = self._normalize_frames(frames)

        if normalized_frames.shape[0] == 0:
             return np.array([], dtype=np.int32)

        tokens = self.kmeans.predict(normalized_frames)

        return np.array(tokens, dtype=np.int32)

    def _normalize_frames(self, frames):
         fn = "SpectrogramTokenizer._normalize_frames"

         if self.frame_mean is None or self.frame_std is None:
              raise RuntimeError(f"{fn}: Tokenizer is not fitted. Cannot normalize frames.")
         if not isinstance(frames, np.ndarray) or frames.ndim != 2 or frames.shape[-1] != N_MELS:
              raise TypeError(f"{fn}: Expected 2D numpy array with {N_MELS} features, got {getattr(frames, 'shape', 'N_A')}")
         if frames.shape[0] == 0:
              return np.array([], dtype=np.float32).reshape(0, N_MELS)

         frames = frames.astype(np.float32)
         normalized_frames = (frames - self.frame_mean) / self.frame_std
         return normalized_frames

    def _denormalize_frames(self, normalized_frames):
         fn = "SpectrogramTokenizer._denormalize_frames"
         if self.frame_mean is None or self.frame_std is None:
              raise RuntimeError(f"{fn}: Tokenizer is not fitted. Cannot denormalize frames.")
         if not isinstance(normalized_frames, np.ndarray) or normalized_frames.ndim != 2 or normalized_frames.shape[-1] != N_MELS:
              raise TypeError(f"{fn}: Expected 2D numpy array with {N_MELS} features, got {getattr(normalized_frames, 'shape', 'N_A')}")
         if normalized_frames.shape[0] == 0:
              debug_print(fn, "Input normalized frames are empty. Returning empty denormalized frames.")
              return np.array([], dtype=np.float32).reshape(0, N_MELS)

         normalized_frames = normalized_frames.astype(np.float32) # Ensure float for calculations
         denormalized_frames = (normalized_frames * self.frame_std) + self.frame_mean
         return denormalized_frames

    def _detokenize_audio_tokens(self, audio_token_ids):
        fn = "SpectrogramTokenizer._detokenize_audio_tokens"
        debug_print(fn, f"Detokenizing {audio_token_ids.size} audio tokens.")
        if audio_token_ids.size == 0:
            debug_print(fn, "Input audio token ID sequence is empty. Returning empty spectrogram.")
            return np.array([], dtype=np.float32).reshape(N_MELS, 0)

        if np.any(audio_token_ids < 0) or np.any(audio_token_ids >= self.n_clusters):
             raise ValueError("{fn}: Received invalid audio token IDs. ")

        # Use cluster centers to reconstruct normalized frames
        normalized_frames_recon = self.kmeans.cluster_centers_[audio_token_ids].copy().astype(np.float32)

        # Denormalize the reconstructed frames
        frames_recon = self._denormalize_frames(normalized_frames_recon)

        # Reshape back to spectrogram format (features, time)
        spectrogram_recon = frames_recon.T

        return spectrogram_recon

    def encode_text(self, text):
        fn = "SpectrogramTokenizer.encode_text"

        if isinstance(text, str):
            if text.startswith('[') and text.endswith(']'):
                 try:
                      potential_list = ast.literal_eval(text)
                      if isinstance(potential_list, list) and all(isinstance(item, int) for item in potential_list):
                           return np.array(potential_list, dtype=np.int32)
                 except (ValueError, SyntaxError, TypeError):
                      pass # Not a valid list string, treat as normal string

            # Check for special token strings
            special_token_id = self.get_special_token_id(text)
            if special_token_id is not None:
                return np.array([special_token_id], dtype=np.int32)

            # Handle unknown strings
            unk_token_id = self.get_special_token_id("[UNK]")
            if unk_token_id is not None:
                 return np.array([unk_token_id], dtype=np.int32)
            else:
                 raise ValueError(f"{fn}: Unknown token string: '{text}'.")

        elif isinstance(text, list):
             encoded_ids = []
             for item in text:
                  item_ids = self.encode_text(item) # Recursive 
                  if item_ids.size > 0:
                       encoded_ids.append(item_ids)
             result = np.concatenate(encoded_ids) if encoded_ids else np.array([], dtype=np.int32)
             return result

        elif isinstance(text, np.ndarray) and np.issubdtype(text.dtype, np.integer):
            return text.astype(np.int32)

        elif isinstance(text, (int, np.integer)):
             return np.array([int(text)], dtype=np.int32)

        else:
            raise TypeError(f"{fn}: Input must be a string, list, or numpy array of integers, got {type(text)}")

    def encode_audio(self, audio_data):
        fn = "SpectrogramTokenizer.encode_audio"

        if self.cluster_centers is None or self.frame_mean is None or self.frame_std is None:
             raise RuntimeError(f"{fn}: Tokenizer is not fitted. Cluster centers or normalization stats missing.")

        if audio_data.size == 0:
             debug_print(fn, "Input audio data is empty. Returning empty token sequence.")
             return np.array([], dtype=np.int32)

        spectrogram = convert_audio_to_mel(audio_data)

        if spectrogram is None or spectrogram.shape[1] == 0:
             return np.array([], dtype=np.int32)

        debug_print(fn, "Tokenizing spectrogram.")
        audio_tokens = self._tokenize_spectrogram(spectrogram)
        debug_print(fn, f"Encoded audio to {audio_tokens.size} tokens.")
        return audio_tokens

    def encode(self, data):
        fn = "SpectrogramTokenizer.encode"

        if isinstance(data, np.ndarray):
            if data.ndim == 2: # Assume spectrogram
                debug_print(fn, "Dispatching to _tokenize_spectrogram for 2D numpy array.")
                return self._tokenize_spectrogram(data)
            elif data.ndim == 1 and np.issubdtype(data.dtype, np.integer): # Assume already token IDs
                 debug_print(fn, "Input is already token IDs (1D numpy array of integers). Returning as is (int32).")
                 return data.astype(np.int32)
            elif data.ndim == 1: # Assume raw audio (1D float array)
                 debug_print(fn, "Dispatching to encode_audio for 1D numpy array.")
                 return self.encode_audio(data)
            else:
                 raise TypeError(f"{fn}: Input numpy array has unsupported dimensions: {data.ndim}")

        elif isinstance(data, (str, list)): # Handle strings and lists
            debug_print(fn, f"Dispatching to encode_text for type: {type(data)}.")
            return self.encode_text(data)

        elif isinstance(data, (int, np.integer)): # Handle single integer IDs
             debug_print(fn, f"Input is a single integer ID: {data}. Wrapping in numpy array.")
             return np.array([int(data)], dtype=np.int32)

        else:
            raise TypeError(f"{fn}: Input data type not supported: {type(data)}")

    def decode(self, token_ids):
        fn = "SpectrogramTokenizer.decode"

        if self.cluster_centers is None or self.frame_mean is None or self.frame_std is None:
             raise RuntimeError(f"{fn}: Tokenizer is not fitted. Cannot decode.")

        if token_ids.size == 0:
             debug_print(fn, "Decoding empty token sequence. Returning empty audio.")
             return np.array([], dtype=np.float32)

        # --- Filter out non-audio token IDs ---
        # Valid audio tokens are IDs from 0 up to n_clusters - 1.
        debug_print(fn, f"Filtering non-audio tokens (IDs < 0 or >= {self.n_clusters}).")
        valid_audio_tokens_mask = (token_ids >= 0) & (token_ids < self.n_clusters)
        audio_token_ids_only = token_ids[valid_audio_tokens_mask]


        if audio_token_ids_only.size == 0:
            return np.array([], dtype=np.float32)

        # Convert the filtered audio tokens back to a spectrogram
        # _detokenize_audio_tokens expects only valid audio token IDs
        detokenized_spectrogram = self._detokenize_audio_tokens(audio_token_ids_only.astype(np.int32))

        if detokenized_spectrogram is None or detokenized_spectrogram.shape[1] == 0:
             return np.array([], dtype=np.float32)

        # Convert the spectrogram back to audio
        reconstructed_audio = mel_to_audio(detokenized_spectrogram.astype(np.float32), sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, target_peak=0.9)

        if reconstructed_audio is None:
             return np.array([], dtype=np.float32)

        return reconstructed_audio.astype(np.float32)

    def tokens_to_strings(self, token_ids):
        fn = "SpectrogramTokenizer.tokens_to_strings"

        if not isinstance(token_ids, np.ndarray) or token_ids.ndim != 1 or not np.issubdtype(token_ids.dtype, np.integer):
             if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
                  token_ids = np.array(token_ids, dtype=np.int32)
             else:
                  raise TypeError(f"{fn}: Expected 1D numpy array or list of integer token IDs for tokens_to_strings.")

        string_sequence = []
        for token_id in token_ids:
            token_id_int = int(token_id)
            if 0 <= token_id_int < self.n_clusters:
                # Represent audio tokens with their ID
                string_sequence.append(f"{token_id_int}") # Or f"AUDIO_{token_id_int}"
            elif token_id_int in self.id_to_token:
                string_sequence.append(self.id_to_token[token_id_int])
            else:
                 string_sequence.append(f"UNKNOWN_ID_{token_id_int}")

        debug_print(fn, f"Converted token IDs to {len(string_sequence)} strings.")
        return string_sequence

    def get_special_token_id(self, token_string):
        """
        Returns the ID for a given special token string.
        Returns None if the special token string is not recognized.
        """
        return self.special_token_map.get(token_string)

    def get_token_string(self, token_id):
        """
        Returns the string for a given token ID (audio or special).
        Returns None if the token ID is not recognized or invalid.
        """
        if not isinstance(token_id, (int, np.integer)):
             return None
        token_id_int = int(token_id)
        if 0 <= token_id_int < self.n_clusters:
            return f"{token_id_int}" # Or f"AUDIO_{token_id_int}"
        return self.id_to_token.get(token_id_int)

#%%
# Example usage
# Procedure: audio file -> load raw audio -> encode -> (Simulate LLM) -> decode -> save audio
if __name__ == "__main__":
    
    # --- Helper functions for the __main__ test block ---
    def single_spectrogram_iterator(spec):
        """Yields a single spectrogram."""
        yield spec
    
    def audio_file_iterator(file_paths):
        """Iterates through audio file paths, loads, and yields raw audio data."""
        fn = "audio_file_iterator"
        debug_print(fn, f"Creating audio file iterator for {len(file_paths)} files.")
        for file_path in file_paths:
            audio_data = load_audio_file(file_path)
            if audio_data is not None:
                debug_print(fn, f"Yielding audio data for {file_path}.")
                yield audio_data
            else:
                debug_print(fn, f"Failed to load audio data from {file_path}.")
    
    def spectrogram_from_files_iterator(file_paths):
        fn = "spectrogram_from_files_iterator"
        debug_print(fn, f"Creating spectrogram iterator from {len(file_paths)} audio files.")
        for file_path in tqdm(file_paths, desc="Generating spectrograms for fitting"):
            audio_data = load_audio_file(file_path)
            if audio_data is not None:
                spectrogram = convert_audio_to_mel(audio_data)
                if spectrogram is not None:
                    # debug_print(fn, f"Yielding spectrogram for {file_path} with shape {spectrogram.shape}.")
                    yield spectrogram
                else:
                    debug_print(fn, f"Could not convert audio from {file_path} to spectrogram.")
            else:
                debug_print(fn, f"Could not load audio from {file_path}.")    
    
    fn = f"*{os.path.basename(__file__)}*"
    print("\nRunning Spectrogram Tokenizer Test Script\n")

    output_dir = "reconstruction_output_tokenizer_test"
    os.makedirs(output_dir, exist_ok=True)

    input_audio_path = 'jazz.00005.wav'
    output_reconstructed_audio_path = os.path.join(output_dir, "reconstructed_llm_" + os.path.basename(input_audio_path))
    original_spectrogram_path = os.path.join(output_dir, "original_mel_spectrogram_llm.png")
    reconstructed_spectrogram_path = os.path.join(output_dir, "reconstructed_mel_spectrogram_llm.png")
    detokenized_spectrogram_from_decode_path = os.path.join(output_dir, "detokenized_mel_spectrogram_from_decode_llm.png")

    # --- Generate a dummy audio if file does not exist ---
    if not os.path.exists(input_audio_path):
        debug_print(fn, f"Creating a dummy audio file at {input_audio_path}")
        # Generate a simple sine wave
        t = np.linspace(0., DURATION, int(DURATION * SAMPLE_RATE))
        amplitude = np.iinfo(np.int16).max * 0.1
        data = amplitude * np.sin(2. * np.pi * 440. * t)
        try:
            sf.write(input_audio_path, data.astype(np.int16), SAMPLE_RATE)
            debug_print(fn, "Dummy audio file created.")
        except Exception as e:
            debug_print(fn, f"Error creating dummy audio file: {str(e)}")
            sys.exit(1)

    # --- Load audio and convert to original mel spectrogram (for visualization) ---
    debug_print(fn, f"Loading original audio file: {input_audio_path}")
    original_audio_data = load_audio_file(input_audio_path)
    if original_audio_data is None or original_audio_data.size == 0:
        raise RuntimeError(f"{fn}: Failed to load valid original audio data for test.")

    debug_print(fn, "Converting original audio data to mel spectrogram for visualization.")
    original_mel_spec_db = convert_audio_to_mel(original_audio_data)
    if original_mel_spec_db is None or original_mel_spec_db.shape[1] == 0:
         debug_print(fn, "Warning: Could not convert original audio to mel spectrogram for visualization.")
    else:
        debug_print(fn, "Visualizing original spectrogram.")
        visualize_mel_spectrogram(
            original_mel_spec_db,
            title="Original Mel Spectrogram",
            output_path=original_spectrogram_path,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH
        )
        debug_print(fn, f"Original spectrogram visualized and saved to {original_spectrogram_path}")

    # --- Initialize and Fit the Tokenizer ---
    debug_print(fn, "Initializing SpectrogramTokenizer and fitting.")
    tokenizer = SpectrogramTokenizer(n_clusters=N_CLUSTERS)

    # Use multiple files for fitting if available, for better clustering
    # For this test, we'll still use just the dummy file.
    file_paths_for_fitting = [input_audio_path]
    spectrogram_fitting_iterator = spectrogram_from_files_iterator(file_paths_for_fitting)

    try:
        tokenizer.fit(spectrogram_fitting_iterator)
        debug_print(fn, "Tokenizer fitted successfully.")
        debug_print(fn, f"Tokenizer vocab size: {tokenizer.vocab_size}")
        debug_print(fn, f"Tokenizer ignore_index: {tokenizer.ignore_index}")
        debug_print(fn, f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
        debug_print(fn, f"Tokenizer special tokens: {tokenizer.special_token_map}")

    except Exception as e:
         debug_print(fn, f"Error during tokenizer fitting: {str(e)}")
         sys.exit(1) # Exit if fitting fails


    # --- LLM Encode: Convert raw audio to token IDs ---
    debug_print(fn, "Encoding original audio data into token IDs.")
    audio_token_ids = tokenizer.encode(original_audio_data)
    if audio_token_ids is None or audio_token_ids.size == 0:
        raise RuntimeError(f"{fn}: Encoding original audio resulted in no tokens.")

    debug_print(fn, f"Encoded audio into {audio_token_ids.size} token IDs.")
    debug_print(fn, f"First 20 encoded token IDs: {audio_token_ids[:20]}")
    debug_print(fn, f"String representation (first 20): {tokenizer.tokens_to_strings(audio_token_ids[:20])}")

    # Test encoding different input types
    debug_print(fn, "\nTesting different encode input types:")
    if audio_token_ids.size >= 5:
        test_str_list_repr = str(list(audio_token_ids[:5]))
        debug_print(fn, f"Encoding string list repr '{test_str_list_repr}': {tokenizer.encode(test_str_list_repr)}")
    else:
         debug_print(fn, "Skipping string list repr test: not enough encoded tokens.")

    # Assuming '<|PREFIX LM|>' is defined in audio_ul2_config and used as a paradigm string
    test_special_str = next(iter(tokenizer.special_token_map.keys()), None) # Get the first special token string
    if test_special_str is not None and test_special_str == "[PAD]" and len(tokenizer.special_token_map) > 1:
         test_special_str = next(iter(list(tokenizer.special_token_map.keys())[1:]), test_special_str) # Get the second, or fall back
    if test_special_str is not None:
        debug_print(fn, f"Encoding special string '{test_special_str}': {tokenizer.encode(test_special_str)}")
    else:
        debug_print(fn, "Skipping special string test: no special tokens defined.")

    # Create a mixed list including special tokens and string representations
    test_list_mixed_items = []
    non_pad_special_token = None
    for token_str, token_id in tokenizer.special_token_map.items():
         if token_str != "[PAD]":
              non_pad_special_token = token_str
              break

    if non_pad_special_token is not None:
        test_list_mixed_items.append(non_pad_special_token)
    if audio_token_ids.size > 0:
        test_list_mixed_items.append(audio_token_ids[0].item()) # Single audio token ID (int)
    test_list_mixed_items.append("[PAD]") # PAD token string
    if audio_token_ids.size > 8:
        test_list_mixed_items.append(str(list(audio_token_ids[5:8]))) # String list repr of audio tokens

    if audio_token_ids.size > 10:
        test_single_int = audio_token_ids[10].item()
        debug_print(fn, f"Encoding single integer {test_single_int}: {tokenizer.encode(test_single_int)}")
    else:
        debug_print(fn, "Skipping single integer encode test: not enough encoded tokens.")

    if audio_token_ids.size > 20:
        test_numpy_array_ids = audio_token_ids[15:20]
    else:
        debug_print(fn, "Skipping numpy array encode test: not enough encoded tokens.")

    debug_print(fn, "-" * 20)

    # --- Simulate LLM interaction ---
    debug_print(fn, "Simulating LLM interaction...")
    llm_output_token_ids = audio_token_ids.copy()

    # Add special tokens and invalid tokens to the simulated LLM output sequence
    special_start_token_id = tokenizer.get_special_token_id("<|PREFIX LM|>") # Using the actual paradigm string
    special_mask_token_id = tokenizer.get_special_token_id("[MASK]")
    pad_token_id_test = tokenizer.get_special_token_id("[PAD]") # Using PAD token ID

    # Example: Prepend a start token (if it exists)
    llm_output_token_ids = np.insert(llm_output_token_ids, 0, special_start_token_id)

    # Example: Replace a few audio tokens with a mask token (if it exists and sequence is long enough)
    llm_output_token_ids[10:15] = special_mask_token_id

    # Example: Insert a padding token (if it exists and sequence is long enough)
    llm_output_token_ids = np.insert(llm_output_token_ids, 5, pad_token_id_test)

    # Simulate an unexpected negative token ID
    if llm_output_token_ids.size > 1:
        llm_output_token_ids[1] = -999 # Use a value clearly outside the audio/special token range

    # Simulate an unexpected out-of-vocab token ID
    if llm_output_token_ids.size > 2:
        # Use a value larger than the tokenizer's vocab size
        out_of_vocab_id = tokenizer.vocab_size + 500
        llm_output_token_ids[2] = out_of_vocab_id


    # --- LLM Decode: Convert token IDs back to raw audio ---
    try:
        reconstructed_audio_llm = tokenizer.decode(llm_output_token_ids)
        debug_print(fn, f"Decoded tokens back {reconstructed_audio_llm.shape}.")

        # --- Save the reconstructed audio ---
        save_success = save_audio(reconstructed_audio_llm, output_reconstructed_audio_path, SAMPLE_RATE)

        if save_success:

            # --- Visualize the final reconstructed spectrogram (from saved audio) ---
            try:
                saved_reconstruction_audio, sr_recon = librosa.load(output_reconstructed_audio_path, sr=SAMPLE_RATE)

                saved_reconstruction_spec = convert_audio_to_mel(saved_reconstruction_audio)

                if saved_reconstruction_spec is not None:
                    debug_print(fn, "Visualizing full roundtrip .")
                    visualize_mel_spectrogram(
                        saved_reconstruction_spec,
                        title="Full Roundtrip Reconstructed Mel Spectrogram (LLM Encode/Decode)",
                        output_path=reconstructed_spectrogram_path,
                        sr=SAMPLE_RATE,
                        hop_length=HOP_LENGTH
                    )
                    debug_print(fn, f"Full roundtrip spectrogram done")
                else:
                    debug_print(fn, "Could not generate spectrogram .")

            except Exception as e:
                 debug_print(fn, f"Error processing saved reconstructed audio: {str(e)}")

        else:
            debug_print(fn, "Saving reconstructed audio failed.")

        # --- Also visualize the spectrogram obtained directly from decoding ---
        # Filter out non-audio tokens manually here to pass to _detokenize_audio_tokens
        audio_tokens_from_llm_output_filtered = llm_output_token_ids[(llm_output_token_ids >= 0) & (llm_output_token_ids < tokenizer.n_clusters)]

        if audio_tokens_from_llm_output_filtered.size > 0:
             debug_print(fn, f"Detokenizing audio tokens from simulated LLM output.")
             try:
                  detok_spec_from_decode = tokenizer._detokenize_audio_tokens(audio_tokens_from_llm_output_filtered)
                  if detok_spec_from_decode is not None:
                       visualize_mel_spectrogram(
                            detok_spec_from_decode,
                            title="Detokenized Mel Spectrogram (from Decode Output before audio conv)",
                            output_path=detokenized_spectrogram_from_decode_path,
                            sr=SAMPLE_RATE,
                            hop_length=HOP_LENGTH
                       )
                  else:
                       debug_print(fn, "Could not detokenize audio tokens .")
             except Exception as e:
                  debug_print(fn, f"Error during direct detokenization: {str(e)}")

        else:
             debug_print(fn, "No audio tokens found in simulated LLM.")


    except Exception as e:
         debug_print(fn, f"Error during LLM Decode or saving: {str(e)}")
         debug_print(fn, "LLM Decode and saving process failed.")

    debug_print(fn, "\nSpectrogram Tokenizer Test Script Finished.\n")
