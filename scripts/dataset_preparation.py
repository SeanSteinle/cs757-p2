# dataset_preparation.py

import sys
import os
import numpy as np
import random
from tqdm import tqdm
import ast
from itertools import chain
from datetime import datetime  

# Add 'birdie' directory 
birdie_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'birdie')
if birdie_path not in sys.path:
    sys.path.insert(0, birdie_path)
import soundfile as sf

from audio_processing import load_audio_file, convert_audio_to_mel
from tokenizer import SpectrogramTokenizer
from audio_ul2_config import (
     N_CLUSTERS, audio_ul2_config, validation_objectives_config , DURATION, SAMPLE_RATE, IGNORE_INDEX
)
from utils import debug_print

#%%
def text_grabber_fn_audio(x):
    fn = "text_grabber_fn_audio"
    if isinstance(x, dict):
        text_data = x.get("text", "")
        if isinstance(text_data, str):
             return text_data
        return "" 
    return ""


# Prepare dataset for Birdie training
def prepare_dataset(data_dir, max_files, config): 
    fn = "prepare_dataset"

    # Find audio files
    if not os.path.isdir(data_dir):
         raise FileNotFoundError(f"{fn}: Data directory not found: {data_dir}")

    audio_files = [os.path.join(data_dir, f)
                   for f in os.listdir(data_dir)
                   if f.endswith('.wav')]

    if max_files is not None:
        audio_files = audio_files[:max_files]

    if not audio_files:
        raise FileNotFoundError(f"{fn}: No WAV files found in {data_dir} ")

    tokenizer = SpectrogramTokenizer(n_clusters=N_CLUSTERS)


    def fitting_spectrogram_iterator():
        for file_path in tqdm(audio_files, desc=f"{fn}: Generating spectrograms for tokenizer"):
            try:
                audio_data = load_audio_file(file_path)
                if audio_data is not None and audio_data.size > 0:
                     spectrogram = convert_audio_to_mel(audio_data)
                     if spectrogram is not None and spectrogram.shape[1] > 0:
                         yield spectrogram
            except Exception as e:
                 debug_print(fn, f"Error processing {file_path} for fitting: {str(e)}")

    iterator_for_check = fitting_spectrogram_iterator()
    try:
        first_item = next(iterator_for_check)
        full_fitting_iterator = chain([first_item], iterator_for_check)
    except StopIteration:
        raise ValueError(f"{fn}: No valid spectrograms : from the audio files : tokenizer fitting.")


    try:
        tokenizer.fit(full_fitting_iterator)

    except Exception as e:
         raise RuntimeError(f"{fn}: Tokenizer fitting failed: {str(e)}")

    return {
        "audio_files": audio_files,
        "tokenizer": tokenizer,
    }

def data_generator(split, worker_id, num_workers, rng_seed, config=None):
    fn = f"data_generator (worker {worker_id}/{num_workers})"
        
    if config is None:
        config = getattr(data_generator, 'config', {})    
    
    base_data_dir = config.get('base_data_dir')
    genre = config.get('genre')
    max_files = config.get('max_files')
    data_dir = os.path.join(base_data_dir, "genres_original", genre)

    all_audio_files = [os.path.join(data_dir, f)
                       for f in os.listdir(data_dir)
                       if f.endswith('.wav')]

    if max_files is not None:
        all_audio_files = all_audio_files[:max_files]

    tokenizer = config.get('tokenizer')
    if tokenizer is None:
        tokenizer = getattr(data_generator, 'tokenizer', None)

    if split == 'train':
         available_objective_configs = config.get('objectives', [])
    elif split == 'validation':
         available_objective_configs = config.get('validation_objectives', [])
    else:
         available_objective_configs = []

    available_objective_names = [obj['name'] for obj in available_objective_configs]

    rng = np.random.default_rng(rng_seed + worker_id)
    shuffled_files = all_audio_files[:]
    rng.shuffle(shuffled_files)

    total_files = len(shuffled_files)
    base_shard_size = total_files // num_workers
    remainder = total_files % num_workers
    shard_size = base_shard_size + (1 if worker_id < remainder else 0)
    start = base_shard_size * worker_id + min(worker_id, remainder)
    end = start + shard_size
    worker_files = shuffled_files[start:end]
    
    # Use tqdm if this is the main process
    is_main_process = False
    if 'accelerator' in config and config['accelerator'] is not None:
        is_main_process = config['accelerator'].is_main_process
    elif worker_id == 0 and num_workers == 1:
        is_main_process = True

    file_iterator = tqdm(worker_files, desc=f"{fn}: Processing audio files for '{split}'") \
        if is_main_process else worker_files

    processed_samples = []
    
    for file_path in file_iterator:
        try:
            audio_data = load_audio_file(file_path)

            if audio_data is None or audio_data.size == 0:
                 continue

            audio_token_ids = tokenizer.encode_audio(audio_data)

            if audio_token_ids is None or audio_token_ids.size == 0:
                 print(fn, f"Warning: Skipping {file_path}: Encoding resulted in no audio tokens.")
                 continue

            token_ids_string = str(audio_token_ids.tolist())

            assigned_objective_name = random.choice(available_objective_names)

            processed_sample = {
                'text': token_ids_string,
                # 'original_token_ids': audio_token_ids.astype(np.int32),
                'file_path': file_path,
                'objective_name': assigned_objective_name,
            }
            processed_samples.append(processed_sample)

        except Exception as e:
             continue

    return processed_samples


#%%
# Example of how prepare_dataset and data_generator would be used in a script (like main.py)
if __name__ == "__main__":
    fn = f"*{os.path.basename(__file__)} (test)*"
    print(f"\nRunning Script {fn}\n")

    example_config = {
        "sequence_length": 256, 
        "num_workers": 2, 
        "rng_seed": 23,
        "genre": "blues",
        "base_data_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'), 
        "max_files": 5, 
        "objectives": audio_ul2_config, 
        "validation_objectives": validation_objectives_config,
        "tokenizer": None, 
        "IGNORE_INDEX": IGNORE_INDEX, 
        "start_generating_paradigm": "\n<|assistant|>\n",
    }

    dummy_data_dir = os.path.join(example_config['base_data_dir'], "genres_original", example_config['genre'])
    if not os.path.exists(dummy_data_dir):
         os.makedirs(dummy_data_dir, exist_ok=True)

    # Create a few dummy audio files for testing sharding
    num_dummy_files_needed = example_config['max_files'] if example_config['max_files'] is not None else 10 
    dummy_file_paths = []
    for i in range(num_dummy_files_needed):
        dummy_audio_path = os.path.join(dummy_data_dir, f"dummy_test_audio_{i:03d}.wav")
        dummy_file_paths.append(dummy_audio_path)
        if not os.path.exists(dummy_audio_path):
             debug_print(fn, f"Creating dummy audio file {i+1}/{num_dummy_files_needed} at {dummy_audio_path}")
             # Generate a simple sine wave
             t = np.linspace(0., DURATION * (1 + i % 3), int(DURATION * (1 + i % 3) * SAMPLE_RATE)) 
             amplitude = np.iinfo(np.int16).max * 0.1
             data = amplitude * np.sin(2. * np.pi * 440. * t)
             try:
                  sf.write(dummy_audio_path, data.astype(np.int16), SAMPLE_RATE)
             except Exception as e:
                  debug_print(fn, f"Error creating dummy audio file {dummy_audio_path}: {str(e)}")

    class MockAccelerator:
        def get_local_process_index(self):
             if not hasattr(self, '_current_worker'):
                  self._current_worker = 0
             return self._current_worker

        def set_current_worker(self, worker_id):
             self._current_worker = worker_id

        def get_num_processes(self):
             return example_config.get("num_workers", 1)

        def is_main_process(self):
             return self.get_local_process_index() == 0 # Main process is worker 0

        # Add a simple wait_for_everyone simulation for structure
        def wait_for_everyone(self):
             debug_print(fn, f"Worker {self.get_local_process_index()} waiting for everyone.")
             # In a real setup, this syncs processes. Here, just a print.
        # Add a minimal print method for testing
        def print(self, *args, **kwargs):
             print(*args, **kwargs)

    mock_accelerator = MockAccelerator()
    example_config['accelerator'] = mock_accelerator 

    data_dir_to_process = os.path.join(example_config['base_data_dir'], "genres_original", example_config['genre'])

    if mock_accelerator.is_main_process():
         try:
              dataset_info = prepare_dataset(data_dir_to_process, example_config.get('max_files'), example_config)
              example_config['tokenizer'] = dataset_info['tokenizer'] # Add fitted tokenizer to config
         except Exception as e:
              import traceback
              debug_print(fn, traceback.format_exc())
              sys.exit(1) 

    mock_accelerator.wait_for_everyone()

    # --- Step 2: Simulate running data_generator for each worker 

    # Ensure tokenizer is in config before simulating data_generator
    if 'tokenizer' not in example_config or example_config['tokenizer'].cluster_centers is None:
        debug_print(fn, "Skipping data_generator simulation .")
    else:
        num_workers = mock_accelerator.get_num_processes()
        rng_seed = example_config.get("rng_seed", 23)
        num_train_samples_to_fetch_per_worker = 3 # Fetch a few samples 

        # Simulate each worker calling the data_generator
        for worker_id in range(num_workers):
            mock_accelerator.set_current_worker(worker_id) # Set the mock worker ID

            debug_print(fn, f"--- Simulating Worker {worker_id} (Train) ---")

            train_gen_fn = data_generator(
                'train',
                worker_id,
                num_workers,
                rng_seed,
                example_config 
            )

            # Call the returned generator function to get the actual iterator
            train_iterator = train_gen_fn()

            # Fetch a few samples from the generator
            fetched_train_samples = []
            try:
                 debug_print(fn, f"Worker {worker_id}: Fetching {num_train_samples_to_fetch_per_worker} training samples...")
                 for i in range(num_train_samples_to_fetch_per_worker):
                      sample = next(train_iterator) # Get the next sample from the iterator
                      fetched_train_samples.append(sample)

                 # Test fetching one more sample to see if it continues (train generator is infinite)
                 debug_print(fn, f"Worker {worker_id}: Fetching one more sample to confirm infinite generator...")
                 try:
                      sample = next(train_iterator)
                      debug_print(fn, f"Worker {worker_id} successfully fetched sample (generator is infinite).")
                 except StopIteration:
                      debug_print(fn, f"Worker {worker_id}'s failed.")

            except Exception as e:
                 debug_print(fn, f"ERROR failed fetching training sample for Worker {worker_id}: {str(e)}")
                 import traceback
                 debug_print(fn, traceback.format_exc())

    debug_print(fn, "\nDataset preparation and generator simulation complete.")
