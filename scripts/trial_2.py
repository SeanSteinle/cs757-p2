import sys
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
import accelerate
import random
from datetime import datetime  

# Add 'birdie' directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'birdie'))
from birdie_rl import Birdie
from ul2_config import ul2_config
from audio_ul2_config import audio_ul2_config, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, DURATION, N_CLUSTERS, DEBUG

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")
        
# Function to load an audio file and convert it to a Mel spectrogram
def load_audio_to_mel(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        # debug_print(f"Loaded {os.path.basename(file_path)} ({len(y)} samples)")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                                hop_length=HOP_LENGTH, n_mels=N_MELS)
        return mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Tokenization: Convert spectrograms into discrete tokens using KMeans clustering
class SpectrogramTokenizer:  
    def __init__(self, n_clusters=N_CLUSTERS):  
        self.n_clusters = n_clusters  
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)  
        self.cluster_centers = None  

    def fit(self, spectrograms):
        debug_print(f"Starting K-Means fitting with {self.n_clusters} clusters")
        all_frames = np.concatenate([spec.T for spec in spectrograms], axis=0)
        debug_print(f"Total frames for clustering: {all_frames.shape[0]}")
        
        self.kmeans.fit(all_frames)
        self.cluster_centers = self.kmeans.cluster_centers_
        debug_print("K-Means fitting completed")

    def tokenize(self, spectrogram):
        debug_print(f"Tokenizing spectrogram with shape {spectrogram.shape}")

        frames = spectrogram.T
        tokens = self.kmeans.predict(frames)
        debug_print(f"Generated {len(tokens)} tokens")
        
        return tokens.astype(int).tolist()    
    
    # def detokenize(self, tokens):
    #     if self.cluster_centers is None:
    #         raise ValueError("Tokenizer not fitted. Call fit() first.")
        
    #     debug_print(f"Detokenizing {len(tokens)} tokens")
    #     reconstructed_spectrogram = np.array([self.cluster_centers[t] for t in tokens]).T
    #     debug_print(f"Reconstructed spectrogram shape: {reconstructed_spectrogram.shape}")
        
    #     return reconstructed_spectrogram
    
    def detokenize(self, tokens):  
        debug_print(f"Input tokens type: {type(tokens)}")  
        debug_print(f"First 10 input tokens: {tokens[:10]}")  
        
        arr_tokens = np.array(tokens, dtype=int)  
        debug_print(f"Array tokens shape: {arr_tokens.shape}")  
        
        debug_print(f"Pre-clip range: {np.min(arr_tokens)}-{np.max(arr_tokens)}")  
        
        safe_tokens = np.clip(arr_tokens, 0, self.n_clusters-1)  
        debug_print(f"Post-clip range: {np.min(safe_tokens)}-{np.max(safe_tokens)}")  
        
        reconstructed = np.array([self.cluster_centers[t] for t in safe_tokens]).T  
        debug_print(f"Reconstruction shape: {reconstructed.shape}")  
        
        return reconstructed  

    def encode(self, data):
        if isinstance(data, str): 
            debug_print("Encoding string data")
            print(data, 10*"\n")
            # data = data.replace("tensor([", "").replace("])", "").replace(",", " ")  
            # return data
        
        debug_print("Encoding spectrogram data")
        return self.tokenize(data)

    # def decode(self, tokens):
    #     debug_print("Decoding token sequence")
    #     return self.detokenize(tokens)
    
    def decode(self, tokens):  
        debug_print(f"Decode input type: {type(tokens)}")  
        if isinstance(tokens, str):  
            debug_print(f"Raw string input: {tokens[:10]}...")  
            
            cleaned = tokens.replace("tensor([", "").replace("])", "").replace(",", " ")  
            debug_print(f"Cleaned string: {cleaned[:10]}...")  
            
            # Convert to integers  
            converted = list(map(int, cleaned.split()))  
            debug_print(f"Converted tokens (first 10): {converted[:10]}")  
            return self.detokenize(converted)  
        
        return self.detokenize(tokens) 

# Prepare dataset for Birdie training
def prepare_dataset(data_dir, max_files=None):
    if max_files:
        audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if f.endswith('.wav')][:max_files]
    else:
        audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if f.endswith('.wav')]

    if not audio_files:
        raise ValueError(f"No .wav files found in directory: {data_dir}")

    debug_print(f"Processing {len(audio_files)} audio files")
    
    spectrograms = []
    failed_files = 0
    for f in audio_files:
        spec = load_audio_to_mel(f)
        if spec is not None:
            spectrograms.append(spec)
            # debug_print(f"Processed {os.path.basename(f)} with Mel shape: {spec.shape}")
        else:
            failed_files += 1

    debug_print(f"Successfully processed {len(spectrograms)} files; Failed: {failed_files}")
    
    if not spectrograms:
        raise ValueError("No valid spectrograms were generated. Check your audio files.")

    # Fit tokenizer 
    debug_print(f"Fitting tokenizer on {len(spectrograms)} spectrograms")
    tokenizer = SpectrogramTokenizer()
    tokenizer.fit(spectrograms)

    # Tokenize all spectrograms
    tokenized_data = [tokenizer.tokenize(spec) for spec in spectrograms]
    
    # Validate tokenization on one sample
    # if DEBUG:
    #     test_spec = spectrograms[0]
    #     tokens = tokenizer.tokenize(test_spec)
    #     debug_print(f"Tokenization check: {len(tokens)} tokens from shape {test_spec.shape}")
        
    return tokenized_data, tokenizer, spectrograms

# Reward function (from Birdie paper)
def reward_fn(action_taken=None, old_loss=None, new_loss=None, old_step_idx=None, new_step_idx=None):
    delta_loss = (new_loss - old_loss)
    rv = (delta_loss / (old_loss + 1e-8))
    n = ((new_loss * old_loss).sqrt() * rv.pow(3) * torch.e)
    reward = (-100 * torch.tanh(n) * torch.e)
    reward = torch.where(torch.isnan(reward), 0.0, reward)
    reward = torch.clamp(reward, -1.0, 1.0)
    return reward

def text_grabber_fn_audio(x):  
    # Handle dictionary format  
    if isinstance(x, dict):  
        tokens = x["input_ids"]  
    # Handle raw token list format  
    elif isinstance(x, (list, np.ndarray, torch.Tensor)):  
        tokens = x  
    else:  
        raise ValueError(f"Unexpected data format: {type(x)}")  
    
    return " ".join(map(str, tokens))  

# Data generator function compatible with Birdie pipeline  
def data_generator(split, worker_id, num_workers, rng_seed=0):  
    genre = config['genre']  
    tokenized_data_path = f"../data/tokenized_audio/{genre}/"  
    debug_print(f"Data generator called: worker_id={worker_id}, num_workers={num_workers}, genre={genre}")  
    
    if not os.path.exists(tokenized_data_path):  
        debug_print(f"Tokenized data path '{tokenized_data_path}' does not exist, creating it")  
        os.makedirs(tokenized_data_path)  

        # Check if we already have the tokenizer in config  
        if 'tokenizer' in config and config['tokenizer'] is not None:  
            debug_print("Using existing tokenizer from config")  
            tokenizer = config['tokenizer']  
            data_dir = f"../data/genres_original/{genre}/"  
            
            # Just use existing tokenizer  
            audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]  
            
            if config['max_files'] is not None:  
                audio_files = audio_files[:config['max_files']]  
                debug_print(f"Limiting to {config['max_files']} files as specified in config")  
            
            debug_print(f"Found {len(audio_files)} audio files to process")  
    
            # Process spectrograms  
            debug_print("Converting audio files to spectrograms")  
            spectrograms = []  
            failed_files = 0  
            for f in audio_files:  
                spec = load_audio_to_mel(f)  
                if spec is not None:  
                    spectrograms.append(spec)  
                else:  
                    failed_files += 1  
            
            debug_print(f"Successfully processed {len(spectrograms)} spectrograms; Failed: {failed_files}")  
            
            # Tokenize using existing tokenizer  
            debug_print("Tokenizing spectrograms with existing tokenizer")  
            tokenized_data = [tokenizer.tokenize(spec) for spec in spectrograms]  
            debug_print(f"Tokenized {len(tokenized_data)} spectrograms")  
        else:  
            debug_print("No existing tokenizer found, processing everything from scratch")  
            # Process everything from scratch  
            data_dir = f"../data/genres_original/{genre}/"  
            max_files = config['max_files']  
            debug_print(f"Calling prepare_dataset with max_files={max_files}")  
            tokenized_data, tokenizer, _ = prepare_dataset(data_dir, max_files=max_files)  
            config['tokenizer'] = tokenizer  
            debug_print("Tokenizer added to config")  
        
        debug_print(f"Saving tokenized data to {tokenized_data_path}")  
        torch.save(  
            [" ".join(map(str, tokens)) for tokens in tokenized_data],  # Store as clean strings  
            os.path.join(tokenized_data_path, "tokenized_data.pt")
            )          
        torch.save(tokenizer.kmeans.cluster_centers_, os.path.join(tokenized_data_path, "cluster_centers.pt"))  
    else:  
        debug_print(f"Tokenized data already exists at '{tokenized_data_path}', loading it")  

    tokenized_data_raw = torch.load(os.path.join(tokenized_data_path, "tokenized_data.pt"), weights_only=False)  
    tokenized_data = [  
        list(map(int, tokens.split()))  # Convert string to integer list  
        for tokens in tokenized_data_raw 
    ]  
    debug_print(f"Loaded {len(tokenized_data)} tokenized spectrograms")  

    # Batch the tokenized data  
    batched_data = [tokenized_data[i:i + config['batch_size']] for i in range(0, len(tokenized_data), config['batch_size'])]  
    debug_print(f"Created {len(batched_data)} batches with batch_size={config['batch_size']}")  
    
    shard_size = len(batched_data) // num_workers  
    start_idx = worker_id * shard_size  
    end_idx = start_idx + shard_size if worker_id < num_workers - 1 else len(batched_data)  
    
    debug_print(f"Worker {worker_id} assigned batches {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")  

    # Return list of dictionaries instead of raw lists  
    return [{  
        "input_ids": torch.tensor(  
            np.array(list(map(int, batch.split())), dtype=int)  # Explicit conversion  
            if isinstance(batch, str)   
            else batch  
        ),  
        "attention_mask": torch.ones(len(batch))  
    } for batch in batched_data[start_idx:end_idx]]  



# Training loop
def train_birdie(birdie_model, config, num_epochs=1):  
    optimizer = torch.optim.Adam(birdie_model.parameters(), lr=1e-3)  
    
    debug_print("Starting training loop")  
    progress_bar = tqdm(total=config['num_steps'], desc="Training")  
    
    for step_idx in range(config['num_steps']):  
        # Get training sample using Birdie's objective-aware sampler  
        train_batch = birdie_model.get_next_training_sample()  
        debug_print(f"Step {step_idx+1} - Batch shape: {[b.shape if hasattr(b, 'shape') else len(b) for b in train_batch]}")  
        
        # Process the batch for model input  
        padded_batch = torch.nn.utils.rnn.pad_sequence(  
            [torch.tensor(seq) for seq in train_batch],   
            batch_first=True,   
            padding_value=0  
        )  
        
        # Forward pass  
        outputs = birdie_model(padded_batch.to(config['accelerator'].device))  
        loss = torch.nn.functional.mse_loss(outputs, padded_batch.float())  
        debug_print(f"Step loss: {loss.item():.4f}")  
        
        # Backward pass  
        optimizer.zero_grad()  
        config['accelerator'].backward(loss)  
        optimizer.step()  
        
        # Validation/reward updating  
        if birdie_model.time_for_eval(step_idx):  
            debug_print("Running validation...")  
            validation_losses = {}  
            
            # Process each validation batch and objective  
            for obj_name, batch in birdie_model.measure_validation_losses():  
                debug_print(f"Validating {obj_name} with batch of shape {[b.shape if hasattr(b, 'shape') else len(b) for b in batch]}")  
                padded_val = torch.nn.utils.rnn.pad_sequence(  
                    [torch.tensor(seq) for seq in batch],  
                    batch_first=True,  
                    padding_value=0  
                )  
                val_outputs = birdie_model(padded_val.to(config['accelerator'].device))  
                val_loss = torch.nn.functional.mse_loss(val_outputs, padded_val.float())  
                debug_print(f"Validation loss ({obj_name}): {val_loss.item():.4f}")  
                
                # Log validation loss to Birdie - THIS IS CRITICAL  
                birdie_model.log_validation_loss(obj_name, val_loss.item(), step_idx)  
                validation_losses[obj_name] = val_loss.item()  
            
            # current objective probabilities  
            current_probs = birdie_model.get_current_action()  
            debug_print(f"Current objective probabilities: {[(obj.get('name', f'obj{i}'), round(prob, 3)) for i, (obj, prob) in enumerate(zip(birdie_model.objectives, current_probs))]}")  
        
        progress_bar.update(1)  
    
    progress_bar.close()  
    debug_print("Training completed successfully")  

def generate_music(birdie_model, config, initial_spectrogram=None, max_length=256):  
    debug_print("Starting music generation...")  
    birdie_model.eval()  
    
    with torch.no_grad():  
        # Start with a seed or create one  
        if initial_spectrogram is None:  
            seed_tokens = torch.randint(0, N_CLUSTERS, (1, 8)).to(config['accelerator'].device)  
            debug_print(f"Using random seed tokens: {seed_tokens}")  
        else:  
            # Encode the initial spectrogram into tokens  
            seed_tokens = torch.tensor(config['tokenizer'].encode(initial_spectrogram)).unsqueeze(0)  
            seed_tokens = seed_tokens[:, :min(seed_tokens.size(1), 8)]  # Limit to first 8 tokens  
            seed_tokens = seed_tokens.to(config['accelerator'].device)  
            debug_print(f"Using seed from spectrogram, shape: {seed_tokens.shape}")  

        # Generate using Birdie's RL-optimized generation  
        debug_print(f"Generating sequence with max_length={max_length}")  
        generated_tokens = []  
        
        current_tokens = seed_tokens.clone()  
        
        # Generate one token at a time, using the model's predictions  
        for i in range(max_length):  
            # Get model prediction for next token  
            logits = birdie_model(current_tokens)  
            next_token_logits = logits[:, -1, :]  
            
            # Sample from the distribution  
            next_token_probs = torch.softmax(next_token_logits / 0.8, dim=-1)  # temperature=0.8  
            next_token = torch.multinomial(next_token_probs, num_samples=1)  
            
            generated_tokens.append(next_token.item())  
            
            # Update current_tokens for next iteration  
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)  
            
            if i % 50 == 0:  
                debug_print(f"Generated {i}/{max_length} tokens")  
        
        debug_print(f"Generated sequence of {len(generated_tokens)} tokens")  
        
        # Detokenize the tokens back into a spectrogram  
        generated_spectrogram = config['tokenizer'].decode(generated_tokens)  
        
        # Convert the spectrogram to audio  
        debug_print("Converting spectrogram to audio...")  
        generated_audio = librosa.feature.inverse.mel_to_audio(  
            librosa.db_to_power(generated_spectrogram),   
            sr=SAMPLE_RATE,   
            n_fft=N_FFT,   
            hop_length=HOP_LENGTH  
        )  
        
        # Save the generated audio  
        output_file = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"  
        debug_print(f"Saving audio to {output_file}")  
        librosa.output.write_wav(output_file, generated_audio, SAMPLE_RATE)  
        
        return output_file, generated_spectrogram, generated_tokens  

#%%
if __name__ == "__main__":  
    # Configuration for Birdie training
    config = {
        "batch_size": 1,
        "max_files": 10,
        "sequence_length": 256,
        "num_workers": 2, 
        "steps_between_evaluations": 1,
        "num_steps": 1,
        "genre": "pop", 
        "text_grabber_fn": text_grabber_fn_audio,
        "start_generating_paradigm": 1024,
    }
        
    # Create and add tokenizer to config
    data_dir = f"../data/genres_original/{config['genre']}/"
    tokenized_data, tokenizer, spectrograms = prepare_dataset(data_dir, max_files=config['max_files']) 
    config['tokenizer'] = tokenizer
    
    config['accelerator'] = accelerate.Accelerator()
    config['reward_fn'] = reward_fn
    # config['objectives'] = ul2_config
    config['objectives'] = audio_ul2_config
    
    config['ds'] = data_generator
    
    #%%
    # Initialize and train Birdie model
    birdie_model = Birdie(config)
    train_birdie(birdie_model, config)
    
    # Generate music  
    debug_print("Generating music...")  
    if spectrograms:  
        output_file, _, _ = generate_music(  
            birdie_model,   
            config,   
            initial_spectrogram=spectrograms[0],  
            max_length=512  
        )  
        debug_print(f"Generated music saved to {output_file}")  
    else:  
        debug_print("No spectrograms available for seeding generation")  
        output_file, _, _ = generate_music(birdie_model, config)  
        debug_print(f"Generated music saved to {output_file}")  
    
    birdie_model.close()
    
    print("Training and music generation complete.")
