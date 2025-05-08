# main.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn 
import accelerate
from datetime import datetime
from tqdm import tqdm
import traceback
import gc

# Add 'birdie' directory to paths
birdie_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'birdie')
if birdie_path not in sys.path:
    sys.path.insert(0, birdie_path)

from birdie_rl import Birdie

from model import SimpleTransformer, generate_audio_from_model
from audio_ul2_config import (
    audio_ul2_config, 
    validation_objectives_config, 
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT,
    N_CLUSTERS, DEFAULT_VOCAB_SIZE, IGNORE_INDEX
)
from audio_processing import load_audio_file, mel_to_audio, save_audio
from tokenizer import SpectrogramTokenizer
from dataset_preparation import prepare_dataset, data_generator, text_grabber_fn_audio
from utils import debug_print

#%%
class DualOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = None 

        try:
            log_dir = os.path.dirname(filename)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.log = open(filename, "w", encoding='utf-8', buffering=1) 
        except Exception as e:
            print(f"Error opening log file {filename}: {e}", file=sys.__stderr__)
            self.log = None 

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception as e:
             print(f"Warning: Error writing to terminal: {e}", file=sys.__stderr__)

        if self.log:
            try:
                self.log.write(message)
                self.log.flush() 
            except Exception as e:
                 print(f"Error writing to log file: {e}", file=sys.__stderr__)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception as e:
             print(f"Error flushing terminal: {e}", file=sys.__stderr__)

        if self.log:
            try:
                self.log.flush()
            except Exception as e:
                print(f"Error flushing log file: {e}", file=sys.__stderr__)


    def close_log(self):
        """ Safely closes the log file if it's open. """
        if self.log and not self.log.closed:
            try:
                self.log.close()
            except Exception as e:
                print(f"Error closing log file: {e}", file=sys.__stderr__)
            self.log = None 

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_dir = "logs" 
output_file_path = os.path.join(log_file_dir, f"pipeline_output_{timestamp}.log")

original_stdout = sys.stdout
dual_output_instance = None 

try:
    dual_output_instance = DualOutput(output_file_path)
    sys.stdout = dual_output_instance
    debug_print("DualOutput", f"Pipeline output saved to {output_file_path}.")
except Exception as e:
    print(f"Error setting up dual: {e}", file=original_stdout)
    sys.stdout = original_stdout 
#%%
# --- reward function ---
def reward_fn(action_taken=None, old_loss=None, new_loss=None, old_step_idx=None, new_step_idx=None, old_loss_vector=None, **kwargs):
    if old_loss is None or new_loss is None:
        return torch.tensor(0.0)  # Return neutral reward if missing values

    if isinstance(old_loss, torch.Tensor) and old_loss.ndim > 0:
        old_loss = old_loss.mean()
    if isinstance(new_loss, torch.Tensor) and new_loss.ndim > 0:
        new_loss = new_loss.mean()
        

    delta_loss = (new_loss - old_loss)
    rv = (delta_loss / (old_loss + 1e-8))
    n = ((new_loss * old_loss).sqrt() * rv.pow(3) * torch.e)
    reward = (-100 * torch.tanh(n) * torch.e)
    
    reward = torch.where(torch.isnan(reward), torch.tensor(0.0), reward)
    reward = torch.clamp(reward, -1.0, 1.0)
    
    return reward

# --- Training  ---
def train_model(accelerator, birdie_orchestrator, model,optimizer, config, model_checkpoint_path="./checkpoints"):
    fn = "train_model"
    
    if not os.path.exists(model_checkpoint_path):
        os.makedirs(model_checkpoint_path, exist_ok=True)
        
    # training parameters
    num_steps = config.get('num_steps')
    steps_between_evaluations = config.get('steps_between_evaluations')
    log_interval = config.get('log_interval')
    ignore_index_for_loss = config.get('ignore_index')
    checkpoint_interval = config.get('checkpoint_interval', steps_between_evaluations)
    
    # validating params
    if any(v is None for v in [num_steps, steps_between_evaluations, log_interval, ignore_index_for_loss]):
        raise ValueError("missing required params in train model.")

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index_for_loss, reduction='mean')
    
    step_iterator = tqdm(range(num_steps), desc="Training") if accelerator.is_main_process else range(num_steps)
    
    model.train()
    device = next(model.parameters()).device
    
    for global_step in step_iterator:
            
        # validation
        if global_step > 0 and global_step % steps_between_evaluations == 0:
            run_validation(accelerator, birdie_orchestrator, model, config, global_step, criterion)
            model.train()  

        # Save checkpoint
        if global_step > 0 and global_step % checkpoint_interval == 0 and accelerator.is_main_process:
            try:
                config_to_save = {
                    k: v for k, v in config.items() 
                    if k in ['num_steps', 'steps_between_evaluations', 'log_interval']
                }
                
                checkpoint = {
                    'step': global_step,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config_to_save,  
                }
                
                model_save_path = os.path.join(model_checkpoint_path, f"model_step_{global_step}.pt")
                
                torch.save(checkpoint, model_save_path)
                print("Checkpoint saved.")
                
                del checkpoint
                gc.collect()
                
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")

        # Training 
        try:
            train_batch = birdie_orchestrator.get_next_training_sample()
            if train_batch is None:
                print("Received None batch. Exiting training loop.")
                break
                
            def process_batch(batch):
                batch = {k: v.to(device) for k, v in batch.items() 
                        if isinstance(v, torch.Tensor)}
                
                input_ids = batch.get('input_ids')
                label_ids = batch.get('label_ids')
                
                if input_ids is None or label_ids is None:
                    return None
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids=input_ids)
                
                # Loss calculation
                loss = criterion(logits.permute(0, 2, 1), label_ids)
                
                # Skip bad batches
                if torch.isnan(loss) or torch.isinf(loss):
                    return None
                
                # Backward and optimize
                accelerator.backward(loss)
                optimizer.step()
                
                return loss.item()
            
            loss_item = process_batch(train_batch)
            
            if loss_item is not None and accelerator.is_main_process and (global_step + 1) % log_interval == 0:
                print(f"Step {global_step+1}/{num_steps}: Loss = {loss_item:.4f}")
                if isinstance(step_iterator, tqdm):
                    step_iterator.set_postfix({"loss": f"{loss_item:.4f}"})
                    
            del train_batch
            gc.collect()
            
        except Exception as e:
            print(f"ERROR during training step {global_step}: {e}")
            traceback.print_exc()
            continue

    if isinstance(step_iterator, tqdm):
        step_iterator.close()
        
    # Save final model
    if accelerator.is_main_process:
        try:
            final_model_path = os.path.join(model_checkpoint_path, "model_final.pt")
            
            checkpoint = {
                'step': global_step,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            }
            
            torch.save(checkpoint, final_model_path)
            print("Final model saved")
            
        except Exception as e:
            print(f"Failed to save final model: {e}")
    
    print(f"Training completed: {global_step + 1} steps.")
    return model
                
# --- Validation Function ---
def run_validation(accelerator, birdie_orchestrator, model, config, step_idx, criterion):
    fn = "run_validation"
    
    # Basic validation
    if birdie_orchestrator is None or not hasattr(birdie_orchestrator, 'measure_validation_losses'):
        print("Invalid Birdie orchestrator. Skipping validation.")
        return

    if criterion is None:
        print("No criterion provided. Skipping validation.")
        return

    model.eval()
    device = next(model.parameters()).device
    
    try:
        flat_validation_batches = birdie_orchestrator.measure_validation_losses()
        
        if flat_validation_batches is None:
            print("Received None from measure_validation_losses. Skip.")
            return
            
    except Exception as e:
        print(f"Error getting validation batches: {e}")
        return

    validation_iterator = tqdm(flat_validation_batches, desc=f"Validation (Step {step_idx})") \
        if accelerator.is_main_process else flat_validation_batches
    
    total_val_loss = 0.0
    num_val_batches_processed = 0

    with torch.no_grad():
        for item in validation_iterator:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if not isinstance(item, tuple) or len(item) != 2:
                print("Invalid validation item format. Skipping.")
                continue
                
            objective_nickname, val_batch = item

            try:
                def process_val_batch(batch, objective):
                    batch = {k: v.to(device) for k, v in batch.items() 
                            if isinstance(v, torch.Tensor)}
                    
                    input_ids = batch.get('input_ids')
                    label_ids = batch.get('label_ids')
                    
                    if input_ids is None or label_ids is None:
                        return None
                    
                    # Forward pass
                    logits = model(input_ids=input_ids)
                    
                    # Loss calculation
                    val_loss = criterion(logits.permute(0, 2, 1), label_ids)
                    
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        return None
                        
                    # Log to Birdie with error handling
                    val_loss_item = val_loss.item()
                    try:
                        birdie_orchestrator.log_validation_loss(
                            key=objective, 
                            loss=val_loss_item,
                            step_idx=step_idx
                        )
                    except Exception as log_e:
                        print(f"Failed to log to Birdie: {log_e}")
                        
                    return val_loss_item
                
                # Process validation batch
                val_loss_item = process_val_batch(val_batch, objective_nickname)
                
                # Update statistics
                if val_loss_item is not None:
                    total_val_loss += val_loss_item
                    num_val_batches_processed += 1
                    
                    # progress bar
                    if accelerator.is_main_process and isinstance(validation_iterator, tqdm):
                        validation_iterator.set_postfix({
                            "obj": objective_nickname, 
                            "loss": f"{val_loss_item:.4f}"
                        })
                
                del val_batch
                gc.collect()
                
            except Exception as e:
                print(f"ERROR validation peocessing batch: {e}")
                continue

    if isinstance(validation_iterator, tqdm) and hasattr(validation_iterator, 'close'):
        validation_iterator.close()

    # Log 
    if num_val_batches_processed > 0 and accelerator.is_main_process:
        avg_val_loss = total_val_loss / num_val_batches_processed
        print(f"Step {step_idx}: Average validation loss: {avg_val_loss:.4f}")
    elif accelerator.is_main_process:
        print(f"Step {step_idx}: No validation batches processed.")
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
#%%
def generate_music(accelerator, model, config, seed_audio=None, length=512):
    fn = "generate_music"
    if not accelerator.is_main_process:
         return None # Only generate on the main process

    debug_print(fn, "Starting music generation.")
    model.eval()
    # Unwrap the model for generation 
    unwrapped_model = accelerator.unwrap_model(model)
    debug_print(fn, f"Model unwrapped for generation (type: {type(unwrapped_model)}).")

    tokenizer = config.get('tokenizer')
    if tokenizer is None:
         debug_print(fn, "ERROR: Tokenizer not found in config. Cannot generate.")
         return None

    device = next(unwrapped_model.parameters()).device
    debug_print(fn, f"Using model device for generation: {device}")

    # Prepare initial prompt text
    initial_prompt_text = ""
    pad_id = tokenizer.pad_token_id
    start_token_str = config.get('start_generating_paradigm', "<|START|>") 
    start_id = tokenizer.get_special_token_id(start_token_str)


    if seed_audio is None:
        if start_id is None:
             debug_print(fn, f"Warning: Start token '{start_token_str}' not found in tokenizer. \
                         Using PAD/dummy ID as start.")
             start_id = 0 # Use dummy audio token ID 0 as a fallback start

             if start_id >= tokenizer.vocab_size or start_id < 0 or start_id == pad_id:

                  start_id = pad_id
                  if start_id is None:
                       debug_print(fn, "ERROR: Cannot determine a valid start token (START, 0, or PAD). \
                                   Cannot generate from scratch.")
                       return None
        initial_prompt_tokens = [pad_id, pad_id, start_id] # Example: PAD, PAD, START
        initial_prompt_text = "[" + ", ".join(map(str, initial_prompt_tokens)) + "]"
        debug_print(fn, f"Generating from scratch using prompt text: '{initial_prompt_text}'")

    elif isinstance(seed_audio, np.ndarray):
        debug_print(fn, f"Encoding audio seed data with shape {seed_audio.shape}.")
        try:
            seed_tokens_np = tokenizer.encode(seed_audio)
            if seed_tokens_np is None or not isinstance(seed_tokens_np, np.ndarray) or seed_tokens_np.size == 0:
                debug_print(fn, "Warning: Encoding audio seed resulted in no tokens. Generating from scratch.")
                if start_id is None:
                     start_id = 0 # dummy ID
                     if start_id >= tokenizer.vocab_size or start_id < 0 or start_id == pad_id:
                         start_id = pad_id
                         if start_id is None:
                              debug_print(fn, "Cannot determine a valid start token for fallback generation.")
                              return None

                initial_prompt_tokens = [pad_id, pad_id, start_id]
                initial_prompt_text = "[" + ", ".join(map(str, initial_prompt_tokens)) + "]"
                debug_print(fn, f"Falling back to generating from scratch with prompt: '{initial_prompt_text}'")

            else:
                 # format encoded seed tokens into the prompt string format
                 initial_prompt_text = "[" + ", ".join(map(str, seed_tokens_np.astype(int).tolist())) + "]"
                 debug_print(fn, f"Encoded audio seed to {seed_tokens_np.size} tokens. Using as prompt text.")

        except Exception as e:
             debug_print(fn, f"ERROR: Failed to encode seed audio: {e}. Generating from scratch instead.")
             traceback.print_exc()
             # fallback to generating from scratch if seed encoding fails
             if start_id is None:
                  start_id = 0 # dummy ID
                  if start_id >= tokenizer.vocab_size or start_id < 0 or start_id == pad_id:
                      start_id = pad_id
                      if start_id is None:
                           debug_print(fn, "ERROR: Cannot determine a valid start token for fallback generation.")
                           return None
             initial_prompt_tokens = [pad_id, pad_id, start_id]
             initial_prompt_text = "[" + ", ".join(map(str, initial_prompt_tokens)) + "]"
             debug_print(fn, f"Falling back to generating from scratch with prompt: '{initial_prompt_text}'")

    else:
        debug_print(fn, f"ERROR: Invalid seed type: {type(seed_audio)}. \
                    Expected None or numpy.ndarray. Cannot generate.")
        return None

    output_audio_path = None
    try:
        debug_print(fn, f"Calling generate_audio_from_model helper...")
        generated_audio_data = generate_audio_from_model(
            model=unwrapped_model, 
            tokenizer=tokenizer,
            initial_text=initial_prompt_text,
            max_audio_tokens=length, 
            device=device, 
            temperature=config.get('generation_temperature', 0.8),
            top_k=config.get('generation_top_k', 50),
            top_p=config.get('generation_top_p', None)
        )

        # --- Save audio ---
        if generated_audio_data is not None and isinstance(generated_audio_data, np.ndarray) and generated_audio_data.size > 0:
            output_file_name = f"generated_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_dir = config.get("generation_output_dir", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_file_name)
            debug_print(fn, f"Saving generated audio ({generated_audio_data.shape})")
            try:
                save_audio(generated_audio_data, output_path, SAMPLE_RATE) # Use save_audio helper
                output_audio_path = output_path
            except Exception as e:
                 debug_print(fn, f"ERROR: Failed to save generated audio to : {e}")
                 traceback.print_exc()
        elif generated_audio_data is not None:
            debug_print(fn, "Warning: Generation pipeline produced empty audio data.")
        else:
             debug_print(fn, "Generation pipeline failed (returned None/invalid data).")

    except Exception as e:
        debug_print(fn, f"ERROR during generate_audio_from_model call: {e}")
        traceback.print_exc()

    debug_print(fn, "Music generation finished.")
    return output_audio_path

#%%
def load_model_checkpoint(checkpoint_path, model, optimizer=None, config=None, device=None):
    fn = "load_model_checkpoint"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    debug_print(fn, f"Loading checkpoint from {checkpoint_path}")
    
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    debug_print(fn, "Model weights loaded")
    
    # optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        debug_print(fn, "Optimizer state loaded")
    
    step = checkpoint.get('step', 0)
    
    # Update config
    loaded_config = checkpoint.get('config', {})
    if config is not None and isinstance(loaded_config, dict):
        # Only update config if values exist in loaded config
        for key, value in loaded_config.items():
            if key not in config:
                config[key] = value
    
    debug_print(fn, f"Checkpoint loaded from step {step}")
    
    return model, optimizer, step, loaded_config

#%%
if __name__ == "__main__":
    fn = f"*{os.path.basename(__file__)}*"
    print(fn, f"\nRunning Script {fn}\n")

    # --- Configuration ---
    config = {
        # Birdie / data config
        "batch_size": 1, 
        "num_validation_batches": 1, 
        "max_files": 10, # Max files to process for tokenizer fitting and dataset creation
        "sequence_length": 64, # Max sequence length for the model and data batches (must match model!)
        "min_seq_len_for_packing": 64, # Minimum seq length for packing
        "min_chunk_len_to_process": 32, # Minimum length of data chunk for objectives
        "num_workers": 1, # Number of Birdie data workers (processes). 
        "num_bp_train": 1, # Number of batches to prefetch/process for training
        "num_bp_val": 1,   # Number of batches to prefetch/process for validation
        "text_grabber_fn": text_grabber_fn_audio, 
        "start_generating_paradigm": "<|START|>", 
        "base_data_dir": r"C:\Users\kaminfar\OneDrive - George Mason University - O365 Production\CS757\cs757-p2\data",
        "genre": "blues", 
        "objectives": audio_ul2_config, # Training objectives 
        "validation_objectives": validation_objectives_config, # Validation objectives 
        "reward_fn": reward_fn, # Reward function 
        "rng_seed": 42, 
        "ds": data_generator,

        # Model config (parameters for SimpleTransformer)
        "ignore_index": IGNORE_INDEX, 
        "model_config": {
             "num_layers": 2,
             "hidden_size": 128,
             "num_heads": 4,
             "ffn_hidden_size": 256,
             "dropout_prob": 0.1,
             # These will be populated dynamically:
             "vocab_size": None,
             "ignore_index": None,
             "sequence_length": None, # This should match the main config's sequence_length
        },

        # Training parameters
        "num_steps": 10, # Total training steps
        "total_steps": 10, # what is this then in birdie pipeline?
        "steps_between_evaluations": 2, 
        "log_interval": 2, 
        # Generation parameters
        "gen_length": 256,
        "generation_temperature": 0.8,
        "generation_top_k": 50,
        "generation_top_p": None, 
        "generation_output_dir": "generated_audio" 
    }

    config['model_config']['sequence_length'] = config['sequence_length']
    if config['model_config']['ffn_hidden_size'] is not None:
         try:
              config['model_config']['ffn_hidden_size'] = int(config['model_config']['ffn_hidden_size'])
         except (ValueError, TypeError):
              config['model_config']['ffn_hidden_size'] = None

#%%
    # --- Step 0: Initialize accelerate ---
    try:
        print(fn, "Initializing accelerate.")
        accelerator = accelerate.Accelerator()
        config['accelerator'] = accelerator 
        print(fn, "Accelerate initialized.")
        accelerate.utils.set_seed(config['rng_seed'])

    except Exception as e:
        print(fn, f"FATAL ERROR: Failed to initialize accelerate: {e}")
        traceback.print_exc()
        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1) 

#%%
    # --- Step 1: Prepare dataset and tokenizer ---
    tokenizer = None
    dataset_preparation_successful = False

    try:
        print(fn, "Preparing dataset and fitting tokenizer")
        data_dir = os.path.join(config['base_data_dir'], "genres_original", config['genre'])
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        dataset_info = prepare_dataset(data_dir, config.get('max_files'), config)
        tokenizer = dataset_info.get('tokenizer')
        if tokenizer is None:
            raise RuntimeError("prepare_dataset did not return a tokenizer.")

        config['tokenizer'] = tokenizer
        config['vocab_size'] = tokenizer.vocab_size
        if hasattr(tokenizer, 'ignore_index') and tokenizer.ignore_index is not None:
             config['ignore_index'] = tokenizer.ignore_index 
             print(fn, f"Using tokenizer's ignore_index: {config['ignore_index']}")
        else:
             print(fn, f"Tokenizer lacks valid ignore_index. default from import")

        # Attach the tokenizer to the data generator for easier access
        data_generator.tokenizer = tokenizer

        data_generator.config = config

        config['model_config']['vocab_size'] = config['vocab_size']
        config['model_config']['ignore_index'] = config['ignore_index']

        print(fn, f"Tokenizer fitted. Vocab size: {config['vocab_size']}, Ignore index: {config['ignore_index']}")
        print(fn, f"Model config after tokenizer fit: {config['model_config']}")
        dataset_preparation_successful = True 

    except Exception as e:
        print(fn, f"FATAL ERROR during dataset/tokenizer preparation: {e}")
        traceback.print_exc()
        dataset_preparation_successful = False

    # --- Check preparation status and exit if failed ---
    # In single-process, this check is sufficient. In multi-process,
    # you would need a barrier and possibly a shared flag or broadcast.
    if not dataset_preparation_successful:
        debug_print(fn, "dataset/tokenizer preparation failure.")
        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1) # Exit the script

#%%
    # --- Step 2: Test data loading before initializing Birdie ---
    try:
        test_samples = data_generator('validation', 0, 1, config['rng_seed'], config)
        print(fn, f"Test validation data loading successful. Got {len(test_samples)} samples.")
        
        test_samples = data_generator('train', 0, 1, config['rng_seed'], config)
        print(fn, f"Test train data loading successful. Got {len(test_samples)} samples.")
    except Exception as e:
        print(fn, f"ERROR: Data loading test failed: {e}")
        traceback.print_exc()
        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1)
    
#%%
    # --- Step 3: Initialize Birdie Orchestrator ---
    birdie_orchestrator = None
    try:
        print(fn, "Initializing Birdie Orchestrator")
        birdie_orchestrator = Birdie(config)
        print(fn, "Birdie Orchestrator initialized")
    except Exception as e:
        print(fn, f"FATAL ERROR Initializing Birdie: {e}")
        traceback.print_exc()

        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1)
    
#%%
    # --- Step 4: Initialize Model ---
    model = None
    try:
        print(fn, "Initializing model")
        model = SimpleTransformer(config=config['model_config'])
        print(fn, "Model initialized")
    except Exception as e:
        print(fn, f"FATAL ERROR Initializing model: {e}")
        traceback.print_exc()
        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1)

#%%
    # --- Step 5: Prepare with Accelerate ---
    optimizer = None
    try:
        print(fn, "Preparing model and optimizer with Accelerate")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer, birdie_orchestrator = accelerator.prepare(
            model, optimizer, birdie_orchestrator
        )
        print(fn, "Model, optimizer, and Birdie prepared")
        
        config['optimizer'] = optimizer
        config['prepared_model'] = model 
        
    except Exception as e:
        print(fn, f"FATAL ERROR during Accelerate prepare: {e}")
        traceback.print_exc()
        # Clean up Birdie
        if birdie_orchestrator and hasattr(birdie_orchestrator, 'close'):
            try:
                birdie_orchestrator.close()
                print("birdie closed")
            except Exception:
                print("birdie NOT closed!")
                pass
        if dual_output_instance: dual_output_instance.close_log()
        sys.exit(1)

#%%
    config["load_checkpoint"] = None  # Set to checkpoint path to load
    
    start_step = 0
    if config["load_checkpoint"] and os.path.exists(config["load_checkpoint"]):
        try:
            model, optimizer, start_step, loaded_config = load_model_checkpoint(
                config["load_checkpoint"], 
                model, 
                optimizer, 
                config,
                accelerator.device
            )
            print(fn, f"Resumed from checkpoint: step {start_step}")
            
            if loaded_config:
                # Only update non-critical parameters to avoid conflicts
                for key in ['num_steps', 'steps_between_evaluations', 'log_interval']:
                    if key in loaded_config:
                        config[key] = loaded_config[key]
        except Exception as e:
            print(fn, f"Error loading checkpoint: {e}")
            traceback.print_exc()
            print(fn, "Starting training from scratch instead.")

#%%
    # --- Step 6: Train Model ---
    try:
        debug_print(fn, "Starting model training.")
        train_model(accelerator, birdie_orchestrator, model, optimizer, config)
        debug_print(fn, "Model training finished.")
    except Exception as e:
        debug_print(fn, f"ERROR during train_model execution: {e}")
        traceback.print_exc()

#%%
    sys.exit(1) # Exit the script and then run the next cell manually.

#%%
    # --- Step 7: Generate Music ---
    if accelerator.is_main_process:
         print(fn, "Starting generation.")

         unwrapped_model = accelerator.unwrap_model(model)

         gen_length = config.get("gen_length", 256) 
         gen_output_dir = config.get("generation_output_dir", "generated_audio")

         os.makedirs(gen_output_dir, exist_ok=True)
         print(fn, f"Generation output directory: {gen_output_dir}")

         # Seed audio is optional
         seed_audio_data = None
         seed_audio_path = os.path.join(config['base_data_dir'], "genres_original", config['genre'], "blues.00000.wav") 
         if os.path.exists(seed_audio_path):
              print(fn, f"Loading seed audio from {seed_audio_path}")
              try:
                   seed_audio_data = load_audio_file(seed_audio_path)
                   if seed_audio_data is None or not isinstance(seed_audio_data, np.ndarray) or seed_audio_data.size == 0:
                        print(fn, "invalid seed audio data. Generating from scratch.")
                        seed_audio_data = None 
              except Exception as e:
                   print(fn, f"Error loading seed audio: {e}. Generating from scratch.")
                   traceback.print_exc()
                   seed_audio_data = None # 
         else:
              print(fn, "Seed audio file not found. Generating from scratch.")

         try:
             generated_audio_path = generate_music(
                  accelerator=accelerator, 
                  model=unwrapped_model, 
                  config=config, 
                  seed_audio=seed_audio_data,
                  length=gen_length 
             )

             if generated_audio_path:
                  print(fn, "Music generation successful.")

         except Exception as e:
              print(fn, f"ERROR during music generation: {e}")
              traceback.print_exc()

    else: 
         print(fn, f"Worker {accelerator.local_process_index}: Skipping music generation (not main process).")

#%%
    # --- Step 8: Cleanup Birdie Orchestrator ---
    # Ensure Birdie is closed gracefully to shut down worker processes.
    debug_print(fn, f"Worker {accelerator.local_process_index}: Cleaning up Birdie.")
    try:
        if birdie_orchestrator and hasattr(birdie_orchestrator, 'close'):
             birdie_orchestrator.close() 

    except Exception as e:
         debug_print(fn, f"Worker {accelerator.local_process_index}: ERROR during Birdie cleanup: {e}")
         traceback.print_exc()

    accelerator.wait_for_everyone()

    debug_print(fn, f"Worker {accelerator.local_process_index}: FINISH.")

    # --- Close Log File and Restore Stdout ---
    if accelerator.is_main_process:
        if dual_output_instance:
            dual_output_instance.close_log()
            if sys.stdout == dual_output_instance:
                 sys.stdout = original_stdout
        else:
            print(f"Script finished but NO log file!.")
