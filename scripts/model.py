# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
import math
import numpy as np
import sys
import os
import random
import traceback

from utils import debug_print
from tokenizer import SpectrogramTokenizer
from dataset_preparation import prepare_dataset
from audio_ul2_config import (
    N_CLUSTERS, DURATION, SAMPLE_RATE, IGNORE_INDEX,
    audio_ul2_config, validation_objectives_config
)
import soundfile as sf 
#%%
class SimpleTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        fn = "SimpleTransformer.__init__"

        self.vocab_size = config.get("vocab_size")
        self.sequence_length = config.get("sequence_length") 
        self.hidden_size = config.get("hidden_size", 512)
        self.num_layers = config.get("num_layers", 6)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_prob = config.get("dropout_prob", 0.1)
        self.ignore_index = config.get("ignore_index")

        ffn_hidden_size_from_config = config.get('ffn_hidden_size', None)
        if ffn_hidden_size_from_config is None:
            self.ffn_hidden_size = self.hidden_size * 4
        else:
            self.ffn_hidden_size = ffn_hidden_size_from_config



        # Embedding 
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)

        # Positional embedding layer 
        self.position_embeddings = nn.Embedding(self.sequence_length, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_prob)

        # Transformer decoder 
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.ffn_hidden_size,
                dropout=self.dropout_prob,
                activation=F.gelu, 
                batch_first=True, # Expect input shape (batch, seq, feature)
                norm_first=True # Apply layer norm before attention/FFN
            ) for _ in range(self.num_layers)
        ])

        # Final layer normalization (applied after decoder stack)
        self.norm_final = nn.LayerNorm(self.hidden_size)

        # Final linear Layer to project to vocabulary size (logits)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)



        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module == self.output_layer:
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
        label_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None, 
        **kwargs # Catch other batch keys from Birdie
    ) -> torch.Tensor:
        fn = "SimpleTransformer.forward"
        B, L = input_ids.shape # Batch size, Sequence length

        # Validate input sequence length and truncate if not
        if L > self.sequence_length:
             input_ids = input_ids[:, :self.sequence_length]
             if label_ids is not None and label_ids.shape[1] > self.sequence_length:
                 label_ids = label_ids[:, :self.sequence_length]
             if attention_mask is not None and attention_mask.shape[1] > self.sequence_length:
                 attention_mask = attention_mask[:, :self.sequence_length]
             L = self.sequence_length # Update L after potential truncation

        input_ids_for_embedding = input_ids.clone()
        input_ids_for_embedding[input_ids == self.ignore_index] = 0 

        
        try:
            token_embeds = self.token_embeddings(input_ids_for_embedding) # (B, L, hidden_size)
        except IndexError as e:
            raise IndexError(f"{fn}: Error during token embedding lookup: {e}") from e

        # Get positional embeddings
        position_ids = torch.arange(L, dtype=torch.long, device=input_ids.device)
        position_embeddings = self.position_embeddings(position_ids) # (L, hidden_size)

        # Combine embeddings (B, L, hidden_size)
        embeddings = token_embeds + position_embeddings.unsqueeze(0) # Add batch dim

        embeddings = self.dropout(embeddings)

        # --- Prepare Masks for TransformerDecoderLayer ---
        # 1. Causal Mask (`tgt_mask`): Prevents attention to future positions.
        #    Shape (L, L). True indicates a masked position.
        tgt_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)

        # 2. Padding Mask (`tgt_key_padding_mask`): Prevents attention to padding tokens.
        #    Shape (B, L). True indicates a padded position.
        #    Derived from the original input_ids where value is ignore_index.
        tgt_key_padding_mask = (input_ids == self.ignore_index)

        # Note on Birdie's `attention_mask`:Birdie provides an `attention_mask` (1 for real tokens, 0 for padding).
        decoder_output = embeddings # Start with the combined embeddings
        for i, layer in enumerate(self.decoder_layers):
            try:
                decoder_output = layer(
                    tgt=decoder_output,
                    memory=decoder_output, # Use the same sequence as memory for self-attention
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
            except Exception as e:
                raise RuntimeError(f"{fn}: Error in TransformerDecoderLayer {i}: {e}") from e


        decoder_output = self.norm_final(decoder_output)

        # Final layer to get logits
        logits = self.output_layer(decoder_output) # Shape (B, L, vocab_size)

        return logits


    @torch.no_grad()
    def generate(self, initial_token_ids: torch.Tensor, max_length: int, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, stop_token_id: Optional[int] = None) -> torch.Tensor:
        fn = "SimpleTransformer.generate"
        generated_sequence = initial_token_ids.long() 

        # Generation loop
        for step in range(max_length - initial_token_ids.size(1)): # Generate up to max_length - initial_len new tokens
            current_seq_len = generated_sequence.size(1)

            if current_seq_len > self.sequence_length:
                 model_input = generated_sequence[:, -self.sequence_length:]
            else:
                 model_input = generated_sequence

            try:
                logits = self.forward(input_ids=model_input) # Shape (1, current_input_len, vocab_size)
            except Exception as e:
                 debug_print(fn, f"Error during model forward pass in generation step {step} (input len {model_input.size(1)}): {e}")
                 traceback.print_exc()
                 break 

            if logits is None or logits.ndim != 3:
                 debug_print(fn, "Error: Model forward pass did not return valid logits. Exiting generation.")
                 break

            next_token_logits = logits[:, -1, :] # Shape (1, vocab_size)

            if temperature != 1.0:
                 next_token_logits = next_token_logits / temperature

            if top_k is not None:
                 v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                 next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                 sorted_indices_to_remove = cumulative_probs > top_p
                 # Shift to keep the first token exceeding p
                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                 sorted_indices_to_remove[..., 0] = False
                 # Scatter mask back to original indices
                 indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                 next_token_logits[indices_to_remove] = -float('Inf')

            # Check for invalid logits after filtering
            if torch.all(torch.isinf(next_token_logits)) or torch.all(torch.isnan(next_token_logits)):
                debug_print(fn, "Warning: All logits became -Inf or NaN after filtering. Cannot sample. Exiting generation.")
                break

            # Calculate probabilities using softmax
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Check for NaN probabilities
            if torch.any(torch.isnan(next_token_probs)):
                 debug_print(fn, "Warning: NaN probabilities encountered during sampling. Exiting generation.")
                 break

            # Sample using multinomial
            try:
                 next_token_id = torch.multinomial(next_token_probs, num_samples=1) # Shape (1, 1)
            except RuntimeError as e:
                 debug_print(fn, f"RuntimeError during multinomial sampling: {e}. Prob sum: {next_token_probs.sum().item()}. Exiting.")
                 traceback.print_exc()
                 break 

            # Append the generated token
            generated_sequence = torch.cat([generated_sequence, next_token_id], dim=1)

            # Check for stop token
            if stop_token_id is not None and next_token_id.item() == stop_token_id:
                 debug_print(fn, f"Generated stop token {stop_token_id}. Stopping generation.")
                 break

        return generated_sequence # Return the full sequence (initial + generated)


@torch.no_grad()
def generate_audio_from_model(model: SimpleTransformer, tokenizer: SpectrogramTokenizer, initial_text: str, max_audio_tokens: int, device: torch.device, **generation_kwargs) -> np.ndarray:
    fn = "generate_audio_from_model"
    debug_print(fn, f"Starting audio generation from text prompt: '{initial_text}'")

    if model.training:
         debug_print(fn, "Setting model to evaluation mode for generation.")
         model.eval()

    # 1. Encode the initial text prompt into token IDs
    try:
         initial_token_ids_np = tokenizer.encode(initial_text)

         initial_token_ids = torch.from_numpy(initial_token_ids_np).unsqueeze(0).to(device).long()
         debug_print(fn, f"Encoded prompt into {initial_token_ids.size(1)} tokens.")

    except Exception as e:
         debug_print(fn, f"Error encoding initial text prompt '{initial_text}': {e}")
         traceback.print_exc()
         return np.array([], dtype=np.float32) 

    # 2. Generate the full sequence of token IDs using the model
    initial_len = initial_token_ids.size(1)
    max_total_length = initial_len + max_audio_tokens

    # Get the actual stop token ID from the tokenizer
    stop_token_id_for_model = tokenizer.get_special_token_id("<|END|>")
    try:
         generated_token_ids_tensor = model.generate(
             initial_token_ids=initial_token_ids,
             max_length=max_total_length,
             stop_token_id=stop_token_id_for_model, 
             temperature=generation_kwargs.get('temperature', 1.0),
             top_k=generation_kwargs.get('top_k'),
             top_p=generation_kwargs.get('top_p')
         )

         if generated_token_ids_tensor is None or generated_token_ids_tensor.size(0) != 1:
              debug_print(fn,"Error: Model generation returned invalid result.")
              return np.array([], dtype=np.float32)

         # Extract only the newly generated tokens (after the initial prompt)
         generated_audio_token_ids_tensor = generated_token_ids_tensor[:, initial_len:]
         num_generated = generated_audio_token_ids_tensor.size(1)
         debug_print(fn, f"Model generated {num_generated} new tokens.")

         if num_generated == 0:
              debug_print(fn, "Warning: Model generated 0 new tokens.")
              return np.array([], dtype=np.float32)

         # Convert generated token IDs back to NumPy array for tokenizer
         generated_audio_token_ids_np = generated_audio_token_ids_tensor.squeeze(0).cpu().numpy()

    except Exception as e:
         debug_print(fn, f"Error during model generation step: {e}")
         traceback.print_exc()
         return np.array([], dtype=np.float32)

    # 3. Decode the generated audio token IDs back to raw audio
    try:
         # tokenizer.decode filters special tokens and returns audio waveform
         reconstructed_audio = tokenizer.decode(generated_audio_token_ids_np)

         if reconstructed_audio is None or not isinstance(reconstructed_audio, np.ndarray):
              return np.array([], dtype=np.float32)
         if reconstructed_audio.size == 0:
              debug_print(fn, f"Generated token IDs were: {generated_audio_token_ids_np.tolist()}")

         return reconstructed_audio.astype(np.float32) 

    except Exception as e:
         debug_print(fn, f"Error decoding generated tokens to audio: {e}")
         traceback.print_exc()
         return np.array([], dtype=np.float32)


#%%
if __name__ == "__main__":
    fn = f"*{os.path.basename(__file__)} (test)*"
    print(f"\nRunning Script {fn} Test Block\n")

    test_config = {
        "N_CLUSTERS": N_CLUSTERS, 
        "ignore_index": IGNORE_INDEX,
        "vocab_size": None, 
        "sequence_length": 128, #
        "hidden_size": 64,   
        "num_layers": 2,   
        "num_heads": 4,  
        "ffn_hidden_size": 128,
        "dropout_prob": 0.1,

        "genre": "test_genre", 
        "base_data_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_test'), 
        "max_files": 5, 

        "objectives": audio_ul2_config,
        "validation_objectives": validation_objectives_config,
        "start_generating_paradigm": "\n<|assistant|>\n",

        "test_max_new_tokens": 50,
        "test_temperature": 0.8,
        "test_top_k": 20,
    }


    test_data_dir = os.path.join(test_config['base_data_dir'], "genres_original", test_config['genre'])
    fitted_tokenizer = None

    try:
        # Create test directory if needed
        os.makedirs(test_data_dir, exist_ok=True)

        num_dummy_files = test_config['max_files']
        for i in range(num_dummy_files):
            dummy_audio_path = os.path.join(test_data_dir, f"dummy_{i:03d}.wav")
            if not os.path.exists(dummy_audio_path):
                 duration = (1.0 + random.random()) # Vary duration slightly
                 t = np.linspace(0., duration, int(duration * SAMPLE_RATE), endpoint=False)
                 freq = 440 + random.randint(-50, 50)
                 amplitude = np.iinfo(np.int16).max * 0.1
                 data = amplitude * np.sin(2. * np.pi * freq * t)
                 sf.write(dummy_audio_path, data.astype(np.int16), SAMPLE_RATE)
        debug_print(fn, "Dummy audio files created/checked.")

        dataset_info = prepare_dataset(
             data_dir=test_data_dir,
             max_files=test_config['max_files'],
             config=test_config 
        )
        fitted_tokenizer = dataset_info.get('tokenizer')
        if fitted_tokenizer is None:
            raise RuntimeError("prepare_dataset did not return a tokenizer.")


        test_config["vocab_size"] = fitted_tokenizer.vocab_size
        if fitted_tokenizer.ignore_index != test_config["ignore_index"]:
            test_config["ignore_index"] = fitted_tokenizer.ignore_index

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)




    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = SimpleTransformer(test_config)
        model.to(device)
        model.eval() # Set to eval mode for testing forward/generate
        debug_print(fn, "SimpleTransformer model initialized successfully.")

        batch_size = 2
        seq_len = min(32, test_config['sequence_length']) # Use a short sequence for forward test
        dummy_input_ids = torch.randint(0, test_config['vocab_size'] - 5, (batch_size, seq_len), dtype=torch.long, device=device)
        # Randomly sprinkle ignore_index (padding)
        pad_mask = torch.rand(batch_size, seq_len, device=device) < 0.1 # Approx 10% padding
        dummy_input_ids[pad_mask] = test_config['ignore_index']

        # Dummy labels (needed if loss were calculated here, but good practice to have for external check)
        dummy_label_ids = torch.randint(0, test_config['vocab_size'] - 5, (batch_size, seq_len), dtype=torch.long, device=device)
        dummy_label_ids[pad_mask] = test_config['ignore_index'] # Match padding

        with torch.no_grad():
            logits = model(input_ids=dummy_input_ids) # Only pass input_ids

        # Check output shape
        expected_shape = (batch_size, seq_len, test_config['vocab_size'])
        if logits.shape == expected_shape:
            debug_print(fn, "Forward pass successful.")
        else:
            raise RuntimeError("Forward pass failed. ")

    except Exception as e:
        traceback.print_exc()
        print("\nCannot proceed with generation test.",e)
        model = None #

    print(f"\n--- {fn} Test Block Finished ---")
