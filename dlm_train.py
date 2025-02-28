#!/usr/bin/env python3
"""
A toy implementation of a diffusion language model using a pretrained Qwen model.
Both the pretrained Qwen model and the resulting diffusion model (after training)
will be saved in the "./projects" folder.

This example:
  • Loads the Qwen model and its tokenizer from a local cache folder.
  • Wraps the model in a simple diffusion objective.
  • Trains the diffusion model to denoise token embeddings.
  • Saves both the base Qwen model files and the resulting diffusion model files in "./projects".
  
IMPORTANT: This is a simplified example intended for prototyping and research.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define project directory where models will be stored.
PROJECTS_DIR = "./projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

#────────────────────────────
# 1. Diffusion schedule functions
#────────────────────────────
def get_noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    """
    Returns:
      betas:      (T,) tensor of noise schedule values.
      sqrt_alpha_bar: (T,) tensor: sqrt(cumprod(1-beta)) up to timestep t.
      sqrt_one_minus_alpha_bar: (T,) tensor.
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    return betas, sqrt_alpha_bar, sqrt_one_minus_alpha_bar

#────────────────────────────
# 2. A simple custom dataset (for illustration only)
#────────────────────────────
class DummyTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

#────────────────────────────
# 3. DiffusionLM model wrapper
#────────────────────────────
class DiffusionLM(nn.Module):
    def __init__(self, llama_model, num_timesteps, hidden_size=None):
        """
        llama_model: a pretrained causal LM (Qwen model in our case)
        num_timesteps: number of diffusion steps (T)
        hidden_size: dimensionality of token embeddings. If None, use model's hidden size.
        """
        super().__init__()
        self.llama = llama_model
        self.num_timesteps = num_timesteps
        
        # Get the hidden size from the model if not provided
        if hidden_size is None:
            if hasattr(llama_model.config, 'hidden_size'):
                hidden_size = llama_model.config.hidden_size
            else:
                # For Qwen, try different attribute names
                hidden_size = getattr(llama_model.config, 'hidden_size', 
                                     getattr(llama_model.config, 'n_embd', 2048))
        
        self.hidden_size = hidden_size
        print(f"Using hidden size: {hidden_size}")
        
        # Learnable embedding for each diffusion timestep.
        self.timestep_embedding = nn.Embedding(num_timesteps, hidden_size)
        # Additional head to predict noise from the transformer hidden states.
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Let's add a projection layer if we're using logits as hidden states
        # (for models where we can't access the last hidden state directly)
        self.use_logits_projection = not hasattr(llama_model, 'get_input_embeddings')
        if self.use_logits_projection:
            vocab_size = llama_model.config.vocab_size
            self.logits_projection = nn.Linear(vocab_size, hidden_size)
            print(f"Added logits projection layer: vocab_size={vocab_size} -> hidden_size={hidden_size}")

    def forward(self, input_ids, timestep, noisy_embeddings, attention_mask=None):
        """
        input_ids: (batch, seq_len) – used for shape information and position embeddings.
        timestep: (batch,) LongTensor with discrete diffusion step indices in [0, T-1]
        noisy_embeddings: (batch, seq_len, hidden_size) – diffusion-noised token embeddings.
        attention_mask: optional (batch, seq_len)
        
        Returns:
          predicted_noise: (batch, seq_len, hidden_size)
        """
        t_emb = self.timestep_embedding(timestep)  # shape: (batch, hidden_size)
        t_emb = t_emb.unsqueeze(1).expand_as(noisy_embeddings)  # (batch, seq_len, hidden_size)
        conditioned_inputs = noisy_embeddings + t_emb

        # Use the language model's inputs_embeds interface.
        outputs = self.llama(
            inputs_embeds=conditioned_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Try to get the hidden states in a model-agnostic way
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Use the last layer's hidden states
            hidden_states = outputs.hidden_states[-1]
        else:
            # For Qwen models, use logits but project them to the right dimension
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            if self.use_logits_projection:
                hidden_states = self.logits_projection(logits)
            else:
                # If we can't project properly, try to extract embeddings directly
                hidden_states = self.llama.get_input_embeddings()(input_ids)
            
        predicted_noise = self.output_proj(hidden_states)
        return predicted_noise

#────────────────────────────
# 4. Main training loop
#────────────────────────────
def main():
    # Hyperparameters.
    num_diffusion_timesteps = 1000  # Total diffusion steps (T)
    beta_start = 0.0001
    beta_end = 0.02
    lr = 1e-4
    num_epochs = 1  # Reduced from 3 to 1
    batch_size = 2  # Reduced from 4 to 2
    gradient_accumulation_steps = 4  # Accumulate gradients over multiple batches
    max_grad_norm = 1.0  # For gradient clipping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the noise schedule.
    betas, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_noise_schedule(
        num_diffusion_timesteps, beta_start, beta_end
    )

    # Define paths to save/load models
    model_path = "qwen/Qwen2.5-3B"  # or your designated identifier/path.
    cache_model_path = os.path.join(PROJECTS_DIR, "models--qwen--Qwen2.5-3B")
    
    print(f"Loading model and tokenizer from {model_path}")
    
    # Load the tokenizer and pretrained Qwen model.
    # First try to load from cache, then from HuggingFace
    try:
        if os.path.exists(cache_model_path):
            print(f"Found cached model at {cache_model_path}, loading from there")
            tokenizer = AutoTokenizer.from_pretrained(cache_model_path)
            llama_model = AutoModelForCausalLM.from_pretrained(
                cache_model_path,
                device_map="auto",
                torch_dtype=torch.float16,  # Use FP16 precision
                use_cache=False,  # Disable KV cache for training
            )
        else:
            print(f"Downloading model from HuggingFace: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=PROJECTS_DIR)
            llama_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=PROJECTS_DIR,
                device_map="auto",
                torch_dtype=torch.float16,  # Use FP16 precision
                use_cache=False,  # Disable KV cache for training
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to direct download from HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=PROJECTS_DIR)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=PROJECTS_DIR,
            device_map="auto",
            torch_dtype=torch.float16,  # Use FP16 precision
            use_cache=False,  # Disable KV cache for training
        )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(llama_model, 'gradient_checkpointing_enable'):
        print("Enabling gradient checkpointing for memory efficiency")
        llama_model.gradient_checkpointing_enable()
    
    llama_model.to(device)
    llama_model.eval()  # Use the base model in evaluation mode.

    # Print model configuration
    print(f"Model config: {llama_model.config}")
    
    # Create diffusion model - let it determine the hidden size from the model
    diff_model = DiffusionLM(llama_model, num_diffusion_timesteps)
    diff_model.to(device)

    # Use AdamW with memory optimizations
    optimizer = optim.AdamW(
        diff_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=True if torch.__version__ >= "2.0.0" and torch.cuda.is_available() else False  # Use fused Adam if available
    )

    # For demonstration, we create a dummy dataset.
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Diffusion models are powerful generative models.",
        "Language modeling is an exciting research area.",
        "PyTorch makes it easy to prototype deep learning models."
    ]
    dataset = DummyTextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mse_loss_fn = nn.MSELoss()

    print("Starting training...")
    diff_model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch_texts in enumerate(dataloader):
            print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
            
            # Tokenize texts.
            encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            B, L = input_ids.shape
            
            print(f"  - Input shape: batch={B}, sequence_length={L}")

            # Get the "clean" token embeddings from the Qwen embedding layer.
            with torch.no_grad():
                clean_embeddings = llama_model.model.embed_tokens(input_ids)

            # Sample a diffusion timestep for each batch item.
            t = torch.randint(0, num_diffusion_timesteps, (B,), device=device).long()
            sqrt_alpha = sqrt_alpha_bar[t].to(device).view(B, 1, 1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha_bar[t].to(device).view(B, 1, 1)

            noise = torch.randn_like(clean_embeddings)
            # Forward diffusion: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise.
            noisy_embeddings = sqrt_alpha * clean_embeddings + sqrt_one_minus_alpha * noise

            # Predict the noise.
            predicted_noise = diff_model(
                input_ids=input_ids,
                timestep=t,
                noisy_embeddings=noisy_embeddings,
                attention_mask=attention_mask
            )

            loss = mse_loss_fn(predicted_noise, noise)
            # Scale the loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(diff_model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale loss back up for logging
            batch_count += 1
            
            print(f"  - Batch {batch_idx+1} Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        avg_epoch_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch {epoch+1} Complete | Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training complete.")
    
    # Save the resulting diffusion model.
    # Note: we need to save our custom diffusion parameters (timestep embeddings and the noise head)
    # while the underlying Qwen model files have been cached already in ./projects.
    diffusion_save_path = os.path.join(PROJECTS_DIR, "diffusion_model")
    os.makedirs(diffusion_save_path, exist_ok=True)
    
    # Save the tokenizer
    tokenizer.save_pretrained(diffusion_save_path)
    
    # Save the diffusion model's state dict
    model_save_path = os.path.join(diffusion_save_path, "diffusion_model.pt")
    print(f"Saving diffusion model to {model_save_path}")
    
    # Save only the state dict of the custom components, not the entire model
    state_dict = {
        'timestep_embedding.weight': diff_model.timestep_embedding.weight,
        'output_proj.weight': diff_model.output_proj.weight,
        'output_proj.bias': diff_model.output_proj.bias
    }
    
    # Add logits projection if it exists
    if hasattr(diff_model, 'use_logits_projection') and diff_model.use_logits_projection:
        state_dict['logits_projection.weight'] = diff_model.logits_projection.weight
        state_dict['logits_projection.bias'] = diff_model.logits_projection.bias
    
    torch.save(state_dict, model_save_path)
    
    # Also save the configuration
    config_info = {
        "num_diffusion_timesteps": num_diffusion_timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "hidden_size": diff_model.hidden_size,
        "base_model_path": model_path,
        "use_logits_projection": hasattr(diff_model, 'use_logits_projection') and diff_model.use_logits_projection
    }
    config_path = os.path.join(diffusion_save_path, "config.pt")
    torch.save(config_info, config_path)
    print(f"Saved configuration to {config_path}")
    
    print(f"Diffusion model saved to {diffusion_save_path}")

if __name__ == "__main__":
    main()