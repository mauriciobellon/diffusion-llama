#!/usr/bin/env python3
"""
A modified version of dlm_train.py that loads a custom dataset from a text file.
Usage: python train_with_custom_dataset.py --dataset path/to/your/dataset.txt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

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
# 2. Custom Dataset class
#────────────────────────────
class TextDataset(Dataset):
    def __init__(self, file_path, max_examples=None):
        """
        Load a dataset from a text file, one sentence per line.
        
        Args:
            file_path: Path to the text file
            max_examples: Maximum number of examples to load (None for all)
        """
        self.texts = []
        
        print(f"Loading dataset from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:  # Skip empty lines
                        self.texts.append(line)
                    if max_examples is not None and i >= max_examples - 1:
                        break
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to a tiny dataset
            self.texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Diffusion models are powerful generative models."
            ]
            
        print(f"Loaded {len(self.texts)} examples")
        
        # Print a few examples
        if len(self.texts) > 0:
            print("Sample texts:")
            for i in range(min(3, len(self.texts))):
                print(f"  - {self.texts[i][:50]}{'...' if len(self.texts[i]) > 50 else ''}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

#────────────────────────────
# 3. DiffusionLM model wrapper (from dlm_train)
#────────────────────────────
from dlm_train import DiffusionLM

#────────────────────────────
# 4. Main training loop
#────────────────────────────
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a diffusion language model on a custom dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file (one sentence per line)")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to load from dataset")
    # Add memory optimization parameters
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training (FP16)")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use mixed precision training (BF16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Use gradient checkpointing to save memory")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
    args = parser.parse_args()
    
    # Hyperparameters from command line
    num_diffusion_timesteps = args.num_steps
    lr = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    beta_start = 0.0001
    beta_end = 0.02
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the noise schedule.
    betas, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_noise_schedule(
        num_diffusion_timesteps, beta_start, beta_end
    )

    # Define paths to save/load models
    model_path = "qwen/Qwen2.5-3B"  # or your designated identifier/path.
    cache_model_path = os.path.join(PROJECTS_DIR, "models--qwen--Qwen2.5-3B")
    
    print(f"Loading model and tokenizer from {model_path}")
    
    # Load the tokenizer and pretrained Qwen model.
    try:
        if os.path.exists(cache_model_path):
            print(f"Found cached model at {cache_model_path}, loading from there")
            tokenizer = AutoTokenizer.from_pretrained(cache_model_path)
            llama_model = AutoModelForCausalLM.from_pretrained(
                cache_model_path,
                device_map="auto" if args.device == "cuda" else None,
                torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
                use_cache=False,  # Disable KV cache for training
            )
        else:
            print(f"Downloading model from HuggingFace: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=PROJECTS_DIR)
            llama_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=PROJECTS_DIR,
                device_map="auto" if args.device == "cuda" else None,
                torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
                use_cache=False,  # Disable KV cache for training
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to direct download from HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=PROJECTS_DIR)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=PROJECTS_DIR,
            device_map="auto" if args.device == "cuda" else None,
            torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
            use_cache=False,  # Disable KV cache for training
        )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing and hasattr(llama_model, 'gradient_checkpointing_enable'):
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

    # Setup mixed precision training if requested
    scaler = None
    if args.fp16 or args.bf16:
        amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
        print(f"Using mixed precision training with {str(amp_dtype)} dtype")
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # scaler only needed for fp16, not bf16

    # Load custom dataset or use dummy data if not provided
    if args.dataset:
        dataset = TextDataset(args.dataset, max_examples=args.max_examples)
    else:
        print("No dataset provided, using dummy data")
        # For demonstration, we create a dummy dataset.
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Diffusion models are powerful generative models.",
            "Language modeling is an exciting research area.",
            "PyTorch makes it easy to prototype deep learning models."
        ]
        dataset = TextDataset("", max_examples=None)
        dataset.texts = texts
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mse_loss_fn = nn.MSELoss()

    print(f"Starting training with {len(dataset)} examples, {batch_size} batch size, {num_epochs} epochs")
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

            # Use mixed precision for forward and backward passes if enabled
            if args.fp16 or args.bf16:
                with torch.cuda.amp.autocast(dtype=torch.float16 if args.fp16 else torch.bfloat16):
                    # Predict the noise.
                    predicted_noise = diff_model(
                        input_ids=input_ids,
                        timestep=t,
                        noisy_embeddings=noisy_embeddings,
                        attention_mask=attention_mask
                    )
                    loss = mse_loss_fn(predicted_noise, noise)
                    # Scale the loss by gradient accumulation steps
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    # Use scaler for fp16
                    scaler.scale(loss).backward()
                else:
                    # bf16 doesn't need scaling
                    loss.backward()
            else:
                # Standard full precision
                predicted_noise = diff_model(
                    input_ids=input_ids,
                    timestep=t,
                    noisy_embeddings=noisy_embeddings,
                    attention_mask=attention_mask
                )
                loss = mse_loss_fn(predicted_noise, noise)
                # Scale the loss by gradient accumulation steps
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Apply gradient clipping to prevent exploding gradients
                if args.fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(diff_model.parameters(), args.max_grad_norm)
                
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps  # Scale loss back up for logging
            batch_count += 1
            
            print(f"  - Batch {batch_idx+1} Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")
        
        avg_epoch_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch {epoch+1} Complete | Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training complete.")
    
    # Save the resulting diffusion model.
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