#!/usr/bin/env python3
"""
An example inference procedure for the diffusion language model.
After training, this routine generates text by starting from a fully noised set of
token embeddings and then iteratively reducing the noise.
NOTE: This is a simplified version and may require adjustments and tuning.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import the DiffusionLM class from the training module
from dlm_train import DiffusionLM
import os

model_path = "qwen/Qwen2.5-3B"
cache_directory = "./projects"  # this directory will be used to save model files

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_directory)
llama_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_directory)

def get_noise_schedule(T, beta_start=0.0001, beta_end=0.02, device="cpu"):
    """
    Returns:
      betas:      (T,) tensor of noise schedule values.
      alphas:     (T,) tensor of 1 - betas.
      alpha_bar:  (T,) cumulative product of alphas.
      sqrt_alpha_bar: (T,) tensor: sqrt(cumprod(1-beta)).
      sqrt_one_minus_alpha_bar: (T,) tensor.
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    return betas, alphas, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar

def generate_text(diff_model, tokenizer, num_diffusion_timesteps, seq_len=32, device="cuda", show_steps=True, step_interval=100):
    """
    Iteratively denoise a noisy embedding to generate text.
    
    Arguments:
      diff_model: your diffusion language model (instance of DiffusionLM)
      tokenizer: the corresponding tokenizer (used here for shape info and final decoding)
      num_diffusion_timesteps: total number of diffusion steps (T)
      seq_len: length of the output sequence (number of tokens)
      device: device to run on.
      show_steps: whether to show intermediate results during denoising
      step_interval: how often to display intermediate results (every N steps)
      
    Returns:
      generated_text: string containing the decoded text.
    """
    # Get noise schedule parameters.
    _, _, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_noise_schedule(num_diffusion_timesteps, device=device)

    # 1. Initialize: start with a fully noised embedding
    hidden_size = diff_model.hidden_size
    print(f"Using hidden size: {hidden_size}")
    
    # Start with standard Gaussian noise
    x_t = torch.randn((1, seq_len, hidden_size), device=device)
    
    # Prepare for tracking intermediate results
    intermediate_texts = []
    if show_steps:
        print(f"\nStarting diffusion sampling with {num_diffusion_timesteps} steps")
        print(f"Will display intermediate results every {step_interval} steps")

    # 2. Iteratively apply the reverse diffusion process
    diff_model.eval()
    with torch.no_grad():
        for t in reversed(range(num_diffusion_timesteps)):
            # Report progress
            if show_steps and (t % step_interval == 0 or t == num_diffusion_timesteps - 1 or t == 0):
                print(f"\nStep {num_diffusion_timesteps-t}/{num_diffusion_timesteps} (t={t})")
            
            # Create a batch tensor for the current timestep
            t_tensor = torch.tensor([t], device=device).long()
            dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

            # Predict the noise at this step
            predicted_noise = diff_model(
                input_ids=dummy_input_ids,
                timestep=t_tensor,
                noisy_embeddings=x_t,
                attention_mask=torch.ones((1, seq_len), device=device)
            )
            
            # Calculate the scaling factors for this timestep
            sqrt_alpha_t = sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_bar[t]
            
            # Estimate the clean embedding
            x_0_est = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

            # For steps > 0, add some noise
            if t > 0:
                noise = torch.randn_like(x_t)
                beta_t = (1.0 - sqrt_alpha_t**2)
                x_t = x_0_est + torch.sqrt(beta_t) * noise
            else:
                x_t = x_0_est
            
            # Show intermediate results if requested
            if show_steps and (t % step_interval == 0 or t == num_diffusion_timesteps - 1 or t == 0):
                # Convert the current embedding to text
                current_outputs = diff_model.llama(
                    inputs_embeds=x_t,
                    attention_mask=torch.ones((1, seq_len), device=device),
                    return_dict=True
                )
                logits = current_outputs.logits
                current_ids = torch.argmax(logits, dim=-1)
                current_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
                
                # Store and print the intermediate result
                intermediate_texts.append((t, current_text))
                print(f"  Text: {current_text}")

    # 3. Convert final embeddings back to tokens.
    outputs = diff_model.llama(
        inputs_embeds=x_t,
        attention_mask=torch.ones((1, seq_len), device=device),
        return_dict=True
    )
    
    # Get the logits and convert to token IDs
    logits = outputs.logits
    generated_ids = torch.argmax(logits, dim=-1)
    
    # Decode token IDs to text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Show summary of the process
    if show_steps and len(intermediate_texts) > 1:
        print("\n=== Diffusion Process Summary ===")
        for i, (step, text) in enumerate(intermediate_texts):
            # Limit text length for display
            if len(text) > 40:
                text = text[:37] + "..."
            print(f"Step {i+1}/{len(intermediate_texts)} (t={step}): {text}")
        print("================================")
    
    return generated_text

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="qwen/Qwen2.5-3B", help="Path or identifier for the Qwen model")
    parser.add_argument("--diffusion_model_path", type=str, default="./projects/diffusion_model", help="Path to the saved diffusion model")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for generated text")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--show_steps", action="store_true", help="Show intermediate steps of the diffusion process")
    parser.add_argument("--step_interval", type=int, default=100, help="Interval between displayed steps")
    parser.add_argument("--num_steps", type=int, default=None, help="Override the number of diffusion steps (default: use value from config)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Check if we have a saved diffusion model configuration
    config_path = os.path.join(args.diffusion_model_path, "config.pt")
    if os.path.exists(config_path):
        config_info = torch.load(config_path, map_location=device)
        num_diffusion_timesteps = config_info.get("num_diffusion_timesteps", 1000)
        model_path = config_info.get("base_model_path", args.model_path)
    else:
        num_diffusion_timesteps = 1000
        model_path = args.model_path
        print("No config found, using default settings")
    
    # Override with command line argument if provided
    if args.num_steps is not None:
        num_diffusion_timesteps = args.num_steps
        print(f"Using custom number of diffusion steps: {num_diffusion_timesteps}")

    # Check if we have a cached model
    cache_dir = "./projects"
    cache_model_path = os.path.join(cache_dir, "models--qwen--Qwen2.5-3B")
    
    print(f"Loading model from {model_path}")
    
    # Load the pretrained Qwen model
    try:
        if os.path.exists(cache_model_path) and model_path == "qwen/Qwen2.5-3B":
            print(f"Found cached model at {cache_model_path}, loading from there")
            tokenizer = AutoTokenizer.from_pretrained(cache_model_path)
            llama_model = AutoModelForCausalLM.from_pretrained(cache_model_path)
        else:
            print(f"Loading model from HuggingFace or specified path: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            llama_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to direct download from HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        llama_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        
    llama_model.to(device)
    llama_model.eval()
    
    # Print model configuration
    print(f"Model config: {llama_model.config}")

    # Load the diffusion model (using the class imported from dlm_train)
    diff_model = DiffusionLM(llama_model, num_diffusion_timesteps)
    
    # Load the saved diffusion model weights if they exist
    diffusion_model_weights_path = os.path.join(args.diffusion_model_path, "diffusion_model.pt")
    if os.path.exists(diffusion_model_weights_path):
        try:
            # Load the state dict
            state_dict = torch.load(diffusion_model_weights_path, map_location=device)
            
            # Create a new state dict with only the keys that exist in our model
            # This handles any potential mismatch between saved weights and model structure
            model_state_dict = diff_model.state_dict()
            for key in state_dict:
                if key in model_state_dict:
                    model_state_dict[key] = state_dict[key]
            
            # Load the filtered state dict
            diff_model.load_state_dict(model_state_dict, strict=False)
            print(f"Loaded diffusion model weights from {diffusion_model_weights_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Continuing with untrained model")
    else:
        print("Warning: No saved diffusion model weights found. Using untrained model.")
        
    diff_model.to(device)
    diff_model.eval()

    # Now generate text
    generated_text = generate_text(
        diff_model, 
        tokenizer, 
        num_diffusion_timesteps, 
        seq_len=args.seq_len, 
        device=device,
        show_steps=args.show_steps,
        step_interval=args.step_interval
    )
    print("\nFinal generated text:")
    print(generated_text)
