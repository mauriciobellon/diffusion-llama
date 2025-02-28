#!/usr/bin/env python3
"""
Modified inference script that allows using a prompt to condition the diffusion process.
This script attempts to guide the diffusion model using a provided prompt.
"""

import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dlm_train import DiffusionLM

def get_noise_schedule(T, beta_start=0.0001, beta_end=0.02, device="cpu"):
    """Get diffusion noise schedule parameters."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    return betas, alphas, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar

def generate_with_prompt(diff_model, tokenizer, prompt, num_diffusion_timesteps, 
                        seq_len=32, guidance_scale=5.0, device="cpu", 
                        show_steps=True, step_interval=10):
    """
    Generate text conditioned on a prompt using the diffusion model.
    
    Args:
        diff_model: The diffusion language model
        tokenizer: The tokenizer
        prompt: Text prompt to condition the generation
        num_diffusion_timesteps: Number of diffusion steps
        seq_len: Length of sequence to generate (including prompt)
        guidance_scale: How strongly to condition on the prompt (higher = stronger)
        device: Device to run on
        show_steps: Whether to show intermediate results
        step_interval: Interval between displaying steps
        
    Returns:
        generated_text: The generated text including the prompt
    """
    # Get noise schedule parameters
    _, _, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = get_noise_schedule(
        num_diffusion_timesteps, device=device
    )
    
    # Tokenize the prompt
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    
    # Make sure we don't exceed the sequence length
    if prompt_len >= seq_len:
        print(f"Warning: Prompt length ({prompt_len}) is >= seq_len ({seq_len}). Truncating prompt.")
        prompt_ids = prompt_ids[:, :seq_len-1]
        prompt_len = prompt_ids.shape[1]
    
    # Get prompt embeddings
    with torch.no_grad():
        prompt_emb = diff_model.llama.model.embed_tokens(prompt_ids)
    
    # Initialize noise
    hidden_size = diff_model.hidden_size
    x_t = torch.randn((1, seq_len, hidden_size), device=device)
    
    # Copy prompt embeddings to the first part of the noisy embeddings
    # This is a simple way to condition, more sophisticated methods exist
    x_t_prompt = x_t.clone()
    x_t_prompt[:, :prompt_len, :] = prompt_emb
    
    # Prepare for tracking intermediate results
    intermediate_texts = []
    if show_steps:
        print(f"\nStarting diffusion sampling with {num_diffusion_timesteps} steps")
        print(f"Using prompt: '{prompt}'")
        print(f"Will display intermediate results every {step_interval} steps")
    
    # Iteratively apply the reverse diffusion process
    diff_model.eval()
    with torch.no_grad():
        for t in reversed(range(num_diffusion_timesteps)):
            # Report progress
            if show_steps and (t % step_interval == 0 or t == num_diffusion_timesteps - 1 or t == 0):
                print(f"\nStep {num_diffusion_timesteps-t}/{num_diffusion_timesteps} (t={t})")
            
            # Create a batch tensor for the current timestep
            t_tensor = torch.tensor([t], device=device).long()
            dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            
            # Predict the noise for both conditional and unconditional
            # Unconditional prediction (no prompt guidance)
            predicted_noise_uncond = diff_model(
                input_ids=dummy_input_ids,
                timestep=t_tensor,
                noisy_embeddings=x_t,
                attention_mask=torch.ones((1, seq_len), device=device)
            )
            
            # Conditional prediction (with prompt)
            predicted_noise_cond = diff_model(
                input_ids=dummy_input_ids,
                timestep=t_tensor,
                noisy_embeddings=x_t_prompt,
                attention_mask=torch.ones((1, seq_len), device=device)
            )
            
            # Apply classifier-free guidance
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)
            
            # Calculate the scaling factors for this timestep
            sqrt_alpha_t = sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_bar[t]
            
            # Estimate the clean embedding
            x_0_est = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            
            # Force the prompt part to stay close to the original prompt
            if prompt_len > 0:
                x_0_est[:, :prompt_len, :] = prompt_emb
            
            # For steps > 0, add some noise
            if t > 0:
                noise = torch.randn_like(x_t)
                beta_t = (1.0 - sqrt_alpha_t**2)
                x_t = x_0_est + torch.sqrt(beta_t) * noise
                
                # Ensure the prompt part remains influenced by the original
                x_t[:, :prompt_len, :] = 0.8 * prompt_emb + 0.2 * x_t[:, :prompt_len, :]
            else:
                x_t = x_0_est
                
                # Final step - make sure prompt is preserved exactly
                if prompt_len > 0:
                    x_t[:, :prompt_len, :] = prompt_emb
            
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
                
                # Highlight the prompt part if possible
                if prompt_len > 0 and len(current_text) > len(prompt):
                    display_text = f"{current_text[:len(prompt)]} | {current_text[len(prompt):]}"
                else:
                    display_text = current_text
                
                # Store and print the intermediate result
                intermediate_texts.append((t, current_text))
                print(f"  Text: {display_text}")
    
    # Convert final embeddings to tokens
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
            if len(text) > 50:
                text = text[:47] + "..."
            print(f"Step {i+1}/{len(intermediate_texts)} (t={step}): {text}")
        print("================================")
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text with a prompt using the diffusion model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt to condition the generation")
    parser.add_argument("--model_path", type=str, default="qwen/Qwen2.5-3B", help="Path to base model")
    parser.add_argument("--diffusion_model_path", type=str, default="./projects/diffusion_model", help="Path to diffusion model")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of diffusion steps")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length to generate")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale (higher = more prompt influence)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--step_interval", type=int, default=10, help="Interval between displaying steps")
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
    
    # Check if we have a cached model
    cache_dir = "./projects"
    cache_model_path = os.path.join(cache_dir, "models--qwen--Qwen2.5-3B")
    
    print(f"Loading model from {model_path}")
    
    # Load models
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
    
    # Create diffusion model
    diff_model = DiffusionLM(llama_model, num_diffusion_timesteps)
    
    # Load the saved diffusion model weights if they exist
    diffusion_model_weights_path = os.path.join(args.diffusion_model_path, "diffusion_model.pt")
    if os.path.exists(diffusion_model_weights_path):
        try:
            # Load the state dict
            state_dict = torch.load(diffusion_model_weights_path, map_location=device)
            
            # Create a new state dict with only the keys that exist in our model
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
    
    print(f"\nGenerating text with prompt: '{args.prompt}'")
    print(f"Using {num_diffusion_timesteps} diffusion steps and guidance scale {args.guidance_scale}")
    
    # Generate text with the prompt
    generated_text = generate_with_prompt(
        diff_model,
        tokenizer,
        args.prompt,
        num_diffusion_timesteps,
        seq_len=args.seq_len,
        guidance_scale=args.guidance_scale,
        device=device,
        show_steps=True,
        step_interval=args.step_interval
    )
    
    print("\nFinal generated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Also show just the continuation
    if len(args.prompt) < len(generated_text):
        continuation = generated_text[len(args.prompt):]
        print("\nContinuation only:")
        print("-" * 50)
        print(continuation)
        print("-" * 50)

if __name__ == "__main__":
    main() 