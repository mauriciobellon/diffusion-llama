#!/usr/bin/env python3
"""
A simple script to visualize the diffusion process with a small number of steps.
This makes it easier to see how the text evolves during the diffusion process.
"""

import os
import argparse
import torch
from dlm_train import DiffusionLM
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Visualize diffusion language model steps")
    parser.add_argument("--steps", type=int, default=20, help="Number of diffusion steps to use")
    parser.add_argument("--seq_len", type=int, default=8, help="Length of generated sequence")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Set paths
    model_path = "qwen/Qwen2.5-3B"
    diffusion_model_path = "./projects/diffusion_model"
    cache_dir = "./projects"
    
    # Check if we have saved diffusion model parameters
    config_path = os.path.join(diffusion_model_path, "config.pt")
    if os.path.exists(config_path):
        config_info = torch.load(config_path, map_location=device)
        model_path = config_info.get("base_model_path", model_path)
        print(f"Using model: {model_path}")
    else:
        print(f"No config found, using default model: {model_path}")
    
    # Load the base language model
    cache_model_path = os.path.join(cache_dir, "models--qwen--Qwen2.5-3B")
    try:
        if os.path.exists(cache_model_path) and model_path == "qwen/Qwen2.5-3B":
            print(f"Loading cached model from {cache_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(cache_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(cache_model_path)
        else:
            print(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to direct download")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        base_model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    
    base_model.to(device)
    base_model.eval()
    
    # Create diffusion model
    diff_model = DiffusionLM(base_model, args.steps)
    
    # Try to load trained weights
    weights_path = os.path.join(diffusion_model_path, "diffusion_model.pt")
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model_state_dict = diff_model.state_dict()
            for key in state_dict:
                if key in model_state_dict:
                    model_state_dict[key] = state_dict[key]
            diff_model.load_state_dict(model_state_dict, strict=False)
            print(f"Loaded diffusion weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using untrained model")
    else:
        print("No saved weights found, using untrained model")
    
    diff_model.to(device)
    diff_model.eval()
    
    # Import generate_text here to avoid circular imports
    from dlm_test import generate_text
    
    # Generate text with visualization
    print(f"\n{'-'*60}")
    print(f"Visualizing diffusion process with {args.steps} steps")
    print(f"{'-'*60}")
    
    generated_text = generate_text(
        diff_model,
        tokenizer,
        args.steps,
        seq_len=args.seq_len,
        device=device,
        show_steps=True,
        step_interval=1  # Show every step
    )
    
    print(f"\n{'-'*60}")
    print("Final generated text:")
    print(generated_text)
    print(f"{'-'*60}")

if __name__ == "__main__":
    main() 