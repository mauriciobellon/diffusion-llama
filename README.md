# Diffusion Language Model (DLM)

This repository contains an implementation of a diffusion-based language model using a pretrained Qwen model. The diffusion process gradually denoises a random embedding to produce coherent text.

## Structure

- `dlm_train.py`: Training script for the diffusion language model
- `dlm_test.py`: Inference script for generating text from the trained diffusion model
- `view_diffusion.py`: Utility script for visualizing the diffusion process with a small number of steps
- `train_with_custom_dataset.py`: Script for training the model with a custom text dataset
- `generate_with_prompt.py`: Script for generating text conditioned on a prompt

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)

You can install the required packages with:

```bash
pip install torch transformers
```

## Training

### Basic Training

To train the diffusion language model with default parameters:

```bash
python dlm_train.py --num_steps 1000 --batch_size 8 --learning_rate 1e-4 --num_epochs 10
```

### Training with a Custom Dataset

To train with your own text dataset:

```bash
python train_with_custom_dataset.py --dataset path/to/your/dataset.txt --batch_size 8 --num_epochs 5
```

The dataset file should contain one text example per line.

Key parameters:

- `--dataset`: Path to text file with one example per line
- `--max_examples`: Maximum number of examples to load from dataset (optional)
- `--num_steps`: Number of diffusion steps (default: 1000)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 3)
- `--device`: Device to use for training (default: cuda if available, else cpu)

## Inference

### Basic Inference

After training, you can generate text using the diffusion model:

```bash
python dlm_test.py --seq_len 32 --show_steps --step_interval 100
```

Key parameters:

- `--seq_len`: Length of the output sequence (number of tokens, default: 32)
- `--show_steps`: Flag to show intermediate steps during the diffusion process
- `--step_interval`: Interval between displaying steps (default: 100)
- `--num_steps`: Override the number of diffusion steps (default: use value from config)
- `--device`: Device to use (default: cuda if available, else cpu)

### Generation with a Prompt

To generate text conditioned on a specific prompt:

```bash
python generate_with_prompt.py --prompt "Once upon a time" --seq_len 40 --guidance_scale 5.0
```

This uses a technique similar to classifier-free guidance to help the model extend your prompt.

Key parameters:

- `--prompt`: Text prompt to condition the generation
- `--seq_len`: Total sequence length to generate (including prompt)
- `--guidance_scale`: How strongly to condition on the prompt (higher = stronger, default: 5.0)
- `--num_steps`: Number of diffusion steps to use
- `--step_interval`: Interval between displaying steps (default: 10)
- `--device`: Device to use (default: cuda if available, else cpu)

## Visualizing the Diffusion Process

To visualize the diffusion process with a small number of steps:

```bash
python view_diffusion.py --steps 20 --seq_len 8
```

This script makes it easier to see how the text evolves during each step of the diffusion process by:

1. Using a small number of steps (default: 20)
2. Showing results after each step
3. Generating a shorter sequence (default: 8 tokens)

Key parameters:

- `--steps`: Number of diffusion steps to use (default: 20)
- `--seq_len`: Length of generated sequence (default: 8)
- `--device`: Device to use (default: cpu)

## Model Files

The scripts will save and load model files in the following directories:

- Diffusion model: `./projects/diffusion_model/`
- Cached base models: `./projects/`

## Diffusion Process

The diffusion process involves:

1. Starting with random Gaussian noise
2. Iteratively denoising the embeddings using the trained model
3. Converting the final denoised embeddings to text

The visualization shows how the text evolves from random noise to coherent output during this process.
