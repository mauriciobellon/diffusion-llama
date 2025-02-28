#!/usr/bin/env python3
"""
Extract a specific column from a Hugging Face dataset and save it to a text file.
This script is useful for preparing training data for the diffusion language model.

Example usage:
    python extract_dataset_column.py --dataset garage-bAInd/Open-Platypus --column instruction --output platypus_instructions.txt
"""

import argparse
import os
from datasets import load_dataset

def extract_column_to_file(dataset_name, column_name, output_file, split="train", max_samples=None):
    """
    Extract a column from a Hugging Face dataset and save it to a text file.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face (e.g., "garage-bAInd/Open-Platypus")
        column_name (str): Name of the column to extract (e.g., "instruction")
        output_file (str): Path to save the extracted column data
        split (str): Which split of the dataset to use (default: "train")
        max_samples (int, optional): Maximum number of samples to extract
    
    Returns:
        int: Number of samples extracted
    """
    print(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Check if the column exists
        if column_name not in dataset.column_names:
            available_columns = ", ".join(dataset.column_names)
            raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {available_columns}")
        
        # Extract the column
        column_data = dataset[column_name]
        
        # Limit to max_samples if specified
        if max_samples is not None and max_samples < len(column_data):
            column_data = column_data[:max_samples]
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            for item in column_data:
                # Skip None or empty values
                if item is not None and item.strip():
                    f.write(item.strip() + "\n")
        
        print(f"Successfully extracted {len(column_data)} samples to {output_file}")
        return len(column_data)
    
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract a column from a Hugging Face dataset and save it to a text file")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset on Hugging Face")
    parser.add_argument("--column", type=str, required=True, help="Name of the column to extract")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to extract")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Extract the column
    num_samples = extract_column_to_file(
        args.dataset, 
        args.column, 
        args.output,
        args.split,
        args.max_samples
    )
    
    if num_samples > 0:
        print(f"\nNext steps:")
        print(f"1. Train your diffusion model with this dataset:")
        print(f"   python train_with_custom_dataset.py --dataset {args.output} --batch_size 8 --num_epochs 5")
        print(f"\nFor a smaller test run, use:")
        print(f"   python train_with_custom_dataset.py --dataset {args.output} --batch_size 4 --num_epochs 2 --max_examples 1000")

if __name__ == "__main__":
    main() 