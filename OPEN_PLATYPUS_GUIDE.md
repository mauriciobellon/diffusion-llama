# Training the Diffusion Language Model with Open-Platypus

This guide walks you through the process of training and using the diffusion language model with the Open-Platypus dataset.

## 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Extract Instructions from Open-Platypus

The Open-Platypus dataset contains various columns, but we'll use the "instruction" column for training:

```bash
python extract_dataset_column.py --dataset garage-bAInd/Open-Platypus --column instruction --output data/platypus_instructions.txt
```

You can limit the number of samples to extract for a faster test run:

```bash
python extract_dataset_column.py --dataset garage-bAInd/Open-Platypus --column instruction --output data/platypus_instructions.txt --max_samples 5000
```

## 3. Training the Diffusion Model

Now train the diffusion model using the extracted instructions:

### For a quick test run

```bash
python train_with_custom_dataset.py --dataset data/platypus_instructions.txt --batch_size 4 --num_epochs 2 --max_examples 1000
```

### For a more substantial training

```bash
python train_with_custom_dataset.py --dataset data/platypus_instructions.txt --batch_size 8 --num_epochs 5
```

For even better results with more data and longer training:

```bash
python train_with_custom_dataset.py --dataset data/platypus_instructions.txt --batch_size 8 --num_epochs 10 --device cuda
```

## 4. Visualizing the Diffusion Process

To see how the diffusion process unfolds after training:

```bash
python view_diffusion.py --steps 20 --seq_len 16
```

## 5. Generating Text with Prompts

To generate text conditioned on a prompt:

```bash
python generate_with_prompt.py --prompt "Explain the concept of diffusion models" --seq_len 50 --guidance_scale 5.0
```

Try different prompts and guidance scale values to see how they affect the output.

## Training Tips

1. **Start small**: Begin with a smaller subset of the data and fewer epochs to validate your setup.
2. **Increase gradually**: Scale up the dataset size and training duration as you confirm the model is learning properly.
3. **Monitor loss**: The training script will print the loss for each batch and epoch. This should generally decrease over time.
4. **Check generated samples**: Periodically run the generation script to see if the quality of generated text is improving.
5. **Adjust hyperparameters**:
   - Increase `guidance_scale` for stronger adherence to prompts
   - Increase `num_steps` for potentially smoother diffusion process (but longer generation time)
   - Adjust `seq_len` based on your typical input/output length needs

## Dataset Characteristics

The Open-Platypus dataset's instruction column contains a diverse set of prompts and instructions, which makes it well-suited for training a general-purpose language model. The instructions cover various topics including explanations, creative writing, problem-solving, and more.

This variety helps the diffusion model learn to generate a wide range of text styles and content types.
