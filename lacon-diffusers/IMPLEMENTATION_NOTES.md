# LaCon-Diffusers Implementation Notes

This document provides technical details about the diffusers implementation of LaCon (Late-Constraint Diffusion).

## Overview

This implementation adapts the original LaCon method from the AlonzoLeeeooo/LCDG repository to work with the Hugging Face diffusers library, providing a more modular and accessible interface for controllable image generation.

## Key Components

### 1. Condition Aligner (`src/models/condition_aligner.py`)

The condition aligner is a neural network that predicts conditions in the latent space from concatenated UNet features:

- **Input**: Concatenated features from multiple UNet blocks
- **Output**: Predicted condition in latent space (4 channels for SD)
- **Architecture**: Series of convolutional blocks with time embedding injection

Key features:
- Sinusoidal timestep embeddings
- Time-conditioned convolutional blocks
- Xavier weight initialization

### 2. LaCon Pipeline (`src/pipelines/pipeline_lacon.py`)

The main pipeline extends `DiffusionPipeline` and implements the late-constraint guidance:

- **Gradient Computation**: Computes gradients through MSE loss between predicted and target conditions
- **Guidance Application**: Applies normalized gradients during DDIM sampling
- **Feature Extraction**: Uses hook-based extraction from UNet blocks

Key methods:
- `_extract_unet_features()`: Extracts intermediate features using hooks
- `_compute_condition_gradient()`: Computes guidance gradients
- `prepare_condition()`: Preprocesses conditions for different types

### 3. UNet Feature Extractor (`src/utils/feature_extractor.py`)

A utility class for clean feature extraction from UNet:

- **Hook Management**: Registers and removes PyTorch hooks
- **Block Indexing**: Maps indices to actual UNet blocks
- **Feature Collection**: Stores features during forward pass

### 4. Training Script (`src/training/train_condition_aligner.py`)

Complete training pipeline for the condition aligner:

- **Dataset**: Handles image-condition-caption triplets
- **Training Loop**: Standard PyTorch training with Accelerate
- **Feature Extraction**: Extracts features at random timesteps
- **Loss**: MSE loss between predicted and target conditions

## Key Differences from Original Implementation

1. **Modular Design**: Separated components for easier maintenance and reuse
2. **Diffusers Integration**: Compatible with existing diffusers models and utilities
3. **Type Safety**: Full type annotations throughout the codebase
4. **Hook-based Features**: Clean feature extraction without modifying UNet
5. **Pipeline Interface**: Standard diffusers pipeline interface

## Implementation Details

### Gradient Guidance Formula

The gradient guidance is computed as:

```
gradient = ∇_x MSE(predicted_condition, target_condition)
normalized_gradient = gradient * condition_scale * ||noise_pred - x|| / ||gradient||
guided_noise_pred = noise_pred - normalized_gradient
```

### Feature Blocks

Default feature extraction blocks for SD v1.4/1.5:
- Block 4: 320 channels, 32x32 spatial
- Block 5: 640 channels, 16x16 spatial  
- Block 7: 1280 channels, 8x8 spatial
- Block 11: 1280 channels, 8x8 spatial

Total concatenated channels: 320 + 640 + 1280 + 1280 = 3520
After upsampling to latent size: 1280 channels (default configuration)

### Condition Types

Supported condition types with preprocessing:
- **edge**: Binary threshold at 0.5
- **mask**: Binary threshold at 0.5
- **color**: RGB normalization to [-1, 1]
- **palette**: RGB normalization to [-1, 1]

### Truncation Steps

Condition guidance is only applied when `t > truncation_steps`:
- Allows initial steps to establish global structure
- Later steps are unconditioned for detail refinement
- Default: 500-600 steps (out of 1000)

## Usage Patterns

### Basic Generation

```python
output = pipe(
    prompt="a beautiful landscape",
    condition=edge_map,
    condition_type="edge",
    condition_scale=2.0,
)
```

### Advanced Control

```python
output = pipe(
    prompt="...",
    condition=condition,
    condition_scale=3.0,  # Stronger guidance
    truncation_steps=700,  # Apply guidance longer
    guidance_scale=5.0,    # Lower CFG
    return_predicted_condition=True,  # For debugging
)
```

## Training Recommendations

1. **Dataset**: ~10K image-condition pairs minimum
2. **Batch Size**: 16-32 depending on GPU memory
3. **Learning Rate**: 1e-4 with cosine annealing
4. **Epochs**: 50-100 depending on dataset size
5. **Feature Blocks**: Default [4, 5, 7, 11] works well

## Future Improvements

1. **Multi-condition Support**: Handle multiple conditions simultaneously
2. **Adaptive Scaling**: Dynamic condition scale based on timestep
3. **Memory Optimization**: Gradient checkpointing for larger batch sizes
4. **Additional Conditions**: Depth maps, normal maps, semantic masks
5. **Scheduler Support**: Beyond DDIM (PNDM, DPM-Solver, etc.)

## References

- Original paper: [LaCon: Late-Constraint Diffusion for Steerable Guided Image Synthesis](https://arxiv.org/abs/2305.11520)
- Original implementation: [AlonzoLeeeooo/LCDG](https://github.com/AlonzoLeeeooo/LCDG)
- Diffusers library: [huggingface/diffusers](https://github.com/huggingface/diffusers)