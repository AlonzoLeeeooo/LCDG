# LaCon-Diffusers: Late-Constraint Diffusion for Diffusers

A diffusers-compatible implementation of [LaCon (Late-Constraint Diffusion)](https://arxiv.org/abs/2305.11520) for controllable image generation with Stable Diffusion.

## Overview

LaCon (Late-Constraint Diffusion) is a method for steerable guided image synthesis that incorporates external conditions (such as edge maps, masks, or color strokes) during the diffusion sampling process. Unlike early-constraint methods that modify the model architecture, LaCon uses a condition aligner to establish alignment between external conditions and internal diffusion model features, enabling flexible control without retraining the base model.

This implementation provides a diffusers-compatible pipeline that integrates seamlessly with Hugging Face's diffusers library.

## Features

- 🎨 **Multiple Condition Types**: Support for edge maps, masks, color strokes, and more
- 🔧 **Diffusers Integration**: Compatible with Hugging Face diffusers ecosystem
- 🚀 **Flexible Control**: Adjustable condition strength and truncation steps
- 📊 **Gradient-based Guidance**: Uses gradient information to guide generation
- 🔍 **Feature Extraction**: Efficient extraction of intermediate UNet features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lacon-diffusers.git
cd lacon-diffusers

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.pipelines import LaConPipeline
from src.models import ConditionAligner
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

# Load base Stable Diffusion components
model_id = "runwayml/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Initialize condition aligner
condition_aligner = ConditionAligner(
    time_channels=128,
    in_channels=1280,  # Concatenated SD features
    out_channels=4,    # SD latent space
)

# Create pipeline
pipe = LaConPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    condition_aligner=condition_aligner,
)

# Generate with edge guidance
edge_map = Image.open("path/to/edge_map.png")
output = pipe(
    prompt="a beautiful landscape",
    condition=edge_map,
    condition_type="edge",
    condition_scale=2.0,
)
output.images[0].save("result.png")
```

## Architecture

### Condition Aligner

The condition aligner is a neural network that takes concatenated features from different layers of the UNet and predicts the corresponding condition in the latent space:

```python
ConditionAligner(
    time_channels=128,      # Timestep embedding dimension
    in_channels=1280,       # Input features dimension
    out_channels=4,         # Output latent dimension
    hidden_dims=None,       # Custom hidden layer dimensions
)
```

### Pipeline Parameters

Key parameters for the `LaConPipeline`:

- `condition`: The control condition (edge map, mask, etc.)
- `condition_type`: Type of condition ("edge", "mask", "color", etc.)
- `condition_scale`: Strength of condition guidance (default: 2.0)
- `truncation_steps`: Steps to apply condition guidance (default: 500)
- `guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `num_inference_steps`: Number of denoising steps (default: 50)

## Supported Conditions

| Condition Type | Description | Preprocessing |
|---------------|-------------|---------------|
| `edge` | Edge maps (Canny, HED, etc.) | Binary threshold at 0.5 |
| `mask` | Binary masks | Binary threshold at 0.5 |
| `color` | Color strokes | RGB normalization |
| `palette` | Image color palette | RGB normalization |

## Training Condition Aligner

To train your own condition aligner:

```python
# Training code example (to be implemented)
from src.training import train_condition_aligner

train_condition_aligner(
    dataset_path="path/to/dataset",
    condition_type="edge",
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=100,
)
```

## Advanced Usage

### Custom Feature Extraction

You can customize which UNet blocks to extract features from:

```python
# Specify custom block indices
pipe.feature_blocks["custom"] = [3, 5, 8, 10]

# Use in generation
output = pipe(
    prompt="...",
    condition=condition,
    block_indices=pipe.feature_blocks["custom"],
)
```

### Gradient Normalization

The gradient guidance is normalized to maintain stable generation:

```python
gradient = gradient * condition_scale * ||e_t - x|| / ||gradient||
```

## Key Differences from Original Implementation

1. **Diffusers Integration**: Built on top of diffusers library for easy integration
2. **Modular Design**: Separate components for condition aligner, feature extraction, and pipeline
3. **Type Safety**: Full type annotations for better developer experience
4. **Hook-based Feature Extraction**: Uses PyTorch hooks for clean feature extraction

## Model Weights

Pre-trained condition aligner weights for different conditions:

| Condition | Model | Link |
|-----------|-------|------|
| Edge (SD 1.4) | `sdv14_edge.pth` | [Download](#) |
| Mask (SD 1.4) | `sdv14_mask.pth` | [Download](#) |
| Color (SD 1.4) | `sdv14_color.pth` | [Download](#) |

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{liu-etal-2024-lacon,
      title={{LaCon: Late-Constraint Diffusion for Steerable Guided Image Synthesis}}, 
      author={{Chang Liu, Rui Li, Kaidong Zhang, Xin Luo, and Dong Liu}},
      year={2024},
      eprint={2305.11520},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

- Original LaCon implementation: [AlonzoLeeeooo/LCDG](https://github.com/AlonzoLeeeooo/LCDG)
- Built on top of [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.