"""
Example script for edge-guided image generation using LaCon pipeline.

This example shows how to use an edge map to guide the generation process
with Late-Constraint Diffusion.
"""

import torch
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path to import our modules
import sys
sys.path.append('..')

from src.pipelines import LaConPipeline
from src.models import ConditionAligner


def load_edge_map(image_path: str, target_size: tuple = (512, 512)) -> Image.Image:
    """
    Load and preprocess an edge map.
    
    Args:
        image_path: Path to edge map image
        target_size: Target size for the image
        
    Returns:
        Preprocessed edge map as PIL Image
    """
    edge_map = Image.open(image_path).convert('L')
    edge_map = edge_map.resize(target_size, Image.BILINEAR)
    return edge_map


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model IDs
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load models
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Initialize condition aligner
    # Note: You would need to load pre-trained weights here
    condition_aligner = ConditionAligner(
        time_channels=128,
        in_channels=1280,  # Default for concatenated SD features
        out_channels=4,    # SD latent space channels
    )
    
    # Load pre-trained condition aligner weights if available
    # condition_aligner.load_state_dict(torch.load("path/to/condition_aligner_weights.pth"))
    
    # Create pipeline
    pipe = LaConPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        condition_aligner=condition_aligner,
    )
    pipe = pipe.to(device)
    
    # Load edge map
    # You can create an edge map using Canny edge detection or other methods
    # For this example, we'll create a simple synthetic edge map
    edge_map = Image.new('L', (512, 512), 0)
    # In practice, you would load a real edge map:
    # edge_map = load_edge_map("path/to/edge_map.png")
    
    # Generation parameters
    prompt = "a beautiful landscape with mountains and a lake, highly detailed"
    negative_prompt = "low quality, blurry, distorted"
    
    # Generate image
    print("Generating image...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        condition=edge_map,
        condition_type="edge",
        condition_scale=2.0,
        num_inference_steps=50,
        guidance_scale=7.5,
        truncation_steps=600,
        return_predicted_condition=True,
    )
    
    # Save results
    output.images[0].save("generated_image.png")
    print("Generated image saved to generated_image.png")
    
    if output.predicted_conditions:
        output.predicted_conditions[0].save("predicted_condition.png")
        print("Predicted condition saved to predicted_condition.png")


if __name__ == "__main__":
    main()