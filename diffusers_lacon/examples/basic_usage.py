#!/usr/bin/env python3
"""
Basic usage example for LaCon (Late-Constraint Diffusion) with diffusers

This example demonstrates how to use the LaCon pipeline for controlled image generation
with various condition types (edges, masks, color strokes, etc.).
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Import our custom components
from ..models.condition_aligner import ConditionAligner
from ..pipelines.pipeline_lacon import LaConPipeline
from ..utils.feature_extractor import SimpleFeatureExtractor


def create_edge_condition(image_path: str, size: int = 512) -> Image.Image:
    """Create edge condition from an image using Canny edge detection"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Convert back to 3-channel image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)


def create_mask_condition(size: int = 512) -> Image.Image:
    """Create a simple mask condition"""
    # Create a white canvas
    mask = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(mask)
    
    # Draw some shapes
    draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill='black')
    draw.rectangle([size//3, size//6, 2*size//3, size//3], fill='black')
    
    return mask


def create_color_stroke_condition(size: int = 512) -> Image.Image:
    """Create a color stroke condition"""
    # Create a white canvas
    canvas = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw some color strokes
    draw.line([100, 100, 400, 100], fill='red', width=10)
    draw.line([100, 200, 400, 200], fill='blue', width=10)
    draw.line([100, 300, 400, 300], fill='green', width=10)
    draw.ellipse([200, 350, 300, 450], fill='yellow')
    
    return canvas


def load_lacon_pipeline(
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    condition_aligner_path: str = None,
    device: str = "cuda"
) -> LaConPipeline:
    """
    Load the LaCon pipeline with pretrained components
    
    Args:
        pretrained_model_name_or_path: Path to pretrained Stable Diffusion model
        condition_aligner_path: Path to trained condition aligner weights
        device: Device to load models on
        
    Returns:
        LaConPipeline instance
    """
    # Load base Stable Diffusion pipeline
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # Initialize condition aligner
    condition_aligner = ConditionAligner(
        time_channels=256,
        in_channels=2560,  # This should match the total channels from feature blocks
        out_channels=4,    # VAE latent channels
    )
    
    # Load condition aligner weights if provided
    if condition_aligner_path:
        condition_aligner.load_state_dict(torch.load(condition_aligner_path, map_location="cpu"))
    
    # Create LaCon pipeline
    lacon_pipeline = LaConPipeline(
        vae=base_pipeline.vae,
        text_encoder=base_pipeline.text_encoder,
        tokenizer=base_pipeline.tokenizer,
        unet=base_pipeline.unet,
        scheduler=DDIMScheduler.from_config(base_pipeline.scheduler.config),
        condition_aligner=condition_aligner,
        safety_checker=None,
        feature_extractor=None,
    )
    
    # Move to device
    lacon_pipeline = lacon_pipeline.to(device)
    
    return lacon_pipeline


def example_edge_controlled_generation():
    """Example: Edge-controlled image generation"""
    print("Loading LaCon pipeline...")
    pipeline = load_lacon_pipeline()
    
    print("Creating edge condition...")
    # For this example, we'll create a simple edge condition
    # In practice, you'd use create_edge_condition() with a real image
    edge_condition = create_mask_condition(512)
    
    print("Generating image with edge control...")
    prompt = "a beautiful landscape with mountains and a lake"
    
    # Generate image with condition
    result = pipeline(
        prompt=prompt,
        condition_image=edge_condition,
        num_inference_steps=50,
        guidance_scale=7.5,
        condition_scale=2.0,
        height=512,
        width=512,
        truncation_steps=25,  # Apply condition for first 25 steps
    )
    
    # Save result
    result.images[0].save("edge_controlled_output.png")
    print("Generated image saved as 'edge_controlled_output.png'")


def example_mask_controlled_generation():
    """Example: Mask-controlled image generation"""
    print("Loading LaCon pipeline...")
    pipeline = load_lacon_pipeline()
    
    print("Creating mask condition...")
    mask_condition = create_mask_condition(512)
    
    print("Generating image with mask control...")
    prompt = "a cat sitting in a garden"
    
    # Generate image with condition
    result = pipeline(
        prompt=prompt,
        condition_image=mask_condition,
        num_inference_steps=50,
        guidance_scale=7.5,
        condition_scale=2.0,
        height=512,
        width=512,
        truncation_steps=30,  # Apply condition for first 30 steps
    )
    
    # Save result
    result.images[0].save("mask_controlled_output.png")
    print("Generated image saved as 'mask_controlled_output.png'")


def example_color_stroke_controlled_generation():
    """Example: Color stroke-controlled image generation"""
    print("Loading LaCon pipeline...")
    pipeline = load_lacon_pipeline()
    
    print("Creating color stroke condition...")
    color_condition = create_color_stroke_condition(512)
    
    print("Generating image with color stroke control...")
    prompt = "an abstract painting with vibrant colors"
    
    # Generate image with condition
    result = pipeline(
        prompt=prompt,
        condition_image=color_condition,
        num_inference_steps=50,
        guidance_scale=7.5,
        condition_scale=2.0,
        height=512,
        width=512,
        truncation_steps=35,  # Apply condition for first 35 steps
    )
    
    # Save result
    result.images[0].save("color_stroke_controlled_output.png")
    print("Generated image saved as 'color_stroke_controlled_output.png'")


def example_batch_generation():
    """Example: Batch generation with different conditions"""
    print("Loading LaCon pipeline...")
    pipeline = load_lacon_pipeline()
    
    # Create different conditions
    conditions = [
        create_mask_condition(512),
        create_color_stroke_condition(512),
    ]
    
    prompts = [
        "a futuristic city skyline",
        "a serene forest scene",
    ]
    
    for i, (condition, prompt) in enumerate(zip(conditions, prompts)):
        print(f"Generating image {i+1}/{len(conditions)}...")
        
        result = pipeline(
            prompt=prompt,
            condition_image=condition,
            num_inference_steps=50,
            guidance_scale=7.5,
            condition_scale=2.0,
            height=512,
            width=512,
            truncation_steps=25,
        )
        
        # Save result
        result.images[0].save(f"batch_output_{i+1}.png")
        print(f"Generated image {i+1} saved as 'batch_output_{i+1}.png'")


def main():
    """Main function to run examples"""
    print("=== LaCon Diffusers Examples ===\n")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Run examples
        print("\n1. Edge-controlled generation:")
        example_edge_controlled_generation()
        
        print("\n2. Mask-controlled generation:")
        example_mask_controlled_generation()
        
        print("\n3. Color stroke-controlled generation:")
        example_color_stroke_controlled_generation()
        
        print("\n4. Batch generation:")
        example_batch_generation()
        
        print("\n=== All examples completed! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install diffusers transformers torch torchvision opencv-python pillow")


if __name__ == "__main__":
    main()