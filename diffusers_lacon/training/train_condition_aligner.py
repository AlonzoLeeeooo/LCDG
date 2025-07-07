#!/usr/bin/env python3
"""
Training script for LaCon Condition Aligner

This script trains the condition aligner to align diffusion features with target conditions.
"""

import argparse
import os
import random
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
import cv2

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler

from ..models.condition_aligner import ConditionAligner
from ..utils.feature_extractor import SimpleFeatureExtractor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConditionDataset(Dataset):
    """Dataset for condition aligner training"""
    
    def __init__(
        self,
        image_dir: str,
        condition_dir: str,
        caption_dir: str = None,
        condition_type: str = "edge",
        image_size: int = 512,
        max_samples: int = None,
    ):
        """
        Initialize the dataset
        
        Args:
            image_dir: Directory containing images
            condition_dir: Directory containing conditions
            caption_dir: Directory containing captions (optional)
            condition_type: Type of condition (edge, mask, color, etc.)
            image_size: Size to resize images to
            max_samples: Maximum number of samples to load
        """
        self.image_dir = image_dir
        self.condition_dir = condition_dir
        self.caption_dir = caption_dir
        self.condition_type = condition_type
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        logger.info(f"Loaded {len(self.image_files)} samples")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        base_name = os.path.splitext(image_file)[0]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load condition
        condition_path = os.path.join(self.condition_dir, f"{base_name}.png")
        if not os.path.exists(condition_path):
            condition_path = os.path.join(self.condition_dir, f"{base_name}.jpg")
        
        condition = Image.open(condition_path).convert('RGB')
        condition = condition.resize((self.image_size, self.image_size))
        condition = np.array(condition).astype(np.float32) / 255.0
        condition = (condition - 0.5) / 0.5  # Normalize to [-1, 1]
        condition = torch.from_numpy(condition).permute(2, 0, 1)
        
        # Load caption if available
        caption = ""
        if self.caption_dir:
            caption_path = os.path.join(self.caption_dir, f"{base_name}.txt")
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    caption = f.read().strip()
        
        return {
            'image': image,
            'condition': condition,
            'caption': caption,
            'filename': image_file,
        }


def setup_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_condition_aligner(
    args: argparse.Namespace,
    model: ConditionAligner,
    pipeline: StableDiffusionPipeline,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
):
    """
    Train the condition aligner
    
    Args:
        args: Training arguments
        model: Condition aligner model
        pipeline: Stable diffusion pipeline
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
    """
    # Setup
    device = args.device
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Feature extractor
    feature_extractor = SimpleFeatureExtractor(feature_blocks=[[2, 4, 8], [2, 4, 8, 12]])
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.logging_dir)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(device)
            conditions = batch['condition'].to(device)
            captions = batch['caption']
            
            # Encode images to latent space
            with torch.no_grad():
                latents = pipeline.vae.encode(images).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
                
                # Encode conditions to latent space
                condition_latents = pipeline.vae.encode(conditions).latent_dist.sample()
                condition_latents = condition_latents * pipeline.vae.config.scaling_factor
                
                # Encode captions
                if captions[0]:  # If captions are provided
                    text_inputs = pipeline.tokenizer(
                        captions,
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]
                else:
                    # Use empty text embeddings
                    text_embeddings = pipeline.text_encoder(
                        torch.zeros(len(captions), pipeline.tokenizer.model_max_length, dtype=torch.long, device=device)
                    )[0]
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                # Extract features
                features = feature_extractor.extract_features(
                    noisy_latents, timesteps, text_embeddings, target_size=64
                )
            
            # Forward pass
            optimizer.zero_grad()
            condition_pred = model(features, timesteps)["condition_pred"]
            
            # Compute loss
            loss = criterion(condition_pred, condition_latents)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to tensorboard
            if global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    conditions = batch['condition'].to(device)
                    captions = batch['caption']
                    
                    # Same forward pass as training
                    latents = pipeline.vae.encode(images).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                    
                    condition_latents = pipeline.vae.encode(conditions).latent_dist.sample()
                    condition_latents = condition_latents * pipeline.vae.config.scaling_factor
                    
                    if captions[0]:
                        text_inputs = pipeline.tokenizer(
                            captions,
                            padding="max_length",
                            max_length=pipeline.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]
                    else:
                        text_embeddings = pipeline.text_encoder(
                            torch.zeros(len(captions), pipeline.tokenizer.model_max_length, dtype=torch.long, device=device)
                        )[0]
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                    
                    features = feature_extractor.extract_features(
                        noisy_latents, timesteps, text_embeddings, target_size=64
                    )
                    
                    condition_pred = model(features, timesteps)["condition_pred"]
                    loss = criterion(condition_pred, condition_latents)
                    
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            writer.add_scalar("val/loss", avg_val_loss, global_step)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                logger.info(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': epoch_loss / len(train_loader),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    logger.info("Training completed!")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train LaCon Condition Aligner")
    
    # Data arguments
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--condition_dir", type=str, required=True, help="Directory containing condition images")
    parser.add_argument("--caption_dir", type=str, help="Directory containing captions (optional)")
    parser.add_argument("--val_image_dir", type=str, help="Directory containing validation images")
    parser.add_argument("--val_condition_dir", type=str, help="Directory containing validation conditions")
    parser.add_argument("--val_caption_dir", type=str, help="Directory containing validation captions")
    parser.add_argument("--condition_type", type=str, default="edge", choices=["edge", "mask", "color", "stroke"], help="Type of condition")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--max_train_samples", type=int, help="Maximum number of training samples")
    parser.add_argument("--max_val_samples", type=int, help="Maximum number of validation samples")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", 
                       help="Path to pretrained model")
    parser.add_argument("--time_channels", type=int, default=256, help="Time embedding channels")
    parser.add_argument("--in_channels", type=int, default=2560, help="Input feature channels")
    parser.add_argument("--out_channels", type=int, default=4, help="Output channels")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_train_steps", type=int, help="Maximum training steps")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=10, help="Save frequency (epochs)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Setup
    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    
    # Load pipeline
    logger.info("Loading Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.to(args.device)
    
    # Initialize model
    logger.info("Initializing condition aligner...")
    model = ConditionAligner(
        time_channels=args.time_channels,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ConditionDataset(
        image_dir=args.image_dir,
        condition_dir=args.condition_dir,
        caption_dir=args.caption_dir,
        condition_type=args.condition_type,
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    
    val_dataset = None
    if args.val_image_dir and args.val_condition_dir:
        val_dataset = ConditionDataset(
            image_dir=args.val_image_dir,
            condition_dir=args.val_condition_dir,
            caption_dir=args.val_caption_dir,
            condition_type=args.condition_type,
            image_size=args.image_size,
            max_samples=args.max_val_samples,
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Set max_train_steps if not provided
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * len(train_loader)
    
    # Start training
    logger.info("Starting training...")
    train_condition_aligner(args, model, pipeline, train_loader, val_loader)


if __name__ == "__main__":
    main()