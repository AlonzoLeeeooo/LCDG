"""
Training script for LaCon Condition Aligner.

This script trains a condition aligner to predict conditions (edge maps, masks, etc.)
from intermediate UNet features during the diffusion process.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from ..models import ConditionAligner
from ..utils import UNetFeatureExtractor


class ConditionAlignerDataset(Dataset):
    """
    Dataset for training condition aligner.
    
    Expects data structure:
    - images/: Original images
    - conditions/: Corresponding conditions (edges, masks, etc.)
    - captions/: Text captions for each image
    """
    
    def __init__(
        self,
        data_root: str,
        condition_type: str = "edge",
        image_size: int = 512,
        vae_scale_factor: int = 8,
    ):
        self.data_root = Path(data_root)
        self.condition_type = condition_type
        self.image_size = image_size
        self.latent_size = image_size // vae_scale_factor
        
        # Get file list
        self.image_files = sorted(list((self.data_root / "images").glob("*.png")))
        self.condition_files = sorted(list((self.data_root / "conditions").glob("*.png")))
        self.caption_files = sorted(list((self.data_root / "captions").glob("*.txt")))
        
        assert len(self.image_files) == len(self.condition_files) == len(self.caption_files), \
            "Number of images, conditions, and captions must match"
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = Image.open(self.image_files[idx]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load condition
        condition = Image.open(self.condition_files[idx])
        if self.condition_type in ["edge", "mask"]:
            condition = condition.convert("L")
        else:
            condition = condition.convert("RGB")
            
        condition = condition.resize((self.image_size, self.image_size), Image.BILINEAR)
        condition = np.array(condition).astype(np.float32)
        
        # Preprocess condition based on type
        if self.condition_type == "edge":
            condition = (condition > 180.0).astype(np.float32) * 255.0
        elif self.condition_type == "mask":
            condition = (condition > 127.5).astype(np.float32) * 255.0
            
        condition = condition / 127.5 - 1.0
        
        if len(condition.shape) == 2:
            condition = np.stack([condition] * 3, axis=-1)
            
        condition = torch.from_numpy(condition).permute(2, 0, 1)
        
        # Load caption
        with open(self.caption_files[idx], 'r') as f:
            caption = f.read().strip()
            
        return {
            "image": image,
            "condition": condition,
            "caption": caption,
            "filename": self.image_files[idx].stem,
        }


class ConditionAlignerTrainer:
    """Trainer for condition aligner model."""
    
    def __init__(
        self,
        model: ConditionAligner,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        feature_blocks: List[int] = [4, 5, 7, 11],
        device: str = "cuda",
        output_dir: str = "outputs",
    ):
        self.model = model
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.feature_blocks = feature_blocks
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            
        # Setup optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Feature extractor
        self.feature_extractor = UNetFeatureExtractor(self.unet)
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler
            )
            
        if self.val_loader:
            self.val_loader = self.accelerator.prepare(self.val_loader)
            
        # Move other models to device
        self.vae = self.vae.to(self.accelerator.device)
        self.unet = self.unet.to(self.accelerator.device)
        self.text_encoder = self.text_encoder.to(self.accelerator.device)
        
        # Set to eval mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for batch in progress_bar:
            # Move data to device
            images = batch["image"]
            conditions = batch["condition"]
            captions = batch["caption"]
            
            with torch.no_grad():
                # Encode images and conditions to latent space
                image_latents = self.vae.encode(images).latent_dist.sample()
                image_latents = image_latents * self.vae.config.scaling_factor
                
                condition_latents = self.vae.encode(conditions).latent_dist.sample()
                condition_latents = condition_latents * self.vae.config.scaling_factor
                
                # Encode text
                text_inputs = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.accelerator.device)
                
                text_embeddings = self.text_encoder(
                    text_inputs.input_ids
                )[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 1000, (images.shape[0],), device=self.accelerator.device
                ).long()
                
                # Add noise to latents
                noise = torch.randn_like(image_latents)
                noisy_latents = self.vae.config.scaling_factor * \
                    torch.sqrt(self.unet.config.alpha_cumprod[timesteps][:, None, None, None]) * image_latents + \
                    torch.sqrt(1 - self.unet.config.alpha_cumprod[timesteps][:, None, None, None]) * noise
                
                # Extract features from UNet
                features = self.feature_extractor.extract_features(
                    noisy_latents,
                    timesteps,
                    text_embeddings,
                    self.feature_blocks,
                )
                
                # Concatenate and upsample features
                feature_list = [features[f"down_block_{i}"] for i in self.feature_blocks if f"down_block_{i}" in features]
                upsampled_features = []
                for feat in feature_list:
                    if feat.shape[-2:] != condition_latents.shape[-2:]:
                        feat = F.interpolate(
                            feat,
                            size=condition_latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    upsampled_features.append(feat)
                    
                concatenated_features = torch.cat(upsampled_features, dim=1)
            
            # Forward pass through condition aligner
            predicted_conditions = self.model(concatenated_features, timesteps)
            
            # Compute loss
            loss = self.criterion(predicted_conditions, condition_latents)
            
            # Backward pass
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        if not self.val_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"]
                conditions = batch["condition"]
                captions = batch["caption"]
                
                # Process batch (same as training but without gradients)
                # ... (similar to training loop)
                
                pass  # Implement validation logic
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log
            if self.accelerator.is_local_main_process:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Save checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    
                # Save regular checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch)
                    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
            
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "feature_blocks": self.feature_blocks,
        }
        
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, self.output_dir / filename)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset")
    parser.add_argument("--condition_type", type=str, default="edge", help="Type of condition")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    # Load models
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    
    # Initialize condition aligner
    condition_aligner = ConditionAligner(
        time_channels=128,
        in_channels=1280,
        out_channels=4,
    )
    
    # Create dataset
    train_dataset = ConditionAlignerDataset(
        data_root=args.data_root,
        condition_type=args.condition_type,
    )
    
    # Create trainer
    trainer = ConditionAlignerTrainer(
        model=condition_aligner,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()