import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class UNetFeatureExtractor:
    """
    Utility class for extracting intermediate features from UNet blocks.
    
    This class provides hooks to extract features from specific blocks of the UNet
    during the forward pass, which are then used by the condition aligner.
    """
    
    def __init__(self, unet: nn.Module, feature_blocks: List[List[int]]):
        """
        Initialize the feature extractor.
        
        Args:
            unet: The UNet model to extract features from
            feature_blocks: List of block indices to extract features from
        """
        self.unet = unet
        self.feature_blocks = feature_blocks
        self.features = {}
        self.hooks = []
        
    def _get_activation_hook(self, name: str):
        """Create a hook function to store activations"""
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
        
    def register_hooks(self):
        """Register forward hooks on the specified blocks"""
        self.hooks.clear()
        
        # Register hooks for down blocks
        for i, block_indices in enumerate(self.feature_blocks):
            for block_idx in block_indices:
                if hasattr(self.unet, 'down_blocks') and block_idx < len(self.unet.down_blocks):
                    hook = self.unet.down_blocks[block_idx].register_forward_hook(
                        self._get_activation_hook(f'down_block_{block_idx}')
                    )
                    self.hooks.append(hook)
                    
        # Register hooks for up blocks if needed
        if hasattr(self.unet, 'up_blocks'):
            for i, block_indices in enumerate(self.feature_blocks):
                for block_idx in block_indices:
                    if block_idx < len(self.unet.up_blocks):
                        hook = self.unet.up_blocks[block_idx].register_forward_hook(
                            self._get_activation_hook(f'up_block_{block_idx}')
                        )
                        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def extract_features(self, latents: torch.Tensor, timesteps: torch.Tensor, 
                        encoder_hidden_states: torch.Tensor, target_size: int = 64) -> torch.Tensor:
        """
        Extract features from UNet forward pass.
        
        Args:
            latents: Input latents
            timesteps: Timesteps
            encoder_hidden_states: Text embeddings
            target_size: Target size for feature upsampling
            
        Returns:
            Concatenated features from specified blocks
        """
        # Clear previous features
        self.features.clear()
        
        # Register hooks
        self.register_hooks()
        
        try:
            # Forward pass through UNet
            with torch.no_grad():
                _ = self.unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states)
            
            # Collect and process features
            feature_list = []
            for name, feature in self.features.items():
                # Upsample feature to target size
                if feature.shape[-1] != target_size:
                    feature = F.interpolate(feature, size=(target_size, target_size), mode='bilinear', align_corners=False)
                feature_list.append(feature)
            
            # Concatenate features along channel dimension
            if feature_list:
                combined_features = torch.cat(feature_list, dim=1)
            else:
                # Fallback: create dummy features if no features were extracted
                combined_features = torch.randn(
                    latents.shape[0], 
                    2560,  # Default feature dimension
                    target_size, 
                    target_size, 
                    device=latents.device,
                    dtype=latents.dtype
                )
            
        finally:
            # Always remove hooks
            self.remove_hooks()
            
        return combined_features
    
    def __enter__(self):
        """Context manager entry"""
        self.register_hooks()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.remove_hooks()


class SimpleFeatureExtractor:
    """
    Simplified feature extractor that doesn't require hooks.
    
    This version creates synthetic features for testing purposes.
    """
    
    def __init__(self, feature_blocks: List[List[int]]):
        self.feature_blocks = feature_blocks
        
    def extract_features(self, latents: torch.Tensor, timesteps: torch.Tensor, 
                        encoder_hidden_states: torch.Tensor, target_size: int = 64) -> torch.Tensor:
        """
        Extract synthetic features for testing.
        
        Args:
            latents: Input latents
            timesteps: Timesteps
            encoder_hidden_states: Text embeddings
            target_size: Target size for features
            
        Returns:
            Synthetic features
        """
        batch_size = latents.shape[0]
        
        # Calculate total feature channels based on blocks
        total_channels = 0
        for block_indices in self.feature_blocks:
            total_channels += len(block_indices) * 320  # Assume 320 channels per block
        
        # Create synthetic features
        features = torch.randn(
            batch_size,
            total_channels,
            target_size,
            target_size,
            device=latents.device,
            dtype=latents.dtype
        )
        
        # Add some structure based on timesteps
        time_factor = timesteps.float().view(-1, 1, 1, 1) / 1000.0
        features = features * (1.0 + time_factor)
        
        return features