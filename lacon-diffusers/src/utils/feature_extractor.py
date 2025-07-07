import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class UNetFeatureExtractor:
    """
    Utility class to extract intermediate features from UNet during forward pass.
    
    This class uses PyTorch hooks to capture activations from specified layers
    during the UNet forward pass.
    """
    
    def __init__(self, unet: nn.Module):
        self.unet = unet
        self.features = {}
        self.hooks = []
        
    def _get_block_by_index(self, block_index: int) -> Tuple[Optional[nn.Module], str]:
        """
        Get UNet block by index.
        
        Args:
            block_index: Index of the block to retrieve
            
        Returns:
            Tuple of (block module, block name)
        """
        # Count total blocks in down path
        num_down_blocks = len(self.unet.down_blocks)
        
        if block_index < num_down_blocks:
            return self.unet.down_blocks[block_index], f"down_block_{block_index}"
        
        # Check if it's the middle block
        block_index -= num_down_blocks
        if block_index == 0:
            return self.unet.mid_block, "mid_block"
        
        # Check up blocks
        block_index -= 1
        num_up_blocks = len(self.unet.up_blocks)
        if block_index < num_up_blocks:
            return self.unet.up_blocks[block_index], f"up_block_{block_index}"
            
        return None, ""
    
    def register_hooks(self, block_indices: List[int]):
        """
        Register forward hooks on specified blocks.
        
        Args:
            block_indices: List of block indices to hook
        """
        self.remove_hooks()  # Clean up any existing hooks
        self.features = {}
        
        for idx in block_indices:
            block, name = self._get_block_by_index(idx)
            if block is not None:
                # Register hook on the last layer of the block
                if hasattr(block, 'resnets') and len(block.resnets) > 0:
                    # For residual blocks, hook the last resnet
                    hook_layer = block.resnets[-1]
                elif hasattr(block, 'attentions') and len(block.attentions) > 0:
                    # For attention blocks, hook the last attention
                    hook_layer = block.attentions[-1]
                else:
                    # Fallback to the block itself
                    hook_layer = block
                
                hook = hook_layer.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, name: str):
        """Create a hook function that stores features."""
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # Some layers return (output, skip_connection)
                self.features[name] = output[0]
            else:
                self.features[name] = output
        return hook_fn
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def extract_features(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified blocks during UNet forward pass.
        
        Args:
            sample: Input latent tensor
            timestep: Timestep tensor
            encoder_hidden_states: Text embeddings
            block_indices: Indices of blocks to extract features from
            
        Returns:
            Dictionary mapping block names to feature tensors
        """
        # Register hooks
        self.register_hooks(block_indices)
        
        # Forward pass
        with torch.no_grad():
            _ = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
        
        # Copy features before removing hooks
        features = self.features.copy()
        
        # Clean up
        self.remove_hooks()
        
        return features
    
    def get_feature_shapes(
        self,
        sample_shape: Tuple[int, ...],
        timestep: torch.Tensor,
        encoder_hidden_states_shape: Tuple[int, ...],
        block_indices: List[int],
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Get the shapes of features that would be extracted from specified blocks.
        
        Args:
            sample_shape: Shape of input latent tensor
            timestep: Timestep tensor
            encoder_hidden_states_shape: Shape of text embeddings
            block_indices: Indices of blocks to extract features from
            
        Returns:
            Dictionary mapping block names to feature shapes
        """
        # Create dummy inputs
        device = next(self.unet.parameters()).device
        sample = torch.zeros(sample_shape, device=device)
        encoder_hidden_states = torch.zeros(encoder_hidden_states_shape, device=device)
        
        # Extract features
        features = self.extract_features(
            sample, timestep, encoder_hidden_states, block_indices
        )
        
        # Get shapes
        feature_shapes = {
            name: tuple(feat.shape) for name, feat in features.items()
        }
        
        return feature_shapes