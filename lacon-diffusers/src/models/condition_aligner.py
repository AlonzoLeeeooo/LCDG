import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from einops import repeat


class TimestepEmbedding(nn.Module):
    """Creates sinusoidal timestep embeddings."""
    
    def __init__(self, time_channels: int, max_period: int = 10000):
        super().__init__()
        self.time_channels = time_channels
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        Returns:
            An [N x time_channels] Tensor of positional embeddings.
        """
        half = self.time_channels // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ConvBlock(nn.Module):
    """Convolutional block with time embedding."""
    
    def __init__(self, time_channels: int, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            t_emb: Time embedding of shape (batch_size, time_channels)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        t_out = self.time_mlp(t_emb)
        x = self.conv(x)
        
        # Reshape time embedding to match spatial dimensions
        while len(t_out.shape) < len(x.shape):
            t_out = t_out[..., None]
            
        return x + t_out


class ConditionAligner(nn.Module):
    """
    Condition Aligner model for Late-Constraint Diffusion.
    
    This model takes concatenated features from different layers of a diffusion model
    and predicts the corresponding condition (e.g., edge map, mask, color stroke) in
    the latent space.
    """
    
    def __init__(
        self, 
        time_channels: int = 128,
        in_channels: int = 1280,  # Default for concatenated SD features
        out_channels: int = 4,     # Default for SD latent space
        hidden_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.time_channels = time_channels
        
        if hidden_dims is None:
            hidden_dims = (640, 320, 512, 256, 128, 64)
            
        # Time embedding module
        self.time_embed = TimestepEmbedding(time_channels)
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        # First block
        self.conv_blocks.append(
            ConvBlock(time_channels, in_channels, hidden_dims[0], 1, 1)
        )
        
        # Intermediate blocks
        for i in range(len(hidden_dims) - 1):
            kernel_size = 3 if i >= 2 else 1
            padding = 1 if kernel_size == 3 else 0
            self.conv_blocks.append(
                ConvBlock(time_channels, hidden_dims[i], hidden_dims[i+1], 
                         kernel_size, 1, padding)
            )
        
        # Final projection to output channels
        self.final_conv = nn.Conv2d(hidden_dims[-1], out_channels, 1, 1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self, init_type: str = 'xavier', gain: float = 0.02):
        """Initialize network weights."""
        
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the condition aligner.
        
        Args:
            x: Concatenated features from diffusion model of shape 
               (batch_size, in_channels, height, width)
            timesteps: Timesteps of shape (batch_size,)
            
        Returns:
            Predicted condition in latent space of shape 
            (batch_size, out_channels, height, width)
        """
        # Get time embeddings
        t_emb = self.time_embed(timesteps)
        
        # Pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x, t_emb)
            
        # Final projection
        x = self.final_conv(x)
        
        return x