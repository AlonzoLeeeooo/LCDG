import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class ConvBlock(nn.Module):
    """Conv block with time embedding support"""
    def __init__(self, time_ch: int, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, out_ch),
        )
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(time_emb)
        x = self.conv(x)
        # Reshape time embedding to match spatial dimensions
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]
        return x + t_emb


class ConditionAligner(ModelMixin, ConfigMixin):
    """
    Condition Aligner model for LaCon (Late-Constraint Diffusion).
    
    This model aligns diffusion features with target conditions during the sampling process.
    """
    
    @register_to_config
    def __init__(
        self,
        time_channels: int = 256,
        in_channels: int = 7040,
        out_channels: int = 4,
        init_type: str = "xavier",
        init_gain: float = 0.02,
    ):
        super().__init__()
        
        self.time_channels = time_channels
        
        # Progressive feature reduction layers
        self.conv_1 = ConvBlock(time_channels, in_channels, in_channels // 2, 1, 1)
        self.conv_2 = ConvBlock(time_channels, in_channels // 2, in_channels // 4, 1, 1)
        self.conv_3 = ConvBlock(time_channels, in_channels // 4, 512, 3, 1, 1)
        self.conv_4 = ConvBlock(time_channels, 512, 256, 3, 1, 1)
        self.conv_5 = ConvBlock(time_channels, 256, 128, 3, 1, 1)
        self.conv_6 = ConvBlock(time_channels, 128, 64, 3, 1, 1)
        self.conv_7 = nn.Conv2d(64, out_channels, 1, 1)
        
        # Initialize weights
        self.init_weights(init_type, init_gain)
        
    def init_weights(self, init_type: str = 'xavier', gain: float = 0.02):
        """Initialize network weights"""
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
        
    def get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int,
        max_period: int = 10000,
        repeat_only: bool = False,
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: 1-D Tensor of N indices, one per batch element
            embedding_dim: dimension of the output
            max_period: controls the minimum frequency of the embeddings
            repeat_only: if True, repeat the timesteps instead of sinusoidal encoding
        """
        if not repeat_only:
            half = embedding_dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
            )
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if embedding_dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = timesteps[:, None].repeat(1, embedding_dim)
            
        return embedding
    
    def forward(
        self,
        features: torch.Tensor,
        timesteps: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass of the condition aligner.
        
        Args:
            features: Input features from diffusion model [B, C, H, W]
            timesteps: Timesteps [B]
            return_dict: Whether to return a dict
            
        Returns:
            Aligned condition prediction
        """
        # Generate timestep embeddings
        time_emb = self.get_timestep_embedding(timesteps, self.time_channels)
        
        # Progressive feature processing
        x = self.conv_1(features, time_emb)
        x = self.conv_2(x, time_emb)
        x = self.conv_3(x, time_emb)
        x = self.conv_4(x, time_emb)
        x = self.conv_5(x, time_emb)
        x = self.conv_6(x, time_emb)
        x = self.conv_7(x)
        
        if not return_dict:
            return (x,)
        
        return {"condition_pred": x}