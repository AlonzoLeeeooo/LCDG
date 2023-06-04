import torch
import torch.nn as nn
import math
from einops import repeat

class ConvBlock(nn.Module):
    def __init__(self, time_ch, in_ch, out_ch, k, s, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )
        self.lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, out_ch),
        )
        
    def forward(self, x, t):
        t_emb = self.lin(t)
        x = self.conv(x)
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]
        out = x + t_emb
        
        return out

# TODO: Base model of Condition Adaptor
class ConditionAdaptor(nn.Module):
    def __init__(self, time_channels, in_channels, out_channels):
        super(ConditionAdaptor, self).__init__()
        self.time_channels = time_channels
        
        print(f"\nInitializing condition adaptor v1.0...\n")
        self.conv_1 = ConvBlock(time_channels, in_channels, in_channels // 2, 1, 1)
        self.conv_2 = ConvBlock(time_channels, in_channels // 2, in_channels // 4, 1, 1)
        self.conv_3 = ConvBlock(time_channels, in_channels // 4, 512, 3, 1, 1)
        self.conv_4 = ConvBlock(time_channels, 512, 256, 3, 1, 1)
        self.conv_5 = ConvBlock(time_channels, 256, 128, 3, 1, 1)
        self.conv_6 = ConvBlock(time_channels, 128, 64, 3, 1, 1)
        self.conv_7 = nn.Conv2d(64, out_channels, 1, 1)
        
    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

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

    def timestep_embedding(self, timesteps, dim, max_period=10, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=timesteps.device)
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, 'b -> b d', d=dim)
        return embedding
    
    def forward(self, x, t):
        t_emb = self.timestep_embedding(timesteps=t, dim=self.time_channels, repeat_only=False)
        x = self.conv_1(x, t_emb)
        x = self.conv_2(x, t_emb)
        x = self.conv_3(x, t_emb)
        x = self.conv_4(x, t_emb)
        x = self.conv_5(x, t_emb)
        x = self.conv_6(x, t_emb)
        x = self.conv_7(x)
        return x
