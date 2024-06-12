import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)            # version
        self.transformer = CLIPTextModel.from_pretrained(version)          # version
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
    
# TODO: Old version of sketch-branch encoder (DISCARDED)
class StructureEncoder(nn.Module):
    def __init__(self, context_dim):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=context_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(context_dim)

        
    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)
        
        x = self.conv5(x)
        x = self.bn5(x.to(torch.float32))
        x = self.act(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        return x
                    
# TODO: Modified sketch-branch encoder according to the one in PITI
class ModifiedPITISketchEncoder(nn.Module):
    def __init__(self, context_dim, in_channels=3):
        super(ModifiedPITISketchEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.LReLU2 = nn.LeakyReLU(0.2)
 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.LReLU3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        self.LReLU4 = nn.LeakyReLU(0.2)
  
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU5 = nn.LeakyReLU(0.2)
    
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=context_dim // 2, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        x = self.LReLU1(self.norm1(self.conv1(x)))
        x = self.LReLU2(self.norm2(self.conv2(x)))
        x = self.LReLU3(self.norm3(self.conv3(x)))
        x = self.LReLU4(self.norm4(self.conv4(x)))
        x = self.LReLU5(self.norm5(self.conv5(x)))
        x = self.conv6(x)
        
        x = rearrange(x, 'b c h w -> b c (h w)')        # [b, c=384, (h w)=1024]
        
        return x     

# TODO: DEEPER PITI-based encoder
# TODO: Since reshaped clip feature is harmful, we convolute the sketch feature into smaller spatial resolution \
    # // Plus, eventually using convolution to linearly transform the feature channel size instead of linear layer \
    # TODO: from [b, c=768, (h w)=16*16=256] -> [b, c=768, n=77]
class DeeperModifiedPITISketchEncoder(nn.Module):
    def __init__(self, context_dim, in_channels=3):
        super(DeeperModifiedPITISketchEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.LReLU1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.LReLU2 = nn.LeakyReLU(0.2)
 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        self.LReLU3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        self.LReLU4 = nn.LeakyReLU(0.2)
  
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU5 = nn.LeakyReLU(0.2)
        
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.norm6 = nn.InstanceNorm2d(512, affine=True)
        self.LReLU6 = nn.LeakyReLU(0.2)
    
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=context_dim // 2, kernel_size=3, stride=1, padding=1)
        
        self.linear = nn.Linear(in_features=256, out_features=77)
 
    def forward(self, x):
        x = self.LReLU1(self.norm1(self.conv1(x)))
        x = self.LReLU2(self.norm2(self.conv2(x)))
        x = self.LReLU3(self.norm3(self.conv3(x)))
        x = self.LReLU4(self.norm4(self.conv4(x)))
        x = self.LReLU5(self.norm5(self.conv5(x)))
        x = self.LReLU6(self.norm6(self.conv6(x)))
        x = self.conv7(x)
        
        x = rearrange(x, 'b c h w -> b c (h w)')        # [b, c=768, (h w)==16*16=256]
        
        x = self.linear(x)                              # [b, c=768, n=256] -> [b, c=768, n=77]
        x = rearrange(x, 'b c n -> b n c')
        
        return x  

# TODO: Merge-modality encoder via concatenation (DISCARDED)
class MergeModalityEncoder(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim)
        self.text_linear1 = nn.Linear(in_features=context_dim, out_features=context_dim // 2)
        self.text_linear2 = nn.Linear(in_features=77, out_features=1024)
                
    def forward(self, sketch, text):        
        # TODO: classifier-free guidance
        if sketch == [''] or text == ['']:
            text_feat = self.text_encoder(text)
            text_feat = rearrange(text_feat, 'b n c -> b c n')         # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
            text_feat = rearrange(text_feat, 'b c n -> b n c')
            
            return text_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)
            text_feat = self.text_encoder(text)
            
            # align text feature with sketch feautre
            text_feat = self.text_linear1(text_feat)                        # [b, n=77, c=768] -> [b, n=77, c=384]
            text_feat = rearrange(text_feat, 'b n c -> b c n')         # [b, n=77, c=384] -> [b, c=384, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=384, n=77] -> [b, c=384, n=1024]
            
            return_feat = torch.cat([sketch_feat, text_feat], dim=1)   # [b, c=384, n=1024] + [b, c=384, n=1024] = [b, c=768, n=1024]
            return_feat = rearrange(return_feat, 'b c n -> b n c')
            
            return return_feat

# TODO: Merge-modality encoder with deeper sketch encoder (without reshaping clip feature to see if there is any deterioration)
# * Changelog: Reshaping clip feature does harm to feature utilization. Therefore DISCARD the reshaping operation and modify the sketch-branch encoder.
class MergeModalityEncoderWithDeeperSketchEncoder(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = DeeperModifiedPITISketchEncoder(context_dim * 2)                             # multiply 2 to make sure the features are addable
                
    def forward(self, sketch, text):
        # TODO: Unconditional conditioning
        if sketch == [''] or text == ['']:
            text_feat = self.text_encoder(text)
            
            return text_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)                       # [b, n=77, c=768]
            text_feat = self.text_encoder(text)                             # [b, n=77, c=768]
            
            return_feat = sketch_feat + text_feat                           # simply adding
            
            return return_feat


# TODO: Explore if reshaping clip feature from [b, n=77, c=768] -> [b, n=1024, c=768] does harm to the feature utilization
# * Conclusion: Reshaping clip feature DOES HARM to the feature utilization
class MergeModalityEncoderWithoutSketchBranch(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)                             # multiply 2 to make sure the features are addable
        self.text_linear1 = nn.Linear(in_features=context_dim, out_features=context_dim // 2)
        self.text_linear2 = nn.Linear(in_features=77, out_features=1024)
                
    def forward(self, sketch, text):
        
        text_feat = self.text_encoder(text)                             # [b, n=77, c=768]
        
        # align text feature with sketch feautre
        text_feat = rearrange(text_feat, 'b n c -> b c n')              # [b, n=77, c=768] -> [b, c=768, n=77]
        text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
        
        return_feat = text_feat 
        return_feat = rearrange(return_feat, 'b c n -> b n c')
        
        return return_feat

# TODO: Check if there is any bug that keep setting alpha as constant value
# * Conclusion: Since gradients cannot backpropogate to the condition encoder, alpha would not be updated during training and keep the value of 0.2
class MergeModalityEncoderWithRegularizationWithoutBounds(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)                             # multiply 2 to make sure the features are addable
        self.text_linear1 = nn.Linear(in_features=context_dim, out_features=context_dim // 2)
        self.text_linear2 = nn.Linear(in_features=77, out_features=1024)
        self.alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32), requires_grad=True)          # learnable alpha for regularization
                
    def forward(self, sketch, text):
        print(self.alpha)
        
        sketch_feat = self.sketch_encoder(sketch)                       # [b, c=768, n=1024]
        text_feat = self.text_encoder(text)                             # [b, n=77, c=768]
        
        # align text feature with sketch feautre
        text_feat = rearrange(text_feat, 'b n c -> b c n')              # [b, n=77, c=768] -> [b, c=768, n=77]
        text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
        
        return_feat = self.alpha * sketch_feat + (1 - self.alpha) * text_feat 
        return_feat = rearrange(return_feat, 'b c n -> b n c')
        
        return return_feat

# TODO: Most effective currently via simple adding features
class MergeModalityEncoderSimpleAdding(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)      # multiply 2 to make sure the features are addable
        self.text_linear = nn.Linear(in_features=77, out_features=1024)
                
    def forward(self, sketch, text):
        # TODO: Classifier-free guidance
        if text == ['']:
            sketch = torch.zeros([1, 3, 512, 512]).cuda()
            sketch_feat = self.sketch_encoder(sketch)
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
            
            text_feat = self.text_encoder(text)
            text_feat = rearrange(text_feat, 'b n c -> b c n')                # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear(text_feat)                           # [b, c=768, n=77] -> [b, c=768, n=1024]
            text_feat = rearrange(text_feat, 'b c n -> b n c')
            
            return_feat = sketch_feat + text_feat
            
            return return_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)                         # [b, c=768, n=1024]
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
            text_feat = self.text_encoder(text)                               # [b, n=77, c=768]
            
            # align text feature with sketch feautre
            text_feat = rearrange(text_feat, 'b n c -> b c n')                # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear(text_feat)                           # [b, c=768, n=77] -> [b, c=768, n=1024]
            text_feat = rearrange(text_feat, 'b c n -> b n c')
            
            return_feat = sketch_feat + text_feat
            
            return return_feat


# TODO: To see what exactly could sketch-branch learn
class MergeModalityEncoderWithoutCLIP(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)      # multiply 2 to make sure the features are addable
                
    def forward(self, sketch, text):
        # TODO: Classifier-free guidance
        if text == ['']:
            sketch = torch.zeros([1, 3, 512, 512]).cuda()
            sketch_feat = self.sketch_encoder(sketch)
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
            
            return_feat = sketch_feat
            
            return return_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)                         # [b, c=768, n=1024]
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
            
            return_feat = sketch_feat
            
            return return_feat


class MergeModalityEncoderWithRegularization(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)                             # multiply 2 to make sure the features are addable
        self.text_linear1 = nn.Linear(in_features=context_dim, out_features=context_dim // 2)
        self.text_linear2 = nn.Linear(in_features=77, out_features=1024)
        self.alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32), requires_grad=True)          # learnable alpha for regularization
                
    def forward(self, sketch, text):
        # TODO: Classifier-free guidance
        if sketch == [''] or text == ['']:
            sketch = torch.zeros([1, 3, 512, 512]).cuda()
            sketch_feat = self.sketch_encoder(sketch)
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
            
            text_feat = self.text_encoder(text)
            text_feat = rearrange(text_feat, 'b n c -> b c n')         # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
            text_feat = rearrange(text_feat, 'b c n -> b n c')
            
            return_feat = sketch_feat + text_feat
            
            return return_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)                       # [b, c=768, n=1024]
            text_feat = self.text_encoder(text)                             # [b, n=77, c=768]
            
            # align text feature with sketch feautre
            text_feat = rearrange(text_feat, 'b n c -> b c n')              # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
            
            # set upper bound and lower bound for regularization parameter alpha
            if self.alpha < 0.2:
                self.alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32), requires_grad=True)
            elif self.alpha > 0.8:
                self.alpha = nn.Parameter(torch.tensor(0.8, dtype=torch.float32), requires_grad=True)
            
            return_feat = self.alpha * sketch_feat + (1 - self.alpha) * text_feat 
            return_feat = rearrange(return_feat, 'b c n -> b n c')
            
            return return_feat


# TODO: concat on ``L'' dimension
class MergeModalityEncoderConcatOnLDim(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.text_encoder = FrozenCLIPEmbedder()
        self.sketch_encoder = ModifiedPITISketchEncoder(context_dim * 2)                             # multiply 2 to make sure the features are addable
        self.text_linear2 = nn.Linear(in_features=77, out_features=1024)
                
    def forward(self, sketch, text):
        # TODO: Unconditional conditioning
        # TODO: When you are writing the training code, make sure the code could be implemented with classifier-free guidance
        if sketch == [''] or text == ['']:      
            sketch = torch.zeros([1, 3, 512, 512]).cuda()
            sketch_feat = self.sketch_encoder(sketch)
            sketch_feat = rearrange(sketch_feat, 'b c n -> b n c')
                  
            text_feat = self.text_encoder(text)
            text_feat = rearrange(text_feat, 'b n c -> b c n')         # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
            text_feat = rearrange(text_feat, 'b c n -> b n c')
            
            return_feat = torch.cat([sketch_feat, text_feat], dim=1)
            
            
            return return_feat
        else:
            sketch_feat = self.sketch_encoder(sketch)                       # [b, c=768, n=1024]
            text_feat = self.text_encoder(text)                             # [b, n=77, c=768]
            
            # align text feature with sketch feautre
            text_feat = rearrange(text_feat, 'b n c -> b c n')              # [b, n=77, c=768] -> [b, c=768, n=77]
            text_feat = self.text_linear2(text_feat)                        # [b, c=768, n=77] -> [b, c=768, n=1024]
            
            return_feat = torch.cat([sketch_feat, text_feat], dim=2)        # [b, c=768, l=2048]
            return_feat = rearrange(return_feat, 'b c n -> b n c')
            
            return return_feat


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)