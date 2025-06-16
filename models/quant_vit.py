import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_utils import TrainableUniformQuantizer

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        x_q = self.quantizer.forward_activation(x)
        weight_q = self.quantizer.forward_weight(self.weight)
        return F.linear(x_q, weight_q, self.bias)

class QuantizedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = QuantizedLinear(embed_dim, embed_dim * 3, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        self.proj = QuantizedLinear(embed_dim, embed_dim, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.quantizer.forward_activation(attn)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)
        return out

class QuantizedMLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = QuantizedLinear(embed_dim, hidden_dim, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        self.act = nn.GELU()
        self.fc2 = QuantizedLinear(hidden_dim, embed_dim, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.quantizer.forward_activation(x)
        x = self.fc2(x)
        return x

class QuantizedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = QuantizedMultiHeadAttention(embed_dim, num_heads, bit, symmetric, per_channel)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = QuantizedMLP(embed_dim, mlp_ratio, bit, symmetric, per_channel)
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm)
        x_attn = self.quantizer.forward_activation(x_attn)
        x = x + x_attn
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x_mlp = self.quantizer.forward_activation(x_mlp)
        x = x + x_mlp
        return x

class QuantizedVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_layers=12, num_heads=12, num_classes=1000, mlp_ratio=4, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.bit = bit
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            QuantizedTransformerBlock(embed_dim, num_heads, mlp_ratio, bit, symmetric, per_channel)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = QuantizedLinear(embed_dim, num_classes, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.quantizer.forward_activation(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x