from models.vit import VisionTransformer
from models.quant_vit import QuantizedVisionTransformer
from utils.quant_utils import TrainableUniformQuantizer
from data.imagenet_loader import get_imagenet_loaders
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value vectors
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        out = self.proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_layers=12, num_heads=12, num_classes=1000, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Add class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use only the class token
        x = self.head(x)
        
        return x




class TrainableUniformQuantizer(nn.Module):
    def __init__(self, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.n_levels = 2 ** bit - 1
        
    def forward_weight(self, weight):
        if self.per_channel:
            # Per-channel quantization
            out_channels = weight.shape[0]
            weight_reshape = weight.reshape(out_channels, -1)
            
            if self.symmetric:
                # Symmetric quantization
                weight_abs = torch.abs(weight_reshape)
                weight_max = weight_abs.max(dim=1, keepdim=True)[0]
                interval = 2 * weight_max / self.n_levels
                
                # Scale and clamp
                weight_scaled = weight_reshape / interval
                weight_clamp = torch.clamp(weight_scaled, -self.n_levels // 2, self.n_levels // 2)
                weight_quant = torch.round(weight_clamp)
                
                # Dequantize
                weight_dequant = weight_quant * interval
                
                # Reshape back
                weight_dequant = weight_dequant.reshape(weight.shape)
            else:
                # Asymmetric quantization
                weight_min = weight_reshape.min(dim=1, keepdim=True)[0]
                weight_max = weight_reshape.max(dim=1, keepdim=True)[0]
                interval = (weight_max - weight_min) / self.n_levels
                
                # Scale and clamp
                weight_scaled = (weight_reshape - weight_min) / interval
                weight_clamp = torch.clamp(weight_scaled, 0, self.n_levels)
                weight_quant = torch.round(weight_clamp)
                
                # Dequantize
                weight_dequant = weight_quant * interval + weight_min
                
                # Reshape back
                weight_dequant = weight_dequant.reshape(weight.shape)
        else:
            # Per-tensor quantization
            if self.symmetric:
                # Symmetric quantization
                weight_abs = torch.abs(weight)
                weight_max = weight_abs.max()
                interval = 2 * weight_max / self.n_levels
                
                # Scale and clamp
                weight_scaled = weight / interval
                weight_clamp = torch.clamp(weight_scaled, -self.n_levels // 2, self.n_levels // 2)
                weight_quant = torch.round(weight_clamp)
                
                # Dequantize
                weight_dequant = weight_quant * interval
            else:
                # Asymmetric quantization
                weight_min = weight.min()
                weight_max = weight.max()
                interval = (weight_max - weight_min) / self.n_levels
                
                # Scale and clamp
                weight_scaled = (weight - weight_min) / interval
                weight_clamp = torch.clamp(weight_scaled, 0, self.n_levels)
                weight_quant = torch.round(weight_clamp)
                
                # Dequantize
                weight_dequant = weight_quant * interval + weight_min
        
        # Straight-through estimator (STE)
        weight_dequant = weight + (weight_dequant - weight).detach()
        
        return weight_dequant
    
    def forward_activation(self, x):
        if self.symmetric:
            # Symmetric quantization
            x_abs = torch.abs(x)
            x_max = x_abs.max()
            interval = 2 * x_max / self.n_levels
            
            # Scale and clamp
            x_scaled = x / interval
            x_clamp = torch.clamp(x_scaled, -self.n_levels // 2, self.n_levels // 2)
            x_quant = torch.round(x_clamp)
            
            # Dequantize
            x_dequant = x_quant * interval
        else:
            # Asymmetric quantization
            x_min = x.min()
            x_max = x.max()
            interval = (x_max - x_min) / self.n_levels
            
            # Scale and clamp
            x_scaled = (x - x_min) / interval
            x_clamp = torch.clamp(x_scaled, 0, self.n_levels)
            x_quant = torch.round(x_clamp)
            
            # Dequantize
            x_dequant = x_quant * interval + x_min
        
        # Straight-through estimator (STE)
        x_dequant = x + (x_dequant - x).detach()
        
        return x_dequant







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
        # Quantize input activation
        x_q = self.quantizer.forward_activation(x)
        
        # Quantize weight
        weight_q = self.quantizer.forward_weight(self.weight)
        
        # Linear operation
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
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Quantize attention weights
        attn = self.quantizer.forward_activation(attn)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value vectors
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
        # Attention block
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm)
        x_attn = self.quantizer.forward_activation(x_attn)
        x = x + x_attn
        
        # MLP block
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
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2
        
        # Quantizer
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            QuantizedTransformerBlock(embed_dim, num_heads, mlp_ratio, bit, symmetric, per_channel)
            for _ in range(num_layers)
        ])
        
        # Normalization and classification
        self.norm = nn.LayerNorm(embed_dim)
        self.head = QuantizedLinear(embed_dim, num_classes, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = self.quantizer.forward_activation(x)
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use only the class token
        x = self.head(x)
        
        return x




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
        # Quantize input activation
        x_q = self.quantizer.forward_activation(x)
        
        # Quantize weight
        weight_q = self.quantizer.forward_weight(self.weight)
        
        # Linear operation
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
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Quantize attention weights
        attn = self.quantizer.forward_activation(attn)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value vectors
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
        # Attention block
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm)
        x_attn = self.quantizer.forward_activation(x_attn)
        x = x + x_attn
        
        # MLP block
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
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2
        
        # Quantizer
        self.quantizer = TrainableUniformQuantizer(bit, symmetric, per_channel)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            QuantizedTransformerBlock(embed_dim, num_heads, mlp_ratio, bit, symmetric, per_channel)
            for _ in range(num_layers)
        ])
        
        # Normalization and classification
        self.norm = nn.LayerNorm(embed_dim)
        self.head = QuantizedLinear(embed_dim, num_classes, bias=True, bit=bit, symmetric=symmetric, per_channel=per_channel)
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = self.quantizer.forward_activation(x)
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use only the class token
        x = self.head(x)
        
        return x






class QKD:
    def __init__(self, teacher_model, student_model, train_loader, val_loader, device):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def self_studying_phase(self, epochs, learning_rate, weight_decay=0.0):
        """
        Phase 1: Self-studying - Train the student model without KD
        """
        print("===== Phase 1: Self-studying =====")
        
        # Optimizer
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.student_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.student_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss /= len(self.train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation
            val_loss, val_acc = self._validate(self.student_model, criterion)
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.student_model.state_dict()
        
        # Load the best model
        self.student_model.load_state_dict(best_model_state)
        return best_model_state
    
    def co_studying_phase(self, epochs, learning_rate, temperature=2.0, weight_decay=0.0):
        """
        Phase 2: Co-studying - Train both teacher and student models
        """
        print("===== Phase 2: Co-studying =====")
        
        # Optimizers
        teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.teacher_model.train()
            self.student_model.train()
            
            train_loss_teacher = 0.0
            train_loss_student = 0.0
            correct_teacher = 0
            correct_student = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass for teacher
                teacher_outputs = self.teacher_model(inputs)
                teacher_loss = criterion(teacher_outputs, targets)
                
                # Forward pass for student
                student_outputs = self.student_model(inputs)
                
                # KL divergence from student to teacher
                kl_student_to_teacher = F.kl_div(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # KL divergence from teacher to student
                kl_teacher_to_student = F.kl_div(
                    F.log_softmax(teacher_outputs / temperature, dim=1),
                    F.softmax(student_outputs / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # Combined loss for teacher
                teacher_combined_loss = teacher_loss + kl_teacher_to_student
                
                # Combined loss for student
                student_ce_loss = criterion(student_outputs, targets)
                student_combined_loss = student_ce_loss + kl_student_to_teacher
                
                # Update teacher
                teacher_optimizer.zero_grad()
                teacher_combined_loss.backward(retain_graph=True)
                teacher_optimizer.step()
                
                # Update student
                student_optimizer.zero_grad()
                student_combined_loss.backward()
                student_optimizer.step()
                
                train_loss_teacher += teacher_combined_loss.item()
                train_loss_student += student_combined_loss.item()
                
                _, predicted_teacher = teacher_outputs.max(1)
                _, predicted_student = student_outputs.max(1)
                total += targets.size(0)
                correct_teacher += predicted_teacher.eq(targets).sum().item()
                correct_student += predicted_student.eq(targets).sum().item()
            
            train_loss_teacher /= len(self.train_loader)
            train_loss_student /= len(self.train_loader)
            train_acc_teacher = 100.0 * correct_teacher / total
            train_acc_student = 100.0 * correct_student / total
            
            # Validation
            val_loss_teacher, val_acc_teacher = self._validate(self.teacher_model, criterion)
            val_loss_student, val_acc_student = self._validate(self.student_model, criterion)
            
            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"Teacher - Train Loss: {train_loss_teacher:.4f}, Train Acc: {train_acc_teacher:.2f}%, Val Loss: {val_loss_teacher:.4f}, Val Acc: {val_acc_teacher:.2f}%")
            print(f"Student - Train Loss: {train_loss_student:.4f}, Train Acc: {train_acc_student:.2f}%, Val Loss: {val_loss_student:.4f}, Val Acc: {val_acc_student:.2f}%")
            
            # Save the best model
            if val_acc_student > best_val_acc:
                best_val_acc = val_acc_student
                best_model_state = self.student_model.state_dict()
        
        # Load the best model
        self.student_model.load_state_dict(best_model_state)
        return best_model_state
    
    def tutoring_phase(self, epochs, learning_rate, temperature=2.0, alpha=0.5, weight_decay=0.0):
        """
        Phase 3: Tutoring - Freeze the teacher and train only the student
        """
        print("===== Phase 3: Tutoring =====")
        
        # Freeze the teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.teacher_model.eval()
            self.student_model.train()
            
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                
                student_outputs = self.student_model(inputs)
                
                # Cross-entropy loss
                ce_loss = criterion(student_outputs, targets)
                
                # KD loss
                kd_loss = F.kl_div(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kd_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = student_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss /= len(self.train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation
            val_loss, val_acc = self._validate(self.student_model, criterion)
            
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.student_model.state_dict()
        
        # Load the best model
        self.student_model.load_state_dict(best_model_state)
        return best_model_state
    
    def _validate(self, model, criterion):
        """
        Validate the model
        """
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc





def main():
    # Hyperparameters
    img_size = 224
    patch_size = 16
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    num_classes = 1000
    mlp_ratio = 4
    bit = 4  # 4-bit quantization
    symmetric = True
    per_channel = True
    
    # KD parameters
    temperature = 2.0
    alpha = 0.5
    
    # Training parameters
    ss_epochs = 50  # Self-studying epochs
    cs_epochs = 100  # Co-studying epochs
    tu_epochs = 50  # Tutoring epochs
    learning_rate = 3e-4
    weight_decay = 0.0
    batch_size = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #"E:\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train2"
    # Data loaders
    train_dataset = torchvision.datasets.ImageNet(
        root='E:/ILSVRC2012_devkit_t12.tar.gz',
        split='train',
        transform=transform_train,
    )
    
    val_dataset = torchvision.datasets.ImageNet(
        root='E:/ILSVRC2012_devkit_t12.tar.gz',
        split='val',
        transform=transform_val,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create models
    # Teacher model (full-precision)
    teacher_model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        mlp_ratio=mlp_ratio,
    ).to(device)
    
    # Student model (quantized)
    student_model = QuantizedVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        mlp_ratio=mlp_ratio,
        bit=bit,
        symmetric=symmetric,
        per_channel=per_channel,
    ).to(device)
    
    # Load pre-trained weights if available
    # teacher_model.load_state_dict(torch.load('pretrained_vit.pth'))
    
    # QKD training
    qkd = QKD(teacher_model, student_model, train_loader, val_loader, device)
    
    # Phase 1: Self-studying
    best_student_state = qkd.self_studying_phase(
        epochs=ss_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Phase 2: Co-studying
    best_student_state = qkd.co_studying_phase(
        epochs=cs_epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        weight_decay=weight_decay,
    )
    
    # Phase 3: Tutoring
    best_student_state = qkd.tutoring_phase(
        epochs=tu_epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        alpha=alpha,
        weight_decay=weight_decay,
    )
    
    # Save the final model
    torch.save(best_student_state, 'quantized_vit.pth')
    print("Training complete!")

if __name__ == '__main__':
    main()






