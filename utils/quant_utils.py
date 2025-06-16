import torch
import torch.nn as nn

class TrainableUniformQuantizer(nn.Module):
    def __init__(self, bit=8, symmetric=False, per_channel=True):
        super().__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.n_levels = 2 ** bit - 1
    def forward_weight(self, weight):
        if self.per_channel:
            out_channels = weight.shape[0]
            weight_reshape = weight.reshape(out_channels, -1)
            if self.symmetric:
                weight_abs = torch.abs(weight_reshape)
                weight_max = weight_abs.max(dim=1, keepdim=True)[0]
                interval = 2 * weight_max / self.n_levels
                weight_scaled = weight_reshape / interval
                weight_clamp = torch.clamp(weight_scaled, -self.n_levels // 2, self.n_levels // 2)
                weight_quant = torch.round(weight_clamp)
                weight_dequant = weight_quant * interval
                weight_dequant = weight_dequant.reshape(weight.shape)
            else:
                weight_min = weight_reshape.min(dim=1, keepdim=True)[0]
                weight_max = weight_reshape.max(dim=1, keepdim=True)[0]
                interval = (weight_max - weight_min) / self.n_levels
                weight_scaled = (weight_reshape - weight_min) / interval
                weight_clamp = torch.clamp(weight_scaled, 0, self.n_levels)
                weight_quant = torch.round(weight_clamp)
                weight_dequant = weight_quant * interval + weight_min
                weight_dequant = weight_dequant.reshape(weight.shape)
        else:
            if self.symmetric:
                weight_abs = torch.abs(weight)
                weight_max = weight_abs.max()
                interval = 2 * weight_max / self.n_levels
                weight_scaled = weight / interval
                weight_clamp = torch.clamp(weight_scaled, -self.n_levels // 2, self.n_levels // 2)
                weight_quant = torch.round(weight_clamp)
                weight_dequant = weight_quant * interval
            else:
                weight_min = weight.min()
                weight_max = weight.max()
                interval = (weight_max - weight_min) / self.n_levels
                weight_scaled = (weight - weight_min) / interval
                weight_clamp = torch.clamp(weight_scaled, 0, self.n_levels)
                weight_quant = torch.round(weight_clamp)
                weight_dequant = weight_quant * interval + weight_min
        weight_dequant = weight + (weight_dequant - weight).detach()
        return weight_dequant
    def forward_activation(self, x):
        if self.symmetric:
            x_abs = torch.abs(x)
            x_max = x_abs.max()
            interval = 2 * x_max / self.n_levels
            x_scaled = x / interval
            x_clamp = torch.clamp(x_scaled, -self.n_levels // 2, self.n_levels // 2)
            x_quant = torch.round(x_clamp)
            x_dequant = x_quant * interval
        else:
            x_min = x.min()
            x_max = x.max()
            interval = (x_max - x_min) / self.n_levels
            x_scaled = (x - x_min) / interval
            x_clamp = torch.clamp(x_scaled, 0, self.n_levels)
            x_quant = torch.round(x_clamp)
            x_dequant = x_quant * interval + x_min
        x_dequant = x + (x_dequant - x).detach()
        return x_dequant