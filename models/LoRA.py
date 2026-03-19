import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LoRA(nn.Module):
    def __init__(self, layer: nn.Linear, r = 4):
        super().__init__()
        in_feature, out_feature = layer.in_features, layer.out_features 
        self.A = nn.Parameter(torch.randn(in_feature, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_feature))
        self.scale = 1/r
    def extra_repr(self) -> str:
        in_feature, r = self.A.shape
        _, out_feature = self.B.shape
        return f"in={in_feature}, out={out_feature}, r={r}, scale={self.scale:.4f}"

    def forward(self, x):
        return x @ self.A @ self.B *self.scale
class LoRAAdapter(nn.Module):
    def __init__(self, original_layer: nn.Linear,lora_module: LoRA = None, r = 4):
        super().__init__()
        self.org_layer = original_layer
        self.lora = LoRA(original_layer) if lora_module is None else lora_module
    def forward(self, x): #x sẽ có shape là (_, in_feature)
        original_out = self.org_layer(x) #(_, out_feature)
        lora_out = self.lora(x)
        return original_out + lora_out

class LoRAExpert:
    def __init__(self):
        super().__init__()
        self.adapters = dict()