import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LoRA(nn.Module):
    def __init__(self, original_layer: nn.Linear, r = 4):
        super().__init__()
        self.org_layer = original_layer
        in_feature, out_feature = original_layer.in_features, original_layer.out_features 
        self.A = nn.Parameter(torch.randn(in_feature, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_feature))
        self.scale = 1/r

    def forward(self, x): #x sẽ có shape là (_, in_feature)
        original_out = self.org_layer(x) #(_, out_feature)
        lora_out = x @ self.A @ self.B * self.scale
        return original_out + lora_out
        


