from models import LoRA, CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
class FineTune(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.add_lora()
    def add_lora(self):
        device = self.wrapper.model.device
        for param in self.wrapper.model.parameters():
            param.requires_grad = False
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                #* Vision
                attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRA(original))

                #*Text
                attn = self.wrapper.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRA(original))
        self.wrapper.model.to(device)