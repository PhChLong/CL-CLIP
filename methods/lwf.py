
from models import LoRAAdapter, CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
class LwF_LoRA(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.add_lora()
    
    def add_lora(self):
        device = self.wrapper.model.device
        # ? Freeze model lại
        for params in self.wrapper.parameters():
            params.requires_grad = False

        #? thêm LoRA vào các layer q_proj, v_proj trong model
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                org_attention_layer = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                org_layer = getattr(org_attention_layer, layer_type)
                setattr(org_attention_layer, layer_type, LoRAAdapter(org_layer, r = self.config.train.r))
                
        #? add model to GPU
        self.wrapper.model.to(device)
    