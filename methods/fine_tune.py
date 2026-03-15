from models import LoRA, CLIPWrapper
from methods.base_trainer import BaseTrainer
class FineTune(BaseTrainer):
    def __init__(self, wrapper:CLIPWrapper, config):
        super().__init__(wrapper, config)
        self.add_lora()
    def add_lora(self):
        for param in self.wrapper.model.parameters():
            param.requires_grad = False
        for i in range(12):
            for layer_type in ['q_proj', 'v_proj']:
                #* Vision
                attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRA(original))

                #*Text
                attn = self.wrapper.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRA(original))
    