from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from models.LoRA import LoRAAdapter

class CLIPWrapper(nn.Module):
    def __init__(
            self,
            checkpoint = 'openai/clip-vit-base-patch32',
            device = None
    ):
        super().__init__()
        
        self.model = CLIPModel.from_pretrained(checkpoint)
        self.processor = CLIPProcessor.from_pretrained(checkpoint)
        if device is not None:
            self.model.to(device)
    
        self.num_layers = len(self.model.text_model.encoder.layers)
    
    def encode_text(self, text):
        inputs = self.processor(
            text=text,
            images=None,
            return_tensors='pt',
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        text_features = self.model.get_text_features(**inputs).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images):
        pixel_values = images.to(self.model.device, non_blocking = True)
        image_features = self.model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_with_text_features(self, text_features, images):
        image_features = self.encode_image(images)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        return logits
    
    def split_and_get_lora(self):
        lora_modules = {}
        device = self.model.device
        for i in range(self.num_layers):
            #* cho text_model
            attn = self.model.text_model.encoder.layers[i].self_attn

            for layer_type in ['q_proj', 'v_proj']:
                module = getattr(attn, layer_type) #clip.model.text_model.encoder.layers[i].self_attn.q_proj

                if isinstance(module, LoRAAdapter):
                    #? lưu lại layer LoRA
                    lora_modules[f"text_model.encoder.layers.{i}.self_attn.{layer_type}"] = module.lora

                    original_layer = None

                    original_layer_name = 'org_layer'
                    if hasattr(module, original_layer_name):
                        original_layer = getattr(module, original_layer_name)
                    
                    if original_layer is None:
                        raise AttributeError(
                            f"Không tìm thấy LoRAAdapter"
                        )
                    setattr(attn, layer_type, original_layer)
            
            #* cho vision_model

            attn = self.model.vision_model.encoder.layers[i].self_attn

            for layer_type in ['q_proj', 'v_proj']:
                module = getattr(attn, layer_type)
                if isinstance(module, LoRAAdapter):
                    #? lưu lại layer LoRA
                    lora_modules[f"vision_model.encoder.layers.{i}.self_attn.{layer_type}"] = module.lora

                    original_layer = None

                    original_layer_name = 'org_layer'
                    if hasattr(module, original_layer_name):
                        original_layer = getattr(module, original_layer_name)
                    
                    if original_layer is None:
                        raise AttributeError(
                            f"Không tìm thấy LoRAAdapter"
                        )
                    setattr(attn, layer_type, original_layer)
            self.model.to(device)
        return lora_modules
    #? lora_modules sẽ trông ntn 
    #? 'vision_model.encoder.layers.10.self_attn.v_proj': LoRA(in=768, out=768, r=4, scale=0.2500)

    def load_lora(self, lora_modules):
        device = self.model.device
        for layer_name in lora_modules.keys():
            attn, layer_type = layer_name.rsplit('.', maxsplit=1)
            attn = self.model.get_submodule(attn)
            layer = self.model.get_submodule(layer_name)
            lora = LoRAAdapter(layer, lora_modules[layer_name])
            current = getattr(attn, layer_name)
            if isinstance(current, LoRAAdapter):
                raise TypeError(
                    f"Layer '{layer_name}' is already a LoRAAdapter. "
                    f"Expected original nn.Linear before loading LoRA."
                )
            setattr(attn, layer_type, lora)
        self.model.to(device)
