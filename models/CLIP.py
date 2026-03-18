from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def split_LoRA(self):
        lora_modules = {}
        device = self.model.device

        for i in range(12):
            attn = self.model.vision_model.encoder.layers[i].self_attn

            for layer_type in ["q_proj", "v_proj"]:
                module = getattr(attn, layer_type)

                if isinstance(module, LoRAAdapter):
                    # lưu adapter
                    lora_modules[f"vision.layers.{i}.{layer_type}"] = module

                    # tìm linear gốc để gắn lại vào model
                    original = None

                    #? attr_name này là tên của original_layer trong class LoRAAdapter
                    attr_name = "org_name"
                    if hasattr(module, attr_name):
                        original = getattr(module, attr_name)
                        break

                    if original is None:
                        raise AttributeError(
                            f"Không tìm thấy layer gốc bên trong LoRAAdapter ở layer {i} - {layer_type}"
                        )

                    setattr(attn, layer_type, original)

        self.model.to(device)
        return lora_modules
    def load_lora(self, lora_modules):
        device = self.model.device

        if not isinstance(lora_modules, dict):
            raise TypeError("lora_modules phải là một dict.")

        #? 12 ở đây là số block transformer trong model CLIP
        for i in range(12):
            attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn

            for layer_type in ["q_proj", "v_proj"]:
                key = f"vision.layers.{i}.{layer_type}"

                if key not in lora_modules:
                    continue

                module = lora_modules[key]

                if not isinstance(module, LoRAAdapter):
                    raise TypeError(f"{key} không phải là LoRAAdapter.")

                setattr(attn, layer_type, module)

        self.model.to(device)
        return self.model