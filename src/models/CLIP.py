from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from src.models.LoRA import LoRAAdapter
from pathlib import Path

MODEL_CACHE_DIR = Path(__file__).parent / "model_cache"

class CLIPWrapper(nn.Module):
    def __init__(
            self,
            checkpoint = 'openai/clip-vit-base-patch32',
            device = None
    ):
        super().__init__()
        
        #@ load model if the model exists, else download the model
        if (MODEL_CACHE_DIR / "model.safetensors").exists() or (MODEL_CACHE_DIR / "pytorch_model.bin").exists():
            print(f"[cache] Loading CLIP from {MODEL_CACHE_DIR}")
            self.model = CLIPModel.from_pretrained(str(MODEL_CACHE_DIR))
            self.processor = CLIPProcessor.from_pretrained(str(MODEL_CACHE_DIR))
        else:
            print(f"[download] Downloading CLIP from {checkpoint}...")
            self.model = CLIPModel.from_pretrained(checkpoint)
            self.processor = CLIPProcessor.from_pretrained(checkpoint)
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(MODEL_CACHE_DIR))
            self.processor.save_pretrained(str(MODEL_CACHE_DIR))
            print(f"[saved] CLIP → {MODEL_CACHE_DIR}")

        if device is not None:
            self.model.to(device)
    
        self.num_layers = len(self.model.text_model.encoder.layers)
    
        self.num_layers = len(self.model.text_model.encoder.layers)
    #@ ======================sub model forward====================
    def forward_image_vision_model(self, image_tensors):
        pixel_values = image_tensors.to(self.model.device, non_blocking = True)
        outs = self.vision_model(pixel_values).pooler_output
        return outs


    def forward_text_text_model(self, prompts):
        inputs = self.processor(
            text=prompts,
            images=None,
            return_tensors='pt',
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        outs = self.text_model(**inputs, return_dict = True).pooler_output
        return outs


    #@=========================full model forward====================    
    def encode_text(self, text_tokenized):
        #? text_tokenized  = labels -> prompts -> processor()
        #? text_tokenized = {'input_ids': tensor, 'attention_mask': tensor}
        text_tokenized = {k: v.to(self.model.device) for k, v in text_tokenized.items() if k in ['input_ids', 'attention_mask']}
        text_features = self.model.get_text_features(**text_tokenized).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image_tensors):
        pixel_values = image_tensors.to(self.model.device, non_blocking = True)
        image_features = self.model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_with_text_features(self, text_features, image_tensors):
        image_features = self.encode_image(image_tensors)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        return logits
        
    def forward(self, text, images):
        inputs = self.processor(text=text, images = images, return_tensors = 'pt', padding = True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs)
    
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
            current = getattr(attn, layer_type)
            if isinstance(current, LoRAAdapter):
                raise TypeError(
                    f"Layer '{layer_name}' is already a LoRAAdapter. "
                    f"Expected original nn.Linear before loading LoRA."
                )
            setattr(attn, layer_type, lora)
        self.model.to(device)
