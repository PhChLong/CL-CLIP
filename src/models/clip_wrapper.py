from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from src.models.lora import LoRAAdapter
from pathlib import Path
from src.config import Config

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
        
        #@ only use config to get number of layers of the model from the base config
        #@ does not matter which config we use
        self.config = Config('base')
        self.num_layers = len(self.model.text_model.encoder.layers)

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
    
    #? When inference, dont need to encode_text multiple times since the promps is small
    def forward_with_text_feature(self, text_features, image_tensors):
        image_features = self.encode_image(image_tensors)
        logit_scale = self.model.logit_scale.exp() #? to scale the logits to get better results
        logits = logit_scale * image_features @ text_features.T
        return logits

    def forward_logits(self, text_tokenized, image_tensors):
        image_features = self.encode_image(image_tensors)
        text_features = self.encode_text(text_tokenized)
        logit_scale = self.model.logit_scale.exp() #? to scale the logits to get better results
        logits = logit_scale * image_features @ text_features.T
        return logits
        
    def forward(self, text, images):
        inputs = self.processor(text=text, images = images, return_tensors = 'pt', padding = True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs)
    
    #@ ================LoRA related===========================
    def add_lora(self, r):
        device = self.model.device

        # ? Freeze model lại
        for param in self.model.parameters():
            param.requires_grad = False

        #? thêm LoRA vào các layer q_proj, v_proj
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                #@ Vision
                attn = self.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r= r))

                #@ Text
                attn = self.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r= r))
        self.model.to(device)

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
        if lora_modules is None:
            return
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
