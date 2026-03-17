from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
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