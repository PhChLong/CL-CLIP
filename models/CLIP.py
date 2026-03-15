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
    
    def forward(self, text, image):
        inputs = self.processor(text=text, images = image, return_tensors = 'pt', padding = True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs)