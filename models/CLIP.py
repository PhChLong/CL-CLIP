from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPWrapper(nn.Module):
    def __init__(
            self,
            checkpoint = 'openai/clip-vit-base-batch32',
            device = None
    ):
        self.model = CLIPModel.from_pretrained(checkpoint)
        self.processor = CLIPProcessor.from_pretrained(checkpoint)
        if device is not None:
            self.model.to(device)
            self.processor.to(device)