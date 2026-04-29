from src.models import CLIPWrapper
import torch
import torch.nn as nn
from .cl_methods import ContinualLearningMethod
from ..models import CLIPWrapper
from ..config import Config

class FineTune(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
   
    #@ đặc biệt dùng khi eval, vì không cần encode text_tokenized nhiều lần
    def compute_loss_inference_mode(self, images, labels, text_features):
        logits = self.wrapper.forward_with_text_feature(text_features, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce

    def compute_loss(self, images, labels, text_tokenized):
        logits = self.wrapper.forward_logits(text_tokenized, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce