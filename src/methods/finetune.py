from src.models import CLIPWrapper
import torch
import torch.nn as nn
from .cl_methods import ContinualLearningMethod
from ..models import CLIPWrapper

class FineTune:
    def __init__(self):
        super().__init__()
        self.criterion = None
        self.wrapper = None
    
    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_wrapper(self, wrapper):
        self.wrapper: CLIPWrapper = wrapper

    #@ đặc biệt dùng khi eval, vì không cần encode text_tokenized nhiều lần
    def compute_loss_inference_mode(self, images, labels, text_features):
        logits = self.wrapper.forward_with_text_feature(text_features, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce

    def compute_loss(self, images, labels, text_tokenized = None):
        logits = self.wrapper.forward_logits(text_tokenized, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce