from src.models import CLIPWrapper
import torch
import torch.nn as nn
from .cl_methods import ContinualLearningMethod
from ..models import CLIPWrapper
from ..config import Config

class FineTune(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
   
    def compute_loss(self, images, labels, text_tokenized):
        logits = self.wrapper.forward_logits(text_tokenized, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce