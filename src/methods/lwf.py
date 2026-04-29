from src.models import CLIPWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from copy import deepcopy
from .cl_methods import ContinualLearningMethod
from ..config import Config

config = Config('lwf')
class LwF_LoRA(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
        self.requires_task_id = True
        self.trained_task_id = []
        self.T = config.train.distill_temp
        self.lambda_old = config.train.lambda_old

    def initialize(self, task_id):
        if task_id in self.trained_task_id:
            return
        self.trained_task_id.append(task_id)

        old_LoRA = self.wrapper.split_and_get_lora()
        self.old_LoRA = deepcopy(old_LoRA)
        self.wrapper.load_lora(old_LoRA)
        
    def compute_loss_inference_mode(self, images, labels, text_features):
        logits = self.wrapper.forward_with_text_feature(text_features, images)
        loss_ce = self.criterion(logits, labels)
        return loss_ce

    def compute_loss(self, images, labels, text_tokenized):
        #@loss_kd
        current_LoRA = self.wrapper.split_and_get_lora()
        if len(self.trained_task_id) != 1: #? which means this is not the first task
            self.wrapper.load_lora(self.old_LoRA)

        with torch.inference_mode():
            old_logits = self.wrapper.forward_logits(text_tokenized, images)
        old_logits_prime:torch.Tensor = F.log_softmax(old_logits/self.T, dim = 1)

        self.wrapper.split_and_get_lora()
        self.wrapper.load_lora(current_LoRA)
        logits: torch.Tensor = self.wrapper.forward_logits(text_tokenized, images)
        
        logits_prime:torch.Tensor = F.log_softmax(logits / self.T, dim = 1)

        loss_kd = F.kl_div(logits_prime, old_logits_prime, reduction= "batchmean", log_target= True) * (self.T * self.T)

        #@ loss_ce
        loss_ce = self.criterion(logits, labels)

        loss = loss_ce + loss_kd * self.lambda_old
        return loss