import torch
from torch import nn
from ..models import CLIPWrapper
class ContinualLearningMethod:
    def __init__(self):
        super().__init__()
        self.criterion = None
        self.wrapper = None
        self.requires_task_id = False
        self.config = None
    
    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_wrapper(self, wrapper):
        self.wrapper: CLIPWrapper = wrapper
    
    def compute_loss_inference_mode(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError
    
    def initialize(self, task_id):
        if self.requires_task_id:
            raise NotImplementedError