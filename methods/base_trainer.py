import torch
import torch.nn as nn

class BaseTrainer:
    def __init__(self, wrapper, config):
        super().__init__()
        self.wrapper = wrapper #* sẽ là CLIPWrapper
        self.config = config
    def train(self):
        raise NotImplementedError
    