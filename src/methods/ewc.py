from torch import nn 
import torch
import torch.nn.functional as F

from src.methods.base_trainer import BaseTrainer
from src.models import CLIPWrapper
from data import TaskData, get_task_sequence
from src.config import Config
class EWC(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)