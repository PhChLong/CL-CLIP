from torch import nn 
import torch
import torch.nn.functional as F

from base_trainer import BaseTrainer
from models import CLIPWrapper
from data import TaskData, get_task_sequence
from config import Config
class EWC(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)