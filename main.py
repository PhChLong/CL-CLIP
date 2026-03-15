from models import CLIPWrapper
from data import get_task_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

tasks = get_task_sequence()
print(tasks)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPWrapper(device=device)
print('heelo')