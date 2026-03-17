from models import CLIPWrapper
import torch
from methods import FineTune
from config import Config
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPWrapper(device=device)
    config = Config('finetune')
    finetune = FineTune(clip, config)
    finetune.train_all_tasks()