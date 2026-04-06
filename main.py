from models import CLIPWrapper
import torch
from methods import FineTune, LwF_LoRA
from config import Config
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPWrapper(device=device)
    # finetune_config = Config('finetune')
    # finetune = FineTune(clip, finetune_config)
    # finetune.train_all_tasks()

    lwf_config = Config('lwf')
    lwf = LwF_LoRA(clip, lwf_config)
    lwf.train_all_tasks()