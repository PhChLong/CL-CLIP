from src.models import CLIPWrapper
import torch
from src.methods import FineTune, LwF_LoRA
from src.config import Config
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPWrapper(device=device)

    clip.split_and_get_lora()
    finetune_config = Config('finetune')
    finetune = FineTune(clip, finetune_config)
    finetune.train_all_tasks()

    clip.split_and_get_lora()
    lwf_config = Config('lwf')
    lwf = LwF_LoRA(clip, lwf_config)
    lwf.train_all_tasks()