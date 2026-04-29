from src.models import CLIPWrapper
import torch
from src.methods import FineTune, LwF_LoRA
from src.config import Config
from src.engine import Train
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPWrapper(device=device)

    # finetune_config = Config('finetune')
    # finetune = FineTune()
    # trainer = Train(clip, finetune_config, finetune)
    # trainer.train_all_tasks(True)

    lwf_config = Config('lwf')
    lwf = LwF_LoRA()
    trainer = Train(clip, lwf_config, lwf)
    trainer.train_all_tasks(True)