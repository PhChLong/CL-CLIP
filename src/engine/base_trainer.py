import torch
from src.models import CLIPWrapper
from src.config import Config
from ..data import TaskData, TaskDataLoader, get_task_sequence
import os
import json
from datetime import datetime
from src.models import LoRAAdapter
from .metrics import compute_all_metrics

class Train:
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__()

        self.wrapper = wrapper #* sẽ là CLIPWrapper
        self.config = config
        optimizers = {
            'adamw': torch.optim.AdamW,
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD
        }
        name = self.config.train.name
        if name not in optimizers:
            raise ValueError(f"Unknown Optimizer: {name}")
        self.optimizer = optimizers[name]
        self.results = []
        #* lưu log training
        self.history = []
        #* lưu log dưới dạng text
        self.logs = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_lora(self):
        device = self.wrapper.model.device

        # ? Freeze model lại
        for param in self.wrapper.model.parameters():
            param.requires_grad = False

        #? thêm LoRA vào các layer q_proj, v_proj
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                #@ Vision
                attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r= self.config.train.r))

                #@ Text
                attn = self.wrapper.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r=self.config.train.r))
        self.wrapper.model.to(device)

    def train(self, task, task_id = None):
        raise NotImplementedError

    def compute_metrics(self):
        return compute_all_metrics(self.results) | {
            "results_matrix": self.results,
            "history": self.history
        }

   #@ Eval accuracy trên tất cả seen tasks, trả về list[float] indexed by task_id
    def eval_all(self) -> list:
        result = [0.0] * self.config.datasets.num_tasks
        self.wrapper.model.eval()
        device = self.wrapper.model.device

        with torch.inference_mode():
            for task_id, task in enumerate(self.tasks):
                data = TaskData(task, 'test', processor=self.wrapper.processor)
                dataloader = TaskDataLoader(data,
                                            batch_size=self.config.datasets.batch_size,
                                            num_workers=self.config.datasets.num_workers,
                                            pin_memory=True)
                text_tokenized = data.text_tokenized

                correct = total = 0
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)  #! images phải move to device
                    logits = self.wrapper.forward_logits(text_tokenized, images)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                result[task_id] = correct / total if total > 0 else 0.0

        return result
    
    def train_all_tasks(self):
        self.tasks = get_task_sequence()
        for task_id, task in enumerate(self.tasks):
            self.train(task, task_id=task_id)
            self.results.append(self.eval_all())
        
        metrics = self.compute_metrics()
        self.save_logs()
        self.save_results()

        return metrics

#@ ==============LOGS=======================
    def save_results(self):
        save_dir = f"results/{self.config.method}"
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"results_{self.run_id}.json")

        with open(path, "w") as f:
            json.dump(self.compute_metrics(), f, indent=4)

        print(f"Saved results to {path}")

    def save_logs(self):
        save_dir = f"results/{self.config.method}"
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"log_{self.run_id}.txt")

        with open(path, "w") as f:
            for line in self.logs:
                f.write(line + "\n")

        print(f"Saved logs to {path}")
