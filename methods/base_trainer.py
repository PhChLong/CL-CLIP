import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CLIPWrapper
from config import Config
from data import TaskData, TaskDataLoader, get_task_sequence
from tqdm import tqdm
import os
import json
from datetime import datetime

class BaseTrainer:
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

    def AVG(self):
        avg = 0.0
        for i, row in enumerate(self.results):
            avg += sum(row[:i+1]) / (i+1)
        return avg / len(self.results)
    
    def Last(self):
        last_row = self.results[-1]
        seen = len(self.results)
        return sum(last_row[:seen]) / seen
    
    def BWT(self):
        T = len(self.results)
        if T <= 1:
            return 0.0
        last_row = self.results[-1]
        bwt = 0.0
        for i in range(T - 1):
            bwt += last_row[i] - self.results[i][i]
        return bwt / (T - 1)

    def Transfer(self):
        T = len(self.results)
        if T <= 1:
            return 0.0
        
        # Upper-right triangle: results[i][j] với j > i
        # results[i][j] = accuracy task j sau khi train xong task i
        # Với j > i: task j chưa được fine-tune → đây là zero-shot transfer
        
        col_avgs = []
        for j in range(1, T):          # mỗi task j (trừ task 0)
            col_vals = []
            for i in range(j):         # tất cả timestamps i < j
                col_vals.append(self.results[i][j])
            col_avgs.append(sum(col_vals) / len(col_vals))
        
        return sum(col_avgs) / len(col_avgs)

    def compute_metrics(self):
        return {
            "AVG": self.AVG(),
            "Last": self.Last(),
            "BWT": self.BWT(),
            "Transfer": self.Transfer(),
            "results_matrix": self.results,
            "history": self.history
        }
    
    def train(self, task, task_id = None):
        raise NotImplementedError
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
                text_features = self.wrapper.encode_text(data.text_tokenized)  #? encode 1 lần trước loop

                correct = total = 0
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)  #! images phải move to device
                    logits = self.wrapper.forward_with_text_features(text_features, images)
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