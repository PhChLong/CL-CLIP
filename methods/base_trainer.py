import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CLIPWrapper
from config import Config
from data import TaskData, get_task_sequence
from tqdm import tqdm
import os
import json
from datetime import datetime

def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)
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
    
    def train(self, task, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))
        criterion = nn.CrossEntropyLoss()
        prompts = [f"a photo of a {name}" for name in task['label_names']]
        #* encode text truowcs
        text_features = self.wrapper.encode_text(prompts).detach()

        #* =============Data and Dataloader==============================
        train_data = TaskData(task, "train", image_processor= self.wrapper.processor.image_processor)
        test_data = TaskData(task, "test", image_processor= self.wrapper.processor.image_processor)
        train_loader = DataLoader(
            train_data,
            batch_size= self.config.datasets.batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.datasets.num_workers,
            pin_memory= True)
        test_loader = DataLoader(
            test_data,
            batch_size = self.config.datasets.batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.datasets.num_workers,
            pin_memory=True
        )

        device = self.wrapper.model.device 

        #* stop khi improvement nhor honw epsilon        
        epsilon = float(self.config.train.epsilon)

        prev_valid_loss = None
        best_valid_loss = float("inf")        

        for epoch in range(self.config.train.max_epoch):
            #* ==============TRAIN=============================
            self.wrapper.model.train()
            train_loss = valid_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = self._loss(images, text_features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            #* ===================Eval===========================
            self.wrapper.model.eval()
            with torch.inference_mode():
                for images, labels in tqdm(test_loader, desc=f"Valid Epoch {epoch+1}", leave=False):
                    labels = labels.to(device)
                    logits = self._loss(images, text_features)
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
            valid_loss /= len(test_loader)

            #* kiểm tra nếu model học
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            #* early stopping
            delta = None if prev_valid_loss is None else (- valid_loss + prev_valid_loss)
                
            
            #* =============lưu history============
            log = {
                "task_id": task_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss
            }

            self.history.append(log)
            message = f"task={task_id+1} epoch={epoch+1} || train_loss={train_loss:.4f} || valid_loss={valid_loss:.4f}"
            self.logs.append(message)
            print(message)

            if delta is not None and delta < epsilon:
                print(f"EARLY STOPPING")
                break

            prev_valid_loss = valid_loss
    def eval_all_seen(self, seen_tasks) -> list:
        result = [0.0] * self.config.datasets.num_tasks
        self.wrapper.model.eval()
        device = self.wrapper.model.device
        with torch.inference_mode():
            for task_id, seen_task in enumerate(seen_tasks):
                print(task_id)
                prompts = [f"a photo of a {name}" for name in seen_task['label_names']]
                text_features = self.wrapper.encode_text(prompts).detach()
                data = TaskData(seen_task, 'test', image_processor= self.wrapper.processor.image_processor)
                dataloader = DataLoader(data,
                                         batch_size = self.config.datasets.batch_size, 
                                         collate_fn=collate_fn, 
                                         num_workers=self.config.datasets.batch_size, 
                                         pin_memory=True)
                
                correct = total = 0

                for images, labels in dataloader:
                    labels = labels.to(device)
                    logits = self._loss(images, text_features)
                    preds = logits.argmax(dim = -1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                acc = correct / total if total >0 else 0.0
                result[task_id] = acc
        return result
    def compute_metrics(self):
        return {
            "AVG": self.AVG(),
            "Last": self.Last(),
            "BWT": self.BWT(),
            "results_matrix": self.results,
            "history": self.history
        }
    def train_all_tasks(self):
        tasks = get_task_sequence()
        seen_tasks = []
        for task_id, task in enumerate(tasks):
            self.train(task, task_id=task_id)
            seen_tasks.append(task)
            self.results.append(self.eval_all_seen(seen_tasks))
        
        metrics = self.compute_metrics()
        self.save_logs()
        self.save_results()

        return metrics

    def _loss(self, images, text_features):
        logits = self.wrapper.forward_with_text_features(text_features, images)
        return logits
    
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