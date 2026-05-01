import torch
from torch import nn
from src.models import CLIPWrapper
from src.config import Config
from ..data import TaskData, TaskDataLoader, get_task_sequence
import os
import json
from datetime import datetime
from ..methods import FineTune
from .metrics import compute_all_metrics
from tqdm import tqdm

class Train:
    def __init__(self, wrapper: CLIPWrapper, config: Config, method:FineTune):
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

        self.method = method
        self.method.set_wrapper(wrapper)
        self.method.set_config(self.config)
        self.method.initialize()

    def train(self, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))

        self.method.set_criterion(nn.CrossEntropyLoss())

        # #@ check if the method requires task_id
        # if self.method.requires_task_id:
        #     self.method.before_task(task_id)

        #* =======================Data and Dataloader============================

        train_data = self.train_datas[task_id]
        test_data = self.test_datas[task_id]
        train_loader = self.train_loaders[task_id]
        test_loader = self.test_loaders[task_id]

        device = self.wrapper.model.device 

        #* stop khi improvement nhor honw epsilon        
        epsilon = float(self.config.train.epsilon)
        patience = int(self.config.train.patience)  #? số epoch liên tiếp không cải thiện trước khi dừng
        patience_counter = 0
        best_valid_loss = float("inf")  

        for epoch in range(self.config.train.max_epoch):
            #* ====================TRAIN=============================
            self.wrapper.model.train()
            train_loss = valid_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                images, labels = images.to(device), labels.to(device)
                text_tokenized = train_data.text_tokenized
                optimizer.zero_grad()

                loss = self.method.compute_loss(images, labels, text_tokenized)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            #* ======================Eval===========================
            self.wrapper.model.eval()
            with torch.inference_mode():
                text_features = self.wrapper.encode_text(test_data.text_tokenized)  #? encode fresh cho eval
                for images, labels in tqdm(test_loader, desc=f"Valid Epoch {epoch+1}", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    loss = self.method.compute_loss_inference_mode(images, labels, text_features)
                    valid_loss += loss.item()
                valid_loss /= len(test_loader)

            #* ====================lưu history================
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

        #* ====================== EARLY STOPPING ============================
            if valid_loss < best_valid_loss - epsilon:
                best_valid_loss = valid_loss
                patience_counter = 0  #? cải thiện đủ → reset counter
            else:
                patience_counter += 1  #? không cải thiện → tăng counter
                if patience_counter >= patience:
                    print(f"EARLY STOPPING (patience={patience})")
                    break


    def compute_metrics(self):
        return compute_all_metrics(self.results) | {
            "results_matrix": self.results,
            "history": self.history
        }

    def eval_all(self) -> list:
        result = [0.0] * len(self.tasks)
        self.wrapper.model.eval()
        device = self.wrapper.model.device

        with torch.inference_mode():
            for task_id in range(len(self.tasks)):
                data = self.test_datas[task_id]
                dataloader = self.test_loaders[task_id]
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
    
    #@ The training script
    def train_all_tasks(self, test_pipeline = False):
        self.wrapper.split_and_get_lora()
        self.wrapper.add_lora(r = self.config.train.r)
        self.tasks = get_task_sequence(test_pipeline)
        self.train_datas = [
            TaskData(task, "train", processor= self.wrapper.processor)
            for task in self.tasks
        ]
        self.train_loaders = [
            TaskDataLoader(
                train_data,
                batch_size= self.config.datasets.batch_size,
                num_workers=self.config.datasets.num_workers,
                pin_memory= bool(self.config.datasets.pin_memory))
            for train_data in self.train_datas
        ]
        self.test_datas = [
            TaskData(task, "test", processor= self.wrapper.processor)
            for task in self.tasks
        ]
        self.test_loaders = [
            TaskDataLoader(
                test_data,
                batch_size= self.config.datasets.batch_size,
                num_workers=self.config.datasets.num_workers,
                pin_memory= bool(self.config.datasets.pin_memory))
            for test_data in self.test_datas
        ]

        for task_id in range(len(self.tasks)):
            self.method.before_task(task_id)
            self.train(task_id=task_id)
            self.results.append(self.eval_all())
            self.method.after_task(task_id)
        
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
