from models import CLIPWrapper, LoRAAdapter
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from data import TaskData, TaskDataLoader, get_task_sequence
from tqdm import tqdm
class C_CLIP(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
    
    def train_all_tasks(self):
        self.tasks = get_task_sequence()
        for task_id, task in enumerate(self.tasks):
            self.add_lora()
            self.train(task, task_id=task_id)
            self.plus_lora()
            self.results.append(self.eval_all())
        
        metrics = self.compute_metrics()
        self.save_logs()
        self.save_results()

        return metrics
    
    def plus_lora(self,lora_modules: LoRAAdapter):
        """ split and get lora, then param += alpha * lora

        Args:
            alpha (float): The hyperparameter for deciding how much information to keep from LoRA. Defaults to 0.5.
        """
        lora_modules = self.wrapper.split_and_get_lora()
        device = self.wrapper.model.device
        for layer_name in lora_modules.keys():
            attn, layer_type = layer_name.rsplit('.', maxsplit=1)
            attn = self.wrapper.model.get_submodule(attn)
            lora = lora_modules[layer_name]
            current = getattr(attn, layer_type)
            current.weight.data += lora.get_matrix().to(device) * float(self.config.train.alpha)
        self.wrapper.model.to(device)

    def train(self, task, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))

        criterion = nn.CrossEntropyLoss()

        #@ =======================Data and Dataloader============================
        train_data = TaskData(task, "train", processor= self.wrapper.processor)
        test_data = TaskData(task, "test", processor= self.wrapper.processor)
        train_loader = TaskDataLoader(
            train_data,
            batch_size= self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory= True)
        test_loader = TaskDataLoader(
            test_data,
            batch_size = self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory=True
        )

        device = self.wrapper.model.device 

        #@ stop khi improvement nhor honw epsilon        
        epsilon = float(self.config.train.epsilon)
        patience = int(self.config.train.patience)  #? số epoch liên tiếp không cải thiện trước khi dừng
        patience_counter = 0
        best_valid_loss = float("inf")  

        for epoch in range(self.config.train.max_epoch):
            #@ ====================TRAIN=============================
            self.wrapper.model.train()
            train_loss = valid_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                text_features = self.wrapper.encode_text(train_data.text_tokenized)
                logits = self.wrapper.forward_with_text_features(text_features, images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            #@ ======================Eval===========================
            self.wrapper.model.eval()
            with torch.inference_mode():
                text_features = self.wrapper.encode_text(train_data.text_tokenized)  #? encode fresh cho eval
                for images, labels in tqdm(test_loader, desc=f"Valid Epoch {epoch+1}", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    logits = self.wrapper.forward_with_text_features(text_features, images)
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
                valid_loss /= len(test_loader)

            #@ ====================lưu history================
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

        #@ ====================== EARLY STOPPING ============================
            if valid_loss < best_valid_loss - epsilon:
                best_valid_loss = valid_loss
                patience_counter = 0  #? cải thiện đủ → reset counter
            else:
                patience_counter += 1  #? không cải thiện → tăng counter
                if patience_counter >= patience:
                    print(f"EARLY STOPPING (patience={patience})")
                    break
