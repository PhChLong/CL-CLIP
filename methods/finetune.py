from models import LoRAAdapter, CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from data import TaskData, TaskDataLoader
from tqdm import tqdm
class FineTune(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.add_lora()
    def add_lora(self):
        device = self.wrapper.model.device
        for param in self.wrapper.model.parameters():
            param.requires_grad = False
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                #* Vision
                attn = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r= self.config.train.r))

                # #*Text
                attn = self.wrapper.model.text_model.encoder.layers[i].self_attn
                original = getattr(attn, layer_type)
                setattr(attn, layer_type, LoRAAdapter(original, r=self.config.train.r))
        self.wrapper.model.to(device)

    def train(self, task, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))

        criterion = nn.CrossEntropyLoss()
        prompts = [f"a photo of a {name}" for name in task['label_names']]

        #* =======================Data and Dataloader============================
        train_data = TaskData(task, "train", image_processor= self.wrapper.processor.image_processor)
        test_data = TaskData(task, "test", image_processor= self.wrapper.processor.image_processor)
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

        #* stop khi improvement nhor honw epsilon        
        epsilon = float(self.config.train.epsilon)

        prev_valid_loss = None
        best_valid_loss = float("inf")  

        for epoch in range(self.config.train.max_epoch):
            #* ====================TRAIN=============================
            text_features = self.wrapper.encode_text(prompts).detach()
            self.wrapper.model.train()
            train_loss = valid_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = self._pred(images, text_features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            #* ======================Eval===========================
            self.wrapper.model.eval()
            with torch.inference_mode():
                for images, labels in tqdm(test_loader, desc=f"Valid Epoch {epoch+1}", leave=False):
                    labels = labels.to(device)
                    logits = self._pred(images, text_features)
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
            valid_loss /= len(test_loader)

            #* kiểm tra nếu model học
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            #* early stopping
            delta = None if prev_valid_loss is None else (- valid_loss + prev_valid_loss)
                
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

            if delta is not None and delta < epsilon:
                print(f"EARLY STOPPING")
                break

            prev_valid_loss = valid_loss
