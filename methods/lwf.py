from models import CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from copy import deepcopy
from data import TaskData, TaskDataLoader
from tqdm import tqdm

class LwF_LoRA(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.wrapper = wrapper
        self.add_lora()

        #? tách riêng các thành phần của model
        self.vision_model = wrapper.model.vision_model
        self.org_visual_proj = wrapper.model.visual_projection
        self.text_model = wrapper.model.text_model
        self.org_text_proj = wrapper.model.text_projection

        #? setpoint cho old tasks:
        #@ {old_task_id: tensor [N, D]}
        self.old_task_setpoints = {}
        
        #? nhiệt độ distillation
        self.distill_temp = self.config.train.distill_temp

        #? trọng số loss cũ
        self.lambda_old = getattr(self.config.train, "lambda_old", 1.0)
        
        #? id của các task đã train
        self.trained_task_id = []
    
    #@ Train LwF cho 1 task: KD loss từ old LoRA + CE loss trên current task
    #@ text_features encode trong batch loop vì dùng train_data.text_tokenized (tokenized sẵn)
    def train(self, task, task_id=None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                lr=float(self.config.train.lr),
                                weight_decay=float(self.config.train.weight_decay))
        criterion = nn.CrossEntropyLoss()
        T = float(self.config.train.distill_temp)

        #* ====================== Data và Dataloader ============================
        train_data = TaskData(task, "train", processor=self.wrapper.processor)
        test_data = TaskData(task, "test", processor=self.wrapper.processor)
        train_loader = TaskDataLoader(train_data, batch_size=self.config.datasets.batch_size,
                                    num_workers=self.config.datasets.num_workers, pin_memory=True)
        test_loader = TaskDataLoader(test_data, batch_size=self.config.datasets.batch_size,
                                    num_workers=self.config.datasets.num_workers, pin_memory=True)

        device = self.wrapper.model.device

        #* snapshot teacher LoRA trước khi train — sẽ không bao giờ bị update
        old_LoRA = self.wrapper.split_and_get_lora()
        old_LoRA_copy = deepcopy(old_LoRA)
        self.wrapper.load_lora(old_LoRA)  #? restore lại current LoRA sau khi split

        epsilon = float(self.config.train.epsilon)
        best_valid_loss = float("inf")
        patience = int(self.config.train.patience)  #? số epoch liên tiếp không cải thiện trước khi dừng
        patience_counter = 0

        for epoch in range(self.config.train.max_epoch):
            #* ====================== TRAIN =====================================
            self.wrapper.model.train()
            train_loss = valid_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
                images, labels = images.to(device), labels.to(device)
                #? encode text trong loop — text_tokenized đã tokenize sẵn, cost thấp
                optimizer.zero_grad()
                text_features = self.wrapper.encode_text(train_data.text_tokenized)

                #* tính old_logits từ teacher (old LoRA)
                current_LoRA = self.wrapper.split_and_get_lora()
                self.wrapper.load_lora(old_LoRA_copy)  #? swap sang teacher
                with torch.inference_mode():
                    teacher_text_features = self.wrapper.encode_text(train_data.text_tokenized)
                    old_logits = self.wrapper.forward_with_text_features(teacher_text_features, images)
                soft_targets = F.softmax(old_logits / T, dim=1)  #? không cần .detach() — inference_mode đã ngăn grad
                self.wrapper.split_and_get_lora()
                self.wrapper.load_lora(current_LoRA)  #? swap lại student

                #* tính current logits và loss
                logits = self.wrapper.forward_with_text_features(text_features, images)
                soft_predictions = F.log_softmax(logits / T, dim=1)
                loss_kd = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (T ** 2)
                loss_ce = criterion(logits, labels)
                loss = loss_kd + loss_ce

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            #* ====================== EVAL ======================================
            self.wrapper.model.eval()
            with torch.inference_mode():
                text_features = self.wrapper.encode_text(train_data.text_tokenized)  #? encode 1 lần cho eval
                for images, labels in tqdm(test_loader, desc=f"Valid Epoch {epoch+1}", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    logits = self.wrapper.forward_with_text_features(text_features, images)
                    loss = criterion(logits, labels)
                    valid_loss += loss.item()
            valid_loss /= len(test_loader)

            log = {"task_id": task_id, "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
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