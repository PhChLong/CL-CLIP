import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import CLIPWrapper
from config import Config
from data import TaskData

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)
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

    def train(self, task, max_epoch = 3):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))
        criterion = nn.CrossEntropyLoss()
        self.prompts = [f"a photo of {name}" for name in task['label_names']]
        train_data = TaskData(task, "train")
        test_data = TaskData(task, "test")
        train_loader = DataLoader(
            train_data,
            batch_size= self.config.data.batch_size,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_data,
            batch_size = self.config.data.batch_size,
            collate_fn=collate_fn
        )
        device = self.wrapper.model.device 
        for epoch in range(max_epoch):
            self.wrapper.model.train()
            train_loss = valid_loss = 0
            for images, labels in train_loader:
                labels = labels.to(device)
                optimizer.zero_grad()
                probs = self._loss(images)
                loss = criterion(probs, labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            self.wrapper.model.eval()
            for images, labels in test_loader:
                labels = labels.to(device)
                probs = self._loss(images)
                loss = criterion(probs, labels)
                valid_loss += loss.item()
            valid_loss /= len(test_loader)
            print(f"{epoch=} || {train_loss=} || {valid_loss=}")
    
    def _loss(self, images):
        outputs = self.wrapper(self.prompts, images)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim = -1)
        return probs