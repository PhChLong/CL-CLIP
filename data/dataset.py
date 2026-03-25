from torch.utils.data import Dataset, DataLoader
import torch

class TaskData(Dataset):
    def __init__(self, task, split = "train", image_processor = None):
        super().__init__()
        self.data = task[split]['img'] if 'img' in task[split].features else task[split]['image']
        self.task = task
        self.label_key = task[split][task['label_key']]
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.image_processor is not None:
            image_tensor = self.image_processor(
                images = image,
                return_tensors = 'pt'
            )['pixel_values'].squeeze(0)
        return image_tensor, self.label_key[idx]   #* image_tensor ở đây đã là tensor rồi
                                            #* label thì vẫn còn là các số label

#* collate_fn sẽ đưa về 1 tensor (đã stack) của image, và đưa label thành tensors
def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)
class TaskDataLoader(DataLoader):
    def __init__(self, data, batch_size, num_workers, pin_memory = True):
        super().__init__(
            dataset = data,
            batch_size = batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
