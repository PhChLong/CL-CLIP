from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
class TaskData(Dataset):
    def __init__(self, task, split, processor):
        self.labels = task[split][task['label_key']]          
        self.imgs = task[split]['img'] if 'img' in task[split].features else task[split]['image']
        self.image_processor = partial(processor.image_processor, return_tensors='pt')
      

        prompts = [f"a photo of a {label_name}" for label_name in task['label_names']]
        self.text_processor = partial(processor, images=None, return_tensors='pt', padding=True)
        self.text_tokenized = self.text_processor(text=prompts)

    def __len__(self):
        return len(self.imgs)     #? đổi từ img_tensors → imgs vì img_tensors không còn tồn tại

    def __getitem__(self, idx):
        img_tensor = self.image_processor(images=self.imgs[idx])['pixel_values'].squeeze(0)  #? preprocess on-the-fly
        return img_tensor, self.labels[idx]                   
    
#* collate_fn sẽ đưa về 1 tensor (đã stack) của image, và đưa label thành tensors
def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

class TaskDataLoader(DataLoader):
    def __init__(self, data, batch_size, num_workers, pin_memory):
        super().__init__(
            dataset = data,
            batch_size = batch_size,
            num_workers= num_workers,
            pin_memory= pin_memory,
            collate_fn=collate_fn
        )
