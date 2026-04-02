from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
class TaskData(Dataset):
    def __init__(self, task, split = "train", processor = None):
        super().__init__()
        #@ 
        self.label_key = task[split][task['label_key']]

        self.imgs = task[split]['img'] if 'img' in task[split].features else task[split]['image']
        self.image_processor = partial(processor.image_processor, return_tensors = 'pt')
        self.img_tensors = [self.image_processor(images = img)['pixel_values'].squeeze(0) for img in self.imgs]
        
        prompts = [f"a photo of a {label_name}" for label_name in task['label_names']]
        self.text_processor = partial(processor, images = None, return_tensors = 'pt', padding = True)
        self.text_tokenized = self.text_processor(text = prompts)

    def __len__(self):
        return len(self.img_tensors)
    
    def __getitem__(self, idx):
        return self.img_tensors[idx], self.label_key[idx]
    
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
