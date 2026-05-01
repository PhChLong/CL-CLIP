from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
import torch.nn.functional as F
from .get_data import get_ref_img_dir, get_ref_text_data
from PIL import Image

class TaskData(Dataset):
    def __init__(self, task, split, processor):
        super().__init__()
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

class RefImageData(Dataset):
    def __init__(self, processor):
        super().__init__()
        self.img_dir: Path = get_ref_img_dir()
        self.image_processor = partial(processor.image_processor, return_tensors='pt')

        image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.img_paths = [path for path in self.img_dir.iterdir() if path.suffix.lower() in image_suffixes]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_tensor = self.image_processor(images= Image.open(self.img_paths[index]))['pixel_values'].squeeze(0)
        return img_tensor
    

class RefTextData(Dataset):
    def __init__(self, processor):
        super().__init__()
        self.text_data: Dataset = get_ref_text_data()
        self.text_processor = partial(processor, images=None, return_tensors='pt', padding=True)

    def __len__(self):
        return self.text_data['caption_data'].num_rows

    def __getitem__(self, index):
        return self.text_processor(text = self.text_data['caption_data'][index]['caption'])

PADDING_ID = 49407 #? EOS/PAD in CLIP, 49406 is BOS
def ref_text_collate_fn(batch):
    # batch will be a list of dictionary {'input_ids': tensor(), 'attention_mask': tensor()}
    input_ids = [tensor['input_ids'].squeeze(0) for tensor in batch]
    attention_masks = [tensor['attention_mask'].squeeze(0) for tensor in batch]
    max_len = max(t.shape[0] for t in input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    for t in input_ids:
        padded_input_ids.append(F.pad(t, (0, max_len - t.shape[0]), value = PADDING_ID))
    for t in attention_masks:
        padded_attention_masks.append(F.pad(t, (0, max_len - t.shape[0]), value = 0))
    return {'input_ids':  torch.stack(padded_input_ids), 'attention_mask': torch.stack(padded_attention_masks)}

class RefTextDataloader(DataLoader):
    def __init__(self, data, batch_size, num_workers, pin_memory):
        super().__init__(
            dataset = data,
            batch_size = batch_size,
            num_workers= num_workers,
            pin_memory= pin_memory,
            collate_fn=ref_text_collate_fn
        )

def ref_image_collate_fn(batch):
    return torch.stack(batch)

class RefImageDataloader(DataLoader):
    def __init__(self, data, batch_size, num_workers, pin_memory):
        super().__init__(
            dataset = data,
            batch_size = batch_size,
            num_workers= num_workers,
            pin_memory= pin_memory,
            collate_fn=ref_image_collate_fn
        )
