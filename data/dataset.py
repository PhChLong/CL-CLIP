from torch.utils.data import Dataset

class TaskData(Dataset):
    def __init__(self, task, split = "train", image_processor = None):
        super().__init__()
        self.data = task[split]['img'] if 'img' in task[split].features else task[split]['image']
        self.task = task
        self.label_names = task['label_names']
        self.label_key = task[split][task['label_key']]
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.image_processor is not None:
            image = self.image_processor(
                images = image,
                return_tensors = 'pt'
            )['pixel_values'].squeeze(0)
        return image, self.label_names[self.label_key[idx]]
