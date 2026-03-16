from torch.utils.data import Dataset

class TaskData(Dataset):
    def __init__(self, task, split = "train", transform=None):
        super().__init__()
        self.data = task[split]['img']
        self.task = task
        self.label_key = task[split][task['label_key']]
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_key[idx]
