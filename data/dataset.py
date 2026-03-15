import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset

def get_cifar100():
    return load_dataset('cifar100')

def get_flowers():
    return load_dataset('nelorth/oxford-flowers')

def get_cars():
    return load_dataset('tanganke/stanford_cars')

def get_aircraft():
    return load_dataset('HuggingFaceM4/FGVC-Aircraft', 'variant')

def get_dtd():
    return load_dataset('tanganke/dtd')

TASK_SEQUENCE = [
    {
        'name' : 'cifar100',
        'loader': get_cifar100,
        'train_split': 'train',
        'test_split': 'test'
    },
    {

        'name' : 'flowers',
        'loader': get_flowers,
        'train_split': 'train',
        'test_split': 'test'
    },
    {

        'name' : 'cars',
        'loader': get_cars,
        'train_split': 'train',
        'test_split': 'test'
    },
    {
        'name' : 'aircraft',
        'loader': get_aircraft,
        'train_split': 'train',
        'test_split': 'test'
    },
    {
        'name' : 'dtd',
        'loader': get_dtd,
        'train_split': 'train',
        'test_split': 'test'
    },
]

def get_task_sequence():
    tasks = []
    for task in TASK_SEQUENCE:
        dataset = task['loader']()
        train_data = dataset[task['train_split']]
        test_data = dataset[task['test_split']]
        label_key = 'fine_label' if 'fine_label' in train_data.features else 'label'
        label_names = train_data.features[label_key].names
        tasks.append( {
            'name': tasks['name'],
            'train': train_data,
            'test': test_data,
            'label_key': label_key,
            'label_names':label_names
        })
    return tasks