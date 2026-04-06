import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, load_from_disk
from pathlib import Path

# ─────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "dataset_cache"

TASK_SEQUENCE = [
    {
        'name': 'cars',
        'loader': lambda: load_dataset('tanganke/stanford_cars'),
        'train_split': 'train',
        'test_split': 'test'
    },
    {
        'name': 'dtd',
        'loader': lambda: load_dataset('tanganke/dtd'),
        'train_split': 'train',
        'test_split': 'test'
    },
    {
        'name': 'cifar100',
        'loader': lambda: load_dataset('cifar100'),
        'train_split': 'train',
        'test_split': 'test'
    }
]

# ─────────────────────────────────────────────

#@ Load 1 task — dùng cache nếu đã có, download và save nếu chưa
#@ Trả về dict với 'name', 'train', 'test', 'label_key', 'label_names'
def load_task(task: dict) -> dict:
    cache_path = CACHE_DIR / task['name']

    if cache_path.exists():
        print(f"[cache] Loading {task['name']} from {cache_path}")
        dataset = load_from_disk(str(cache_path))  #?: load_from_disk nhận str, không phải Path
    else:
        print(f"[download] Downloading {task['name']}...")
        dataset = task['loader']()
        cache_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))
        print(f"[saved] {task['name']} → {cache_path}")

    train_data = dataset[task['train_split']]
    test_data  = dataset[task['test_split']]

    label_key   = 'fine_label' if 'fine_label' in train_data.features else 'label'
    label_names = train_data.features[label_key].names

    return {
        'name'       : task['name'],
        'train'      : train_data,
        'test'       : test_data,
        'label_key'  : label_key,
        'label_names': label_names
    }

#@ Load toàn bộ task sequence, tự động cache từng task riêng lẻ
def get_task_sequence() -> list[dict]:
    return [load_task(task) for task in TASK_SEQUENCE]