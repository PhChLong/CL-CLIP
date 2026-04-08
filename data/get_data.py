import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, load_from_disk
from pathlib import Path

# ─────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "dataset_cache"

TASK_SEQUENCE = [
    {
        'name': 'eurosat',
        'hf_id': 'tanganke/eurosat',
        'loader_kwargs': {},
        'train_split': 'train',
        'test_split': 'test',
        'label_key': 'label',
    },
    {
        'name': 'flowers102',
        'hf_id': 'pufanyi/flowers102',
        'loader_kwargs': {},
        'train_split': 'train',
        'test_split': 'test',
        'label_key': 'label',
    },
    {
        'name': 'cars',
        'hf_id': 'tanganke/stanford_cars',
        'loader_kwargs': {},
        'train_split': 'train',
        'test_split': 'test',
        'label_key': 'label',
    },
    {
        'name': 'cifar100',
        'hf_id': 'cifar100',
        'loader_kwargs': {},
        'train_split': 'train',
        'test_split': 'test',
        'label_key': 'fine_label',  #?: cifar100 dùng 'fine_label' thay vì 'label'
    },
    {
        'name': 'dtd',
        'hf_id': 'tanganke/dtd',
        'loader_kwargs': {},
        'train_split': 'train',
        'test_split': 'test',
        'label_key': 'label',
    },
]
# ─────────────────────────────────────────────
#@ Load 1 task — dùng cache nếu đã có, download nếu chưa
#@ Trả về dict với 'name', 'train', 'test', 'label_key', 'label_names'
def load_task(task: dict) -> dict:
    cache_path = CACHE_DIR / task['name']

    if cache_path.exists():
        print(f"[cache] Loading {task['name']} from {cache_path}")
        dataset = load_from_disk(str(cache_path))
    else:
        print(f"[download] Downloading {task['name']}...")
        dataset = load_dataset(task['hf_id'], **task['loader_kwargs'])
        cache_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))
        print(f"[saved] {task['name']} → {cache_path}")

    train_data = dataset[task['train_split']]
    test_data  = dataset[task['test_split']]
    label_key  = task['label_key']

    feature = train_data.features[label_key]
    #? ClassLabel có .names, Value('int64') thì không — cần handle cả hai
    label_names = feature.names if hasattr(feature, 'names') else None

    return {
        'name'       : task['name'],
        'train'      : train_data,
        'test'       : test_data,
        'label_key'  : label_key,
        'label_names': label_names,
    }

#@ Load toàn bộ task sequence, tự động cache từng task riêng lẻ
def get_task_sequence() -> list[dict]:
    return [load_task(task) for task in TASK_SEQUENCE]
