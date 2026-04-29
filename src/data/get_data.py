import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset, load_from_disk
from pathlib import Path

# ─────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "dataset_cache"

TASK_SEQUENCE = [
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
        'name': 'eurosat',
        'hf_id': 'tanganke/eurosat',
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
def load_task(task: dict, test_pipeline: bool) -> dict:
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

    #@ néu chỉ đơn giản là test xem chạy oke ko thì sẽ chỉ lấy 1% data thôi
    if test_pipeline:
        subset_ratio = 0.001
        train_n = int(max(1, len(train_data) * subset_ratio))
        test_n = int(max(1, len(test_data) * subset_ratio))

        train_data = train_data.shuffle(seed = 42).select(range(train_n))
        test_data = test_data.shuffle(seed = 42).select(range(test_n))

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
def get_task_sequence(test_pipeline = False) -> list[dict]:
    return [load_task(task, test_pipeline) for task in TASK_SEQUENCE]

REF_DATASET = {
    'name': 'stl10_unlabeled',
    'hf_id': 'tanganke/stl10',
    'split': 'unlabeled',
    'n_samples': 5000,
}

#@ Load reference dataset cho ZSCL feature distillation — STL-10 unlabeled, 5K ảnh random
#@ Trả về HuggingFace Dataset (chỉ có 'image', không có label)
def get_ref_data() -> object:
    cache_path = CACHE_DIR / REF_DATASET['name']
    if (cache_path / 'dataset_info.json').exists():
        print(f"[cache] Loading ref data from {cache_path}")
        return load_from_disk(str(cache_path))

    print(f"[download] Downloading {REF_DATASET['hf_id']} ({REF_DATASET['split']})...")
    dataset = load_dataset(REF_DATASET['hf_id'], split=REF_DATASET['split'])
    dataset = dataset.select(range(REF_DATASET['n_samples']))  #?: lấy 5K đầu — shuffle trước nếu cần random hơn
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    print(f"[saved] ref data → {cache_path}")
    return dataset