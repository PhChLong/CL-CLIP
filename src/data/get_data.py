import shutil
import kagglehub
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


#@Ref data for ZSCL

REF_DATASET = {
    'name_image': 'miniimagenet',
    'image_kaggle_id': 'deeptrial/miniimagenet',
    'name_text': "conceptual_captions",
    'text_hf_id': "conceptual_captions"
}

def get_ref_text_dir(test_pipeline:bool):
    cache_path = CACHE_DIR / REF_DATASET['name_text']

    if cache_path.exists():
        print(f"[cache] Loading {REF_DATASET['name_text']} from {cache_path}")
        dataset = load_from_disk(str(cache_path))
    else:
        print(f"[download] Downloading {REF_DATASET['name']}...")
        dataset = load_dataset(REF_DATASET['text_hf_id'])
        dataset = dataset['train']
        dataset = dataset.shuffle(seed = 42).select(range(5000))

        cache_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))
        print(f"[saved] {REF_DATASET['name']} → {cache_path}")

    #@ néu chỉ đơn giản là test xem chạy oke ko thì sẽ chỉ lấy 1% data thôi
    if test_pipeline:
        subset_ratio = 0.01
        size = int(max(1, len(dataset) * subset_ratio))

        dataset = dataset.select(range(size))

    feature = train_data.features[label_key]
    #? ClassLabel có .names, Value('int64') thì không — cần handle cả hai
    label_names = feature.names if hasattr(feature, 'names') else None

    return {
        'name'       : REF_DATASET['name_text'],
        'label_key'  : label_key,
        'label_names': label_names,
    }

def get_ref_img_dir() -> Path:
    image_cache_path = CACHE_DIR / REF_DATASET['name_image']
    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    #@ check nếu trong folder đã chứa ảnh, thì return
    cached_images = [
        path for path in image_cache_path.glob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    ] if image_cache_path.exists() else []

    if cached_images:
        print(f"[cache] ref images already stored in {image_cache_path}")
        return image_cache_path

    #@ nếu chưa tồn tại thì download data
    print(f"[download] downloading {REF_DATASET['image_kaggle_id']} ")
    image_cache_path.mkdir(parents=True, exist_ok=True)

    kaggle_cache_dir = image_cache_path / "kagglehub_cache"
    os.environ["KAGGLEHUB_CACHE"] = str(kaggle_cache_dir)

    try:
        downloaded_path = Path(kagglehub.dataset_download(REF_DATASET['image_kaggle_id']))
        source_images = [
            path for path in downloaded_path.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        ]

        if not source_images:
            raise FileNotFoundError(f"No image files found in downloaded dataset: {downloaded_path}")

        copied_count = 0
        for image in source_images:
            relative = image.relative_to(downloaded_path).with_suffix("")
            dest_name = "_".join(relative.parts) + image.suffix.lower()
            dest_path = image_cache_path / dest_name

            counter = 1
            while dest_path.exists():
                dest_path = image_cache_path / f"{Path(dest_name).stem}_{counter}{image.suffix.lower()}"
                counter += 1

            shutil.copy2(image, dest_path)
            copied_count += 1
    finally:
        if kaggle_cache_dir.exists():
            shutil.rmtree(kaggle_cache_dir)

    print("Original KaggleHub path:", downloaded_path)
    print(f"Copied {copied_count} images to:", image_cache_path)
    return image_cache_path
