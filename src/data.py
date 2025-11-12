# utils/data.py
import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

def download_data(dataset_name, output_dir="./data", cache_dir="./data/hf_cache"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    target_dir = os.path.join(output_dir, dataset_name.replace("/", "__"))
    print(f"[download] guardando en: {target_dir}")

    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    ds.save_to_disk(target_dir)
    print(f"[download] dataset {dataset_name} guardado en {target_dir}")

def load_data(
    dataset_name,
    split="train",
    batch_size=32,
    output_dir="./data",
    img_size=224,
    cache_dir="./data/hf_cache",
):
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    local_path = os.path.join(output_dir, dataset_name.replace("/", "__"))

    if os.path.isdir(local_path):
        ds_dict = load_from_disk(local_path)
        if split not in ds_dict:
            raise ValueError(f"El split '{split}' no existe en el dataset guardado. Splits: {list(ds_dict.keys())}")
        ds = ds_dict[split]
        label_feature = ds_dict["train"].features["label"]
    else:
        ds_all = load_dataset(dataset_name, cache_dir=cache_dir)
        if split not in ds_all:
            raise ValueError(f"El split '{split}' no existe en el dataset remoto. Splits: {list(ds_all.keys())}")
        ds = ds_all[split]
        label_feature = ds_all["train"].features["label"]

    # num_classes
    if getattr(label_feature, "names", None):
        num_classes = len(label_feature.names)
    else:
        num_classes = int(label_feature.num_classes)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def apply_transforms(examples):
        # The 'examples' dictionary contains 'image' (PIL Image) and 'label'
        examples['image'] = [transform(img.convert('RGB')) for img in examples['image']]
        return examples

    ds.set_transform(apply_transforms, columns=['image'], output_all_columns=True)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=2,
        pin_memory=True
    )

    return loader, num_classes