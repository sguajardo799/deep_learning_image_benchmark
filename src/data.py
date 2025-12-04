# utils/data.py
import os
from typing import Dict, Optional

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

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
    task: str = "classification",
):
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    local_path = os.path.join(output_dir, dataset_name.replace("/", "__"))

    if os.path.isdir(local_path):
        ds_dict = load_from_disk(local_path)
    else:
        ds_dict = load_dataset(dataset_name, cache_dir=cache_dir)

    if split not in ds_dict:
        # Intentar alias comunes
        if split == "validation" and "val" in ds_dict:
            split = "val"
        elif split == "val" and "validation" in ds_dict:
            split = "validation"
        else:
            raise ValueError(f"El split '{split}' no existe. Splits disponibles: {list(ds_dict.keys())}")

    ds = ds_dict[split]
    task = task.lower()

    if task == "classification":
        return _build_classification_dataloader(ds, batch_size, split, img_size)
    if task == "detection":
        return _build_detection_dataloader(ds, batch_size, split, img_size)

    raise ValueError(f"Tarea no soportada: {task}")


def _build_classification_dataloader(ds, batch_size, split, img_size):
    features = ds.features
    if "label" not in features:
        raise ValueError("El dataset seleccionado no contiene la columna 'label' para clasificación.")

    label_feature = features["label"]
    num_classes = _infer_num_classes(label_feature)

    transforms_list = []
    if img_size is not None:
        transforms_list.append(T.Resize((img_size, img_size)))
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    transform = T.Compose(transforms_list)

    def apply_transforms(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    ds.set_transform(apply_transforms, columns=["image"], output_all_columns=True)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=2,
        pin_memory=True,
    )
    return loader, num_classes


def _build_detection_dataloader(ds, batch_size, split, img_size):
    features = ds.features
    objects_feature = features.get("objects")
    if objects_feature is None:
        raise ValueError("El dataset no contiene la columna 'objects' necesaria para detección.")

    label_key, label_feature = _resolve_detection_label_feature(objects_feature)
    label_names = getattr(label_feature, "names", None)
    label_mapping = {name: idx for idx, name in enumerate(label_names)} if label_names else None
    base_num_classes = _infer_num_classes(label_feature)
    num_classes = base_num_classes + 1  # background

    dataset = DetectionDataset(
        ds,
        label_key=label_key,
        label_mapping=label_mapping,
        img_size=img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_detection_collate_fn,
    )
    return loader, num_classes


def _infer_num_classes(label_feature):
    names = getattr(label_feature, "names", None)
    if names:
        return len(names)
    
    # Check for inner feature (e.g. Sequence(ClassLabel))
    inner_feature = getattr(label_feature, "feature", None)
    if inner_feature:
        names = getattr(inner_feature, "names", None)
        if names:
            return len(names)
        num_classes = getattr(inner_feature, "num_classes", None)
        if num_classes is not None:
            return int(num_classes)

    num_classes = getattr(label_feature, "num_classes", None)
    if num_classes is not None:
        return int(num_classes)
    raise ValueError("No se pudo inferir el número de clases del dataset.")


def _resolve_detection_label_feature(objects_feature):
    feature = getattr(objects_feature, "feature", None)
    if feature is None:
        # Fallback: si es un dict o Sequence que actúa como dict
        if isinstance(objects_feature, dict) or hasattr(objects_feature, "keys"):
            feature = objects_feature
        else:
            raise ValueError("La columna 'objects' no expone sub-características utilizables.")

    candidate_keys = ("category", "label", "category_id", "id", "class_id")
    for key in candidate_keys:
        try:
            return key, feature[key]
        except (KeyError, TypeError):
            continue
    raise ValueError("No se encontró un campo de etiquetas dentro de 'objects' (p. ej. 'category').")


class DetectionDataset(Dataset):
    """Envuelve un Dataset de HuggingFace para producir batches estilo torchvision."""

    def __init__(
        self,
        hf_dataset,
        label_key: str,
        label_mapping: Optional[Dict[str, int]] = None,
        img_size: Optional[int] = None,
    ):
        self.dataset = hf_dataset
        self.label_key = label_key
        self.label_mapping = label_mapping or {}
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((img_size, img_size)) if img_size else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"].convert("RGB")
        orig_w, orig_h = image.size
        scale_x = scale_y = 1.0

        if self.resize:
            image = self.resize(image)
            new_w, new_h = image.size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

        image = self.to_tensor(image)

        objects = example.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("Cada ejemplo de detección debe incluir un diccionario 'objects'.")

        boxes = []
        labels = []
        areas = []
        crowd_flags = []

        bboxes = objects.get("bbox") or []
        raw_labels = objects.get(self.label_key) or []
        raw_areas = objects.get("area") or []
        raw_crowd = (
            objects.get("is_crowd")
            or objects.get("iscrowd")
            or objects.get("crowd")
            or []
        )

        for box_idx, bbox in enumerate(bboxes):
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])

            label_value = raw_labels[box_idx] if box_idx < len(raw_labels) else 0
            labels.append(self._encode_label(label_value))

            if box_idx < len(raw_areas) and raw_areas[box_idx] is not None:
                scaled_area = float(raw_areas[box_idx]) * (scale_x * scale_y)
                areas.append(scaled_area)

            if box_idx < len(raw_crowd):
                crowd_flags.append(int(raw_crowd[box_idx]))

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64),
        }

        if areas:
            target["area"] = torch.tensor(areas, dtype=torch.float32)
        if crowd_flags:
            target["iscrowd"] = torch.tensor(crowd_flags, dtype=torch.int64)

        image_id = example.get("image_id", idx)
        target["image_id"] = torch.tensor([int(image_id)], dtype=torch.int64)

        return image, target

    def _encode_label(self, label):
        if label is None:
            return 0
        if isinstance(label, str):
            if label in self.label_mapping:
                return self.label_mapping[label]
            raise KeyError(f"Etiqueta '{label}' no está mapeada en label_mapping.")
        return int(label)


def _detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
