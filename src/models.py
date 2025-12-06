from typing import Any, Dict

import torch.nn as nn
from torchvision.models import resnet18, vit_b_16
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, detr_resnet50

def create_model(task: str, model_name: str, num_classes: int, **kwargs):
    task = task.lower()
    if task == "classification":
        return create_classification_model(model_name, num_classes)
    elif task == "detection":
        return create_detection_model(model_name, num_classes, **kwargs)
    else:
        raise ValueError(f"tarea no soportada: {task}")
    
def create_classification_model(model_name:str, num_classes:int):
    if model_name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "vit_16":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"modelo no soportado: {model_name}")

def create_detection_model(
    model_name: str,
    num_classes: int,
    *,
    model_kwargs: Dict[str, Any] | None = None,
):
    model_name = model_name.lower()
    extra_args = model_kwargs or {}

    if model_name in {"faster_rcnn", "fasterrcnn_resnet50_fpn"}:
        return fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            **extra_args,
        )
    elif model_name in {"ssd", "ssd300_vgg16"}:
        return ssd300_vgg16(
            weights=None,
            num_classes=num_classes,
            **extra_args,
        )
    elif model_name in {"detr", "detr_resnet50"}:
        return detr_resnet50(
            weights=None,
            num_classes=num_classes,
            **extra_args,
        )
    elif model_name.startswith("yolo"):
        raise NotImplementedError(
            "La integración con modelos YOLO (Ultralytics) está planificada "
            "pero aún no se encuentra disponible."
        )
    else:
        raise ValueError(f"modelo de detección no soportado: {model_name}")
