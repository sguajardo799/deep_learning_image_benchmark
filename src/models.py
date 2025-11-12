import torch.nn as nn
from torchvision.models import resnet18, vit_b_16

def create_model(task: str, model_name: str, num_classes: int, **kwargs):
    task = task.lower()
    if task == "classification":
        return create_classification_model(model_name, num_classes)
    elif task == "detection":
        raise ValueError("[EN DESARROLLO]")
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