from typing import Any, Dict

import torch.nn as nn
from torchvision.models import resnet18, vit_b_16
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16

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
        try:
            # Try importing from torchvision (future proofing)
            from torchvision.models.detection import detr_resnet50
            return detr_resnet50(
                weights=None,
                num_classes=num_classes,
                **extra_args,
            )
        except ImportError:
            import torch
            print("DETR not found in torchvision, loading from torch.hub...")
            # Load model from hub
            model = torch.hub.load(
                'facebookresearch/detr:main',
                'detr_resnet50',
                pretrained=False,
                num_classes=num_classes,
            )
            return DetrWrapper(model, num_classes)

    elif model_name.startswith("yolo"):
        raise NotImplementedError(
            "La integración con modelos YOLO (Ultralytics) está planificada "
            "pero aún no se encuentra disponible."
        )
    else:
        raise ValueError(f"modelo de detección no soportado: {model_name}")


class DetrWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self._setup_hub_imports()
        
        from models.matcher import build_matcher
        from models.detr import SetCriterion
        from util.misc import nested_tensor_from_tensor_list
        
        self.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        
        # Configuration for Matcher and Criterion
        class Args:
            set_cost_class = 1
            set_cost_bbox = 5
            set_cost_giou = 2
            bbox_loss_coef = 5
            giou_loss_coef = 2
            eos_coef = 0.1
            
        args = Args()
        matcher = build_matcher(args)
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
        losses = ['labels', 'boxes', 'cardinality']
        
        # Note: num_classes in SetCriterion must match the model's num_classes
        self.criterion = SetCriterion(num_classes, matcher, weight_dict, args.eos_coef, losses)

    def _setup_hub_imports(self):
        import torch
        import glob
        import os
        import sys
        hub_dir = torch.hub.get_dir()
        # Find the repo dir. It usually starts with facebookresearch_detr
        repo_dirs = glob.glob(os.path.join(hub_dir, 'facebookresearch_detr_*'))
        if repo_dirs:
            repo_dir = repo_dirs[0]
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
        else:
            # Fallback: maybe it's named differently or not yet downloaded?
            # It should be downloaded by torch.hub.load
            pass

    def forward(self, images, targets=None):
        # images: list of tensors [C, H, W]
        # targets: list of dicts
        
        # Ensure criterion is on the correct device
        if self.criterion.weight_dict['loss_ce'] > 0:
             # Check device of a parameter
             device = next(self.model.parameters()).device
             # SetCriterion doesn't have parameters usually, but buffers?
             # Actually SetCriterion is a Module, so .to(device) works if it has buffers.
             # But it mainly uses the device of inputs.
             pass

        # Convert images to NestedTensor
        samples = self.nested_tensor_from_tensor_list(images)
        samples = samples.to(images[0].device)
        
        outputs = self.model(samples)
        
        if self.training or targets is not None:
            if targets is None:
                raise ValueError("Targets must be provided in training mode")
            
            # Convert targets to DETR format (cx, cy, w, h) normalized
            detr_targets = self._convert_targets(targets, images)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, detr_targets)
            
            # Return only weighted losses
            return {k: v for k, v in loss_dict.items() if k in self.criterion.weight_dict}
        else:
            return outputs

    def _convert_targets(self, targets, images):
        import torch
        new_targets = []
        for t, img in zip(targets, images):
            h, w = img.shape[-2:]
            boxes = t["boxes"]
            if boxes.numel() == 0:
                new_boxes = torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
            else:
                # xyxy to cxcywh
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                
                # Normalize
                cx = cx / w
                cy = cy / h
                bw = bw / w
                bh = bh / h
                
                new_boxes = torch.stack([cx, cy, bw, bh], dim=1)
            
            new_t = {k: v for k, v in t.items()}
            new_t["boxes"] = new_boxes
            new_targets.append(new_t)
        return new_targets
