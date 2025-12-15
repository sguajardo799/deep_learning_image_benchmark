#!/usr/bin/env python3
"""
export_detection_to_onnx.py

Export Torch detection models (SSD / DETR) to ONNX.

Supports:
- SSD (torchvision):  ssd300_vgg16, ssdlite320_mobilenet_v3_large
  * ONNX wrapper pads detections to fixed --max-dets and returns:
      boxes:  [B, max_dets, 4]  (xyxy, in pixels)
      scores: [B, max_dets]
      labels: [B, max_dets]     (int64)
- DETR:
  * torch.hub facebookresearch/detr: detr_resnet50 / detr_resnet101
  * or torchvision.models.detection.detr_resnet50 (if available in your torchvision)
  * ONNX outputs:
      pred_logits: [B, num_queries, num_classes+1]
      pred_boxes:  [B, num_queries, 4]  (cxcywh normalized)

Checkpoint loading:
- If you trained the model, pass --checkpoint; it will try common keys:
  model_state_dict / state_dict / raw state_dict.

No class mapping: we don't map labels.

Example (DETR):
  python export_detection_to_onnx.py \
    --arch detr_resnet50 --source hub \
    --checkpoint /app/checkpoints/detr.pth \
    --num-classes 91 \
    --img-size 800 800 \
    --out /app/checkpoints/detr.onnx

Example (SSD):
  python export_detection_to_onnx.py \
    --arch ssd300_vgg16 --source torchvision \
    --checkpoint /app/checkpoints/ssd.pth \
    --num-classes 91 \
    --img-size 300 300 \
    --max-dets 100 \
    --out /app/checkpoints/ssd300.onnx
"""

from __future__ import annotations
import argparse
from collections import OrderedDict
from typing import Dict

import torch


def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    cleaned = OrderedDict()
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys (up to 20): {missing[:20]} (total={len(missing)})")
    if unexpected:
        print(f"[WARN] Unexpected keys (up to 20): {unexpected[:20]} (total={len(unexpected)})")


def build_model(arch: str, source: str, num_classes: int) -> torch.nn.Module:
    arch = arch.lower()
    source = source.lower()

    if arch.startswith("detr"):
        if source == "hub":
            m = torch.hub.load("facebookresearch/detr", arch, pretrained=False)
            # hub DETR uses num_classes+1 (includes "no object")
            if hasattr(m, "class_embed") and getattr(m.class_embed, "out_features", None) not in (None, num_classes + 1):
                try:
                    in_f = m.class_embed.in_features
                    m.class_embed = torch.nn.Linear(in_f, num_classes + 1)
                except Exception:
                    pass
            return m

        if source == "torchvision":
            import torchvision
            if hasattr(torchvision.models.detection, "detr_resnet50"):
                return torchvision.models.detection.detr_resnet50(weights=None, num_classes=num_classes)
            raise ValueError("torchvision doesn't expose detr_resnet50 in this version; use --source hub.")
        raise ValueError("For DETR, use --source hub or --source torchvision.")

    # SSD family (torchvision)
    import torchvision
    if arch == "ssd300_vgg16":
        return torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=num_classes)
    if arch == "ssdlite320_mobilenet_v3_large":
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)

    raise ValueError(f"Unsupported arch: {arch}")


class SSDOnnxWrapper(torch.nn.Module):
    """Torchvision SSD -> pad detections to tensors for ONNX."""
    def __init__(self, model: torch.nn.Module, max_dets: int = 100):
        super().__init__()
        self.model = model
        self.max_dets = int(max_dets)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        dets = self.model(list(x))  # List[Dict]
        B = x.shape[0]
        device = x.device
        boxes = torch.zeros((B, self.max_dets, 4), device=device, dtype=torch.float32)
        scores = torch.zeros((B, self.max_dets), device=device, dtype=torch.float32)
        labels = torch.zeros((B, self.max_dets), device=device, dtype=torch.int64)

        for i, d in enumerate(dets):
            b = d.get("boxes", torch.empty((0, 4), device=device)).to(torch.float32)
            s = d.get("scores", torch.empty((0,), device=device)).to(torch.float32)
            l = d.get("labels", torch.empty((0,), device=device)).to(torch.int64)
            n = min(b.shape[0], self.max_dets)
            if n > 0:
                boxes[i, :n] = b[:n]
                scores[i, :n] = s[:n]
                labels[i, :n] = l[:n]

        return boxes, scores, labels


class DETROnnxWrapper(torch.nn.Module):
    """Return raw DETR tensors (pred_logits, pred_boxes)."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, dict) and "pred_logits" in out and "pred_boxes" in out:
            return out["pred_logits"], out["pred_boxes"]
        raise RuntimeError(f"Unexpected DETR output type/keys: {type(out)} {getattr(out,'keys',lambda:[])()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["ssd300_vgg16", "ssdlite320_mobilenet_v3_large", "detr_resnet50", "detr_resnet101"])
    ap.add_argument("--source", default="torchvision", choices=["torchvision", "hub"],
                    help="Where to build DETR from. SSD is always torchvision.")
    ap.add_argument("--checkpoint", default="", help=".pth checkpoint path (optional)")
    ap.add_argument("--num-classes", type=int, required=True, help="Number of classes in your training (no mapping).")

    ap.add_argument("--img-size", nargs=2, type=int, default=[800, 800], help="H W dummy input size for export")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)

    ap.add_argument("--max-dets", type=int, default=100, help="SSD only: pad detections to this many boxes")

    ap.add_argument("--out", required=True, help="Output .onnx path")
    ap.add_argument("--fp16", action="store_true", help="Export with fp16 dummy input (model still fp32 unless you cast).")
    ap.add_argument("--dynamic-batch", action="store_true", help="Make batch dimension dynamic in ONNX.")
    args = ap.parse_args()

    model = build_model(args.arch, args.source, args.num_classes)
    if args.checkpoint:
        load_checkpoint_flexible(model, args.checkpoint)
    model.eval()

    H, W = int(args.img_size[0]), int(args.img_size[1])
    dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.randn(args.batch, 3, H, W, dtype=dtype)

    dynamic_axes: Dict[str, Dict[int, str]] = {}
    if args.dynamic_batch:
        dynamic_axes["input"] = {0: "batch"}

    input_names = ["input"]

    if args.arch.startswith("detr"):
        wrapper = DETROnnxWrapper(model)
        output_names = ["pred_logits", "pred_boxes"]
        if args.dynamic_batch:
            dynamic_axes["pred_logits"] = {0: "batch"}
            dynamic_axes["pred_boxes"] = {0: "batch"}
    else:
        wrapper = SSDOnnxWrapper(model, max_dets=args.max_dets)
        output_names = ["boxes", "scores", "labels"]
        if args.dynamic_batch:
            dynamic_axes["boxes"] = {0: "batch"}
            dynamic_axes["scores"] = {0: "batch"}
            dynamic_axes["labels"] = {0: "batch"}

    print(f"[INFO] Exporting {args.arch} to ONNX: {args.out}")
    print(f"[INFO] Dummy input: {tuple(dummy.shape)} dtype={dummy.dtype} opset={args.opset}")
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )
    print("[OK] Export done.")


if __name__ == "__main__":
    main()
