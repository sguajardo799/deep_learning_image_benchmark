#!/usr/bin/env python3
"""
export_to_onnx.py

Export a Torch checkpoint (.pth) for torchvision ResNet-18 or ViT-B/16 to ONNX.

Examples:
  python export_to_onnx.py --model resnet18 --num-classes 100 --checkpoint /path/best.pth --output resnet18_fp32.onnx
  python export_to_onnx.py --model vit_b_16 --num-classes 100 --checkpoint /path/best.pth --output vit16_fp32.onnx
  # optional: also write an FP16-converted ONNX (keep IO in FP32 by default)
  python export_to_onnx.py --model resnet18 --num-classes 100 --checkpoint best.pth --output resnet18_fp32.onnx --export-fp16-onnx resnet18_fp16.onnx
"""
from __future__ import annotations
import argparse
import os
from collections import OrderedDict
import torch
from torch import nn
import torchvision

def build_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in ("vit_b_16", "vit16", "vit_b16", "vit-b-16"):
        model = torchvision.models.vit_b_16(weights=None)
        # torchvision ViT has model.heads.head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported --model '{name}'. Use resnet18 or vit_b_16.")

def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # strip common prefixes
    cleaned = OrderedDict()
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys (showing up to 20): {missing[:20]}  (total={len(missing)})")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 20): {unexpected[:20]}  (total={len(unexpected)})")

def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset: int,
    batch: int,
    input_size: int,
    dynamic_batch: bool,
) -> None:
    model.eval()
    dummy = torch.randn(batch, 3, input_size, input_size, dtype=torch.float32)

    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] Wrote ONNX: {os.path.abspath(output_path)}")

def convert_fp16_onnx(input_onnx: str, output_onnx_fp16: str, keep_io_types: bool = True) -> None:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(input_onnx)
    m16 = float16.convert_float_to_float16(m, keep_io_types=keep_io_types)
    onnx.save(m16, output_onnx_fp16)
    print(f"[OK] Wrote FP16 ONNX: {os.path.abspath(output_onnx_fp16)} (keep_io_types={keep_io_types})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vit_b_16"], help="Which architecture to instantiate.")
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint (state_dict or dict with state_dict).")
    ap.add_argument("--output", required=True, help="Output ONNX path (FP32).")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch axis.")
    ap.add_argument("--export-fp16-onnx", default=None, help="If set, also writes a converted FP16 ONNX to this path.")
    ap.add_argument("--fp16-keep-io", action="store_true", help="When converting to FP16, keep input/output types (recommended).")
    args = ap.parse_args()

    model = build_model(args.model, args.num_classes)
    load_checkpoint_flexible(model, args.checkpoint)

    export_onnx(
        model=model,
        output_path=args.output,
        opset=args.opset,
        batch=args.batch,
        input_size=args.input_size,
        dynamic_batch=args.dynamic_batch,
    )

    if args.export_fp16_onnx:
        convert_fp16_onnx(args.output, args.export_fp16_onnx, keep_io_types=args.fp16_keep_io)

if __name__ == "__main__":
    main()
