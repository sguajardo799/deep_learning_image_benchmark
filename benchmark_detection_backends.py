#!/usr/bin/env python3
"""
export_detection_to_onnx_v2.py

Fixes for torchvision SSD export:
- Avoids downloading VGG16 backbone weights by setting weights_backbone=None.
- Adds --auto-num-classes to infer SSD num_classes from checkpoint head shapes.
  (torchvision SSD "num_classes" INCLUDES background).

Why you hit mismatch:
- Your checkpoint head conv out_channels are multiples of anchors_per_loc.
  Example first head has out=324 and anchors=4 -> num_classes=324/4=81.
  You passed 91, so current model expects out=4*91=364.

Outputs:
- SSD: boxes [B,K,4], scores [B,K], labels [B,K] (padded to --max-dets)
- DETR: pred_logits [B,Q,C+1], pred_boxes [B,Q,4] (cxcywh normalized)

Examples:

SSD (auto classes):
  python export_detection_to_onnx_v2.py \
    --arch ssd300_vgg16 --checkpoint /app/checkpoints/best_model.pth \
    --auto-num-classes \
    --img-size 300 300 --max-dets 100 --dynamic-batch \
    --out /app/checkpoints/ssd_fp32.onnx

SSD (explicit classes, includes background):
  python export_detection_to_onnx_v2.py \
    --arch ssd300_vgg16 --checkpoint /app/checkpoints/best_model.pth \
    --num-classes 81 \
    --img-size 300 300 --max-dets 100 --dynamic-batch \
    --out /app/checkpoints/ssd_fp32.onnx
"""

from __future__ import annotations
import argparse
from collections import OrderedDict
from typing import Dict, Tuple, Optional

import torch


SSD_ANCHORS_PER_LOC = (4, 6, 6, 6, 4, 4)  # torchvision ssd300_vgg16 / ssdlite320 heads


def _extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # could already be a state_dict-like mapping
        # (still accept it)
        return ckpt  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def load_checkpoint_flexible(model: torch.nn.Module, ckpt_path: str, *, strict_shapes: bool = True) -> Tuple[int, int]:
    """
    Returns (missing_count, unexpected_count). If strict_shapes=False, drops mismatched-shape tensors.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    cleaned = OrderedDict()
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    if strict_shapes:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        return len(missing), len(unexpected)

    # drop mismatched shapes
    model_sd = model.state_dict()
    filtered = OrderedDict()
    dropped = 0
    for k, v in cleaned.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped += 1
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if dropped:
        print(f"[WARN] Dropped {dropped} tensors due to shape mismatch (strict_shapes=False).")
    return len(missing), len(unexpected)


def infer_ssd_num_classes_from_ckpt(ckpt_path: str) -> Optional[int]:
    """
    Infer torchvision SSD num_classes (includes background) from classification head conv0 out_channels.
    Uses anchors_per_loc[0]=4 for module_list.0.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    # normalize prefixes like in loader
    keys = list(sd.keys())
    def norm(k: str) -> str:
        for prefix in ("module.", "model.", "net."):
            if k.startswith(prefix):
                return k[len(prefix):]
        return k

    # find a likely key
    want = "head.classification_head.module_list.0.weight"
    for k in keys:
        if norm(k) == want:
            w = sd[k]
            out_ch = int(w.shape[0])
            a0 = SSD_ANCHORS_PER_LOC[0]
            if out_ch % a0 != 0:
                return None
            return out_ch // a0

    # fallback: try any module_list.*.weight and use anchors list if possible
    for k in keys:
        nk = norm(k)
        if nk.startswith("head.classification_head.module_list.") and nk.endswith(".weight"):
            # parse index
            try:
                idx = int(nk.split(".")[4])
            except Exception:
                continue
            if idx < 0 or idx >= len(SSD_ANCHORS_PER_LOC):
                continue
            out_ch = int(sd[k].shape[0])
            a = SSD_ANCHORS_PER_LOC[idx]
            if out_ch % a != 0:
                continue
            return out_ch // a

    return None


def build_model(arch: str, source: str, num_classes: int) -> torch.nn.Module:
    arch = arch.lower()
    source = source.lower()

    if arch.startswith("detr"):
        if source == "hub":
            m = torch.hub.load("facebookresearch/detr", arch, pretrained=False)
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

    # SSD family (torchvision) - avoid backbone downloads:
    import torchvision
    if arch == "ssd300_vgg16":
        return torchvision.models.detection.ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)
    if arch == "ssdlite320_mobilenet_v3_large":
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)

    raise ValueError(f"Unsupported arch: {arch}")


class SSDOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_dets: int = 100):
        super().__init__()
        self.model = model
        self.max_dets = int(max_dets)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        dets = self.model(list(x))
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
    ap.add_argument("--source", default="torchvision", choices=["torchvision", "hub"])
    ap.add_argument("--checkpoint", default="", help=".pth checkpoint path")
    ap.add_argument("--num-classes", type=int, default=0,
                    help="Num classes INCLUDING background for SSD. If 0, requires --auto-num-classes for SSD.")
    ap.add_argument("--auto-num-classes", action="store_true",
                    help="SSD only: infer num_classes from checkpoint head shapes.")
    ap.add_argument("--allow-head-mismatch", action="store_true",
                    help="If set, will drop mismatched head tensors instead of failing (NOT recommended for correct export).")

    ap.add_argument("--img-size", nargs=2, type=int, default=[800, 800])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--max-dets", type=int, default=100)

    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dynamic-batch", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint:
        raise SystemExit("ERROR: --checkpoint is required for your use case (export trained model).")

    arch_is_ssd = args.arch.startswith("ssd") or args.arch.startswith("ssdlite")
    num_classes = args.num_classes

    if arch_is_ssd and (num_classes <= 0):
        if not args.auto_num_classes:
            raise SystemExit("ERROR: SSD needs --num-classes (includes background) or use --auto-num-classes.")
        inferred = infer_ssd_num_classes_from_ckpt(args.checkpoint)
        if inferred is None:
            raise SystemExit("ERROR: Could not infer num_classes from checkpoint. Provide --num-classes explicitly.")
        num_classes = inferred
        print(f"[INFO] Inferred SSD num_classes (includes background) = {num_classes}")

    model = build_model(args.arch, args.source, int(num_classes))

    # load checkpoint
    try:
        load_checkpoint_flexible(model, args.checkpoint, strict_shapes=not args.allow_head_mismatch)
    except RuntimeError as e:
        # if SSD mismatch, give a direct hint
        if arch_is_ssd:
            inferred = infer_ssd_num_classes_from_ckpt(args.checkpoint)
            if inferred is not None:
                print(f"[HINT] Your checkpoint looks like SSD num_classes={inferred} (includes background).")
                print("[HINT] Re-run with: --num-classes", inferred, "or add --auto-num-classes")
        raise

    model.eval()

    H, W = int(args.img_size[0]), int(args.img_size[1])
    dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.randn(args.batch, 3, H, W, dtype=dtype)

    dynamic_axes: Dict[str, Dict[int, str]] = {}
    if args.dynamic_batch:
        dynamic_axes["input"] = {0: "batch"}

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
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )
    print("[OK] Export done.")


if __name__ == "__main__":
    main()