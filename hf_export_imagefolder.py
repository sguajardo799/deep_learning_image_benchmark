#!/usr/bin/env python3
"""
hf_export_imagefolder.py

Export N samples from a Hugging Face dataset split to a local ImageFolder layout:
  <out_dir>/<split>/<class_name>/<00000001>.jpg

Why this helps in your Docker setup:
- Your evaluation crashes inside pyarrow/ucx while reading remote parquet.
- Once images are materialized to disk, benchmarking can use torchvision.ImageFolder
  (pure file I/O) and you avoid the problematic parquet/pyarrow path entirely.

Typical usage (Docker):
  python hf_export_imagefolder.py \
    --dataset timm/mini-imagenet --split validation \
    --num-samples 2000 \
    --out-dir /data/hf_cache/mini-imagenet \
    --streaming \
    --drop-ucx

Then run your benchmark with:
  --data-source imagefolder --data-dir /data/hf_cache/mini-imagenet/validation

Notes:
- Uses streaming by default to avoid downloading the entire split.
- Writes metadata.csv and labels.json for reproducibility.
- If your dataset doesn't have a "label" or "image" column, pass --image-key/--label-key.
"""

from __future__ import annotations
import argparse, os, json, csv
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


def sanitize_ld_library_path(drop_ucx: bool) -> None:
    """Remove UCX/HPCX paths from LD_LIBRARY_PATH (common cause of pyarrow/ucx segfaults)."""
    if not drop_ucx:
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld:
        return
    bad = ("/opt/hpcx/ucx/lib", "/opt/hpcx", "/hpcx/ucx/lib")
    parts = [p for p in ld.split(":") if p and all(b not in p for b in bad)]
    new_ld = ":".join(parts)
    if new_ld != ld:
        os.environ["LD_LIBRARY_PATH"] = new_ld
        print("[INFO] Sanitized LD_LIBRARY_PATH (dropped UCX/HPCX paths).")


def safe_class_name(s: str) -> str:
    s = s.strip().replace("/", "_").replace("\\", "_")
    s = "".join(ch if (ch.isalnum() or ch in "._- ") else "_" for ch in s)
    return s.replace(" ", "_")[:120] or "class"


def get_label_names(ds, label_key: str) -> Optional[list]:
    try:
        feat = ds.features.get(label_key)
        if feat is not None and hasattr(feat, "names") and feat.names:
            return list(feat.names)
    except Exception:
        pass
    return None


def to_pil(img_any: Any) -> Image.Image:
    if isinstance(img_any, Image.Image):
        return img_any
    if isinstance(img_any, dict):
        if "bytes" in img_any and img_any["bytes"] is not None:
            from io import BytesIO
            return Image.open(BytesIO(img_any["bytes"]))
        if "path" in img_any and img_any["path"]:
            return Image.open(img_any["path"])
    return Image.open(img_any)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. timm/mini-imagenet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out-dir", required=True, help="Output root directory")
    ap.add_argument("--num-samples", type=int, required=True, help="How many samples to export from the split")

    ap.add_argument("--streaming", action="store_true", help="Use streaming=True (recommended)")
    ap.add_argument("--cache-dir", default="", help="HF datasets cache dir (optional)")
    ap.add_argument("--revision", default="", help="Dataset revision/commit (optional)")
    ap.add_argument("--token", default="", help="HF token (optional, for gated datasets)")

    ap.add_argument("--image-key", default="image")
    ap.add_argument("--label-key", default="label")

    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=95)

    ap.add_argument("--drop-ucx", action="store_true", help="Sanitize LD_LIBRARY_PATH to reduce pyarrow/ucx crashes")
    ap.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = ap.parse_args()

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")

    sanitize_ld_library_path(args.drop_ucx)

    from datasets import load_dataset

    out_root = Path(args.out_dir).expanduser().resolve()
    split_dir = out_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, Any] = {}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.token:
        load_kwargs["token"] = args.token

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming, **load_kwargs)

    label_names = get_label_names(ds, args.label_key)
    labels_json_path = out_root / "labels.json"
    if label_names:
        labels_json_path.write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    meta_path = out_root / f"metadata_{args.split}.csv"
    meta_exists = meta_path.exists()
    meta_f = meta_path.open("a", newline="", encoding="utf-8")
    meta_w = csv.DictWriter(meta_f, fieldnames=["idx", "relpath", "label", "label_name"])
    if not meta_exists:
        meta_w.writeheader()

    exported = 0
    idx = 0
    try:
        for ex in ds:
            if exported >= args.num_samples:
                break

            if args.image_key not in ex or args.label_key not in ex:
                raise KeyError(f"Example missing '{args.image_key}' or '{args.label_key}'. Keys: {list(ex.keys())}")

            label = int(ex[args.label_key])
            label_name = label_names[label] if label_names and 0 <= label < len(label_names) else str(label)
            label_folder = split_dir / safe_class_name(label_name)
            label_folder.mkdir(parents=True, exist_ok=True)

            fname = f"{idx:08d}.{args.format}"
            out_path = label_folder / fname

            if args.resume and out_path.exists():
                relpath = out_path.relative_to(out_root).as_posix()
                meta_w.writerow({"idx": idx, "relpath": relpath, "label": label, "label_name": label_name})
                exported += 1
                idx += 1
                continue

            img = to_pil(ex[args.image_key]).convert("RGB")
            if args.format == "jpg":
                img.save(out_path, format="JPEG", quality=args.jpeg_quality, optimize=True)
            else:
                img.save(out_path, format="PNG", optimize=True)

            relpath = out_path.relative_to(out_root).as_posix()
            meta_w.writerow({"idx": idx, "relpath": relpath, "label": label, "label_name": label_name})

            exported += 1
            idx += 1

            if exported % 100 == 0:
                print(f"[INFO] Exported {exported}/{args.num_samples}")

    finally:
        meta_f.close()

    print(f"[OK] Exported {exported} samples to: {split_dir}")
    print(f"[OK] Metadata: {meta_path}")
    if labels_json_path.exists():
        print(f"[OK] Labels: {labels_json_path}")


if __name__ == "__main__":
    main()
