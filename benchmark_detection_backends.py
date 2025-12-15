#!/usr/bin/env python3
"""
benchmark_detection_backends.py

Benchmark SSD / DETR:
- Torch (.pth)
- ONNXRuntime (.onnx) (CPU or CUDA EP)
- TensorRT Engine (.engine) (FP16 engine built with trtexec)

Metrics:
- Detection mAP via torchmetrics MeanAveragePrecision:
    map, map_50, map_75, mar_100
- Latency: avg ms/img + FPS (measured after warmup)

No class mapping: assumes model labels already match your HF dataset.

HF dataset expectations (per example):
  ex["image"] -> PIL Image or HF Image
  ex["objects"]["bbox"] -> list of [x,y,w,h] (pixels)
  ex["objects"]["category"] -> list[int]

To reduce HF/pyarrow instability:
- use --streaming and small --max-samples
- set --num-workers 0
- optionally set --drop-ucx to remove UCX/HPCX from LD_LIBRARY_PATH.

Example:
  python benchmark_detection_backends.py \
    --task detr --arch detr_resnet50 --source hub \
    --num-classes 91 \
    --checkpoint /app/checkpoints/detr.pth \
    --onnx /app/checkpoints/detr.onnx \
    --engine /app/checkpoints/detr_fp16.engine \
    --dataset detection-datasets/coco --config default --split val \
    --streaming --max-samples 200 \
    --img-size 800 800 \
    --batch-size 1 --warmup-batches 0 --max-batches 200 \
    --score-thr 0.05 --topk 100 \
    --ort-provider cpu \
    --csv /app/checkpoints/bench_det_results.csv
"""

from __future__ import annotations
import argparse, os, time, csv, datetime, ctypes
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image

from torchmetrics.detection.mean_ap import MeanAveragePrecision


def sanitize_ld_library_path(drop_ucx: bool) -> None:
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


def append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


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


def build_torch_model(task: str, arch: str, source: str, num_classes: int) -> torch.nn.Module:
    task = task.lower()
    arch = arch.lower()
    source = source.lower()

    if task == "ssd":
        import torchvision
        if arch == "ssd300_vgg16":
            return torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=num_classes)
        if arch == "ssdlite320_mobilenet_v3_large":
            return torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        raise ValueError("SSD arch must be ssd300_vgg16 or ssdlite320_mobilenet_v3_large")

    if task == "detr":
        if source == "hub":
            m = torch.hub.load("facebookresearch/detr", arch, pretrained=False)
            if hasattr(m, "class_embed") and getattr(m.class_embed, "out_features", None) not in (None, num_classes + 1):
                try:
                    m.class_embed = nn.Linear(m.class_embed.in_features, num_classes + 1)
                except Exception:
                    pass
            return m
        if source == "torchvision":
            import torchvision
            if hasattr(torchvision.models.detection, "detr_resnet50"):
                return torchvision.models.detection.detr_resnet50(weights=None, num_classes=num_classes)
            raise ValueError("torchvision DETR not available; use --source hub")
        raise ValueError("DETR source must be hub or torchvision")

    raise ValueError("task must be ssd or detr")


def to_tensor_normalized(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    import torchvision.transforms.functional as F
    H, W = size_hw
    img = img.convert("RGB")
    img = F.resize(img, [H, W])
    x = F.to_tensor(img)  # float32 0..1
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


def coco_targets_from_example(ex: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    objs = ex.get("objects", {})
    bboxes = objs.get("bbox", [])
    labels = objs.get("category", [])
    b = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    # xywh -> xyxy
    if b.size:
        b[:, 2] = b[:, 0] + b[:, 2]
        b[:, 3] = b[:, 1] + b[:, 3]
    return {
        "boxes": torch.from_numpy(b),
        "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    }


def iter_hf_samples(dataset: str, config: str, split: str, streaming: bool, max_samples: int,
                    token: str, cache_dir: str, revision: str):
    from datasets import load_dataset
    kwargs = {}
    if token:
        kwargs["token"] = token
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision

    ds = load_dataset(dataset, config, split=split, streaming=streaming, **kwargs)
    if streaming:
        for ex in ds.take(max_samples):
            yield ex
    else:
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            yield ex


@torch.no_grad()
def torch_predict(task: str, model: torch.nn.Module, images: torch.Tensor, score_thr: float, topk: int) -> List[Dict[str, torch.Tensor]]:
    task = task.lower()
    if task == "ssd":
        dets = model(list(images))
        out = []
        for d in dets:
            boxes = d["boxes"]; scores = d["scores"]; labels = d["labels"]
            keep = scores >= score_thr
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.shape[0] > topk:
                idx = torch.topk(scores, k=topk).indices
                boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
            out.append({"boxes": boxes, "scores": scores, "labels": labels})
        return out

    if task == "detr":
        out = model(images)
        logits = out["pred_logits"]
        boxes  = out["pred_boxes"]  # cxcywh normalized
        prob = logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)  # ignore "no object"
        B = scores.shape[0]
        preds: List[Dict[str, torch.Tensor]] = []
        for i in range(B):
            s = scores[i]; l = labels[i]; b = boxes[i]
            keep = s >= score_thr
            s, l, b = s[keep], l[keep], b[keep]
            if s.numel() > topk:
                idx = torch.topk(s, k=topk).indices
                s, l, b = s[idx], l[idx], b[idx]
            preds.append({"boxes": b, "scores": s, "labels": l})
        return preds

    raise ValueError("task must be ssd or detr")


def detr_boxes_to_xyxy_pixels(boxes_cxcywh: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = img_hw
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x0 = (cx - 0.5 * w) * W
    y0 = (cy - 0.5 * h) * H
    x1 = (cx + 0.5 * w) * W
    y1 = (cy + 0.5 * h) * H
    return torch.stack([x0, y0, x1, y1], dim=-1)


def softmax_np(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def ort_session(onnx_path: str, provider: str):
    import onnxruntime as ort
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0})]
    else:
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def ort_run(sess, inp: np.ndarray) -> List[np.ndarray]:
    inp_name = sess.get_inputs()[0].name
    return sess.run(None, {inp_name: inp})


def ort_predict(task: str, outs: List[np.ndarray], img_hw: Tuple[int, int], score_thr: float, topk: int) -> List[Dict[str, torch.Tensor]]:
    task = task.lower()
    if task == "ssd":
        boxes, scores, labels = outs  # [B,K,4], [B,K], [B,K]
        B, K, _ = boxes.shape
        preds = []
        for i in range(B):
            s = scores[i]
            l = labels[i].astype(np.int64)
            b = boxes[i]
            keep = s >= score_thr
            s = s[keep]; l = l[keep]; b = b[keep]
            if s.shape[0] > topk:
                idx = np.argsort(-s)[:topk]
                s, l, b = s[idx], l[idx], b[idx]
            preds.append({"boxes": torch.from_numpy(b.astype(np.float32)),
                          "scores": torch.from_numpy(s.astype(np.float32)),
                          "labels": torch.from_numpy(l.astype(np.int64))})
        return preds

    if task == "detr":
        pred_logits, pred_boxes = outs
        prob = softmax_np(pred_logits, axis=-1)
        scores = prob[..., :-1].max(axis=-1)
        labels = prob[..., :-1].argmax(axis=-1)
        B = scores.shape[0]
        preds = []
        for i in range(B):
            s = scores[i]
            l = labels[i].astype(np.int64)
            b = pred_boxes[i]
            keep = s >= score_thr
            s = s[keep]; l = l[keep]; b = b[keep]
            if s.shape[0] > topk:
                idx = np.argsort(-s)[:topk]
                s, l, b = s[idx], l[idx], b[idx]
            b_t = torch.from_numpy(b.astype(np.float32))
            b_xyxy = detr_boxes_to_xyxy_pixels(b_t, img_hw)
            preds.append({"boxes": b_xyxy,
                          "scores": torch.from_numpy(s.astype(np.float32)),
                          "labels": torch.from_numpy(l.astype(np.int64))})
        return preds

    raise ValueError("task must be ssd or detr")


class Cudart:
    def __init__(self):
        names = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        lib = None; last = None
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError as e:
                last = e
        if lib is None:
            raise RuntimeError(f"Could not load libcudart: {last}")

        self.cudaMalloc = lib.cudaMalloc
        self.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudaMalloc.restype = ctypes.c_int

        self.cudaFree = lib.cudaFree
        self.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudaFree.restype = ctypes.c_int

        self.cudaMemcpyAsync = lib.cudaMemcpyAsync
        self.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.cudaMemcpyAsync.restype = ctypes.c_int

        self.cudaStreamCreate = lib.cudaStreamCreate
        self.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cudaStreamCreate.restype = ctypes.c_int

        self.cudaStreamDestroy = lib.cudaStreamDestroy
        self.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cudaStreamDestroy.restype = ctypes.c_int

        self.cudaStreamSynchronize = lib.cudaStreamSynchronize
        self.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cudaStreamSynchronize.restype = ctypes.c_int

        self.cudaMemcpyHostToDevice = 1
        self.cudaMemcpyDeviceToHost = 2


def trt_load_engine(engine_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")
    return engine


def trt_get_io(engine):
    import tensorrt as trt
    ins, outs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            ins.append(name)
        else:
            outs.append(name)
    if len(ins) != 1:
        raise RuntimeError(f"Expected 1 input tensor, got {ins}")
    return ins[0], outs


def trt_dtype_to_np(dt):
    import tensorrt as trt
    if dt == trt.DataType.HALF:
        return np.float16
    if dt == trt.DataType.FLOAT:
        return np.float32
    if dt == trt.DataType.INT32:
        return np.int32
    if dt == trt.DataType.INT64:
        return np.int64
    raise RuntimeError(f"Unsupported TRT dtype: {dt}")


class TrtRunnerMultiOut:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("Failed to create TRT execution context.")
        self.in_name, self.out_names = trt_get_io(engine)

        self.cudart = Cudart()
        self.stream = ctypes.c_void_p()
        err = self.cudart.cudaStreamCreate(ctypes.byref(self.stream))
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

        self.d_in = ctypes.c_void_p()
        self.in_bytes = 0

        self.d_out: Dict[str, ctypes.c_void_p] = {}
        self.out_bytes: Dict[str, int] = {}
        self.out_shape: Dict[str, Tuple[int, ...]] = {}
        self.out_dtype: Dict[str, Any] = {}

    def _ensure_in(self, x_np: np.ndarray):
        self.ctx.set_input_shape(self.in_name, x_np.shape)
        need = x_np.nbytes
        if need > self.in_bytes:
            if self.d_in.value:
                self.cudart.cudaFree(self.d_in)
            tmp = ctypes.c_void_p()
            err = self.cudart.cudaMalloc(ctypes.byref(tmp), need)
            if err != 0:
                raise RuntimeError(f"cudaMalloc input failed: {err}")
            self.d_in = tmp
            self.in_bytes = need

    def _ensure_out(self):
        for name in self.out_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dt = self.engine.get_tensor_dtype(name)
            np_dt = trt_dtype_to_np(dt)
            self.out_shape[name] = shape
            self.out_dtype[name] = np_dt
            need = np.empty(shape, dtype=np_dt).nbytes
            have = self.out_bytes.get(name, 0)
            if need > have:
                if name in self.d_out and self.d_out[name].value:
                    self.cudart.cudaFree(self.d_out[name])
                tmp = ctypes.c_void_p()
                err = self.cudart.cudaMalloc(ctypes.byref(tmp), need)
                if err != 0:
                    raise RuntimeError(f"cudaMalloc output {name} failed: {err}")
                self.d_out[name] = tmp
                self.out_bytes[name] = need

    def infer(self, x_np: np.ndarray) -> List[np.ndarray]:
        self._ensure_in(x_np)
        self._ensure_out()

        err = self.cudart.cudaMemcpyAsync(self.d_in, ctypes.c_void_p(x_np.ctypes.data), x_np.nbytes,
                                         self.cudart.cudaMemcpyHostToDevice, self.stream)
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

        self.ctx.set_tensor_address(self.in_name, int(self.d_in.value))
        for name in self.out_names:
            self.ctx.set_tensor_address(name, int(self.d_out[name].value))

        if not self.ctx.execute_async_v3(int(self.stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 returned False")

        outs: List[np.ndarray] = []
        for name in self.out_names:
            y = np.empty(self.out_shape[name], dtype=self.out_dtype[name])
            err = self.cudart.cudaMemcpyAsync(ctypes.c_void_p(y.ctypes.data), self.d_out[name], y.nbytes,
                                             self.cudart.cudaMemcpyDeviceToHost, self.stream)
            if err != 0:
                raise RuntimeError(f"cudaMemcpyAsync D2H {name} failed: {err}")
            outs.append(y)

        err = self.cudart.cudaStreamSynchronize(self.stream)
        if err != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

        return outs

    def close(self):
        if self.d_in.value:
            self.cudart.cudaFree(self.d_in)
        for name, ptr in self.d_out.items():
            if ptr.value:
                self.cudart.cudaFree(ptr)
        if self.stream.value:
            self.cudart.cudaStreamDestroy(self.stream)


def eval_backend(task: str,
                 backend: str,
                 model: Optional[torch.nn.Module],
                 ort_sess,
                 trt_runner: Optional[TrtRunnerMultiOut],
                 samples_iter,
                 img_hw: Tuple[int, int],
                 device: str,
                 score_thr: float,
                 topk: int,
                 warmup_batches: int,
                 max_batches: int,
                 batch_size: int) -> Dict[str, Any]:

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    t_sum = 0.0
    n_imgs_timed = 0

    batch_imgs: List[torch.Tensor] = []
    batch_tgts: List[Dict[str, torch.Tensor]] = []

    def flush(batch_idx: int):
        nonlocal t_sum, n_imgs_timed
        if not batch_imgs:
            return
        images = torch.stack(batch_imgs, dim=0)
        targets = batch_tgts

        if backend == "torch":
            assert model is not None
            model.eval().to(device)
            images_dev = images.to(device, non_blocking=True)

            if use_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds_raw = torch_predict(task, model, images_dev, score_thr, topk)
            if use_cuda: torch.cuda.synchronize()
            t1 = time.perf_counter()

            preds: List[Dict[str, torch.Tensor]] = []
            for p in preds_raw:
                if task == "detr":
                    b_xyxy = detr_boxes_to_xyxy_pixels(p["boxes"].detach().cpu(), img_hw)
                    preds.append({"boxes": b_xyxy,
                                  "scores": p["scores"].detach().cpu(),
                                  "labels": p["labels"].detach().cpu()})
                else:
                    preds.append({k: v.detach().cpu() for k, v in p.items()})

        elif backend == "ort":
            inp = images.numpy().astype(np.float32, copy=False)
            if use_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = ort_run(ort_sess, inp)
            if use_cuda: torch.cuda.synchronize()
            t1 = time.perf_counter()
            preds = ort_predict(task, outs, img_hw, score_thr, topk)

        elif backend == "trt":
            assert trt_runner is not None
            inp = images.numpy().astype(np.float32, copy=False)
            if use_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = trt_runner.infer(inp)
            if use_cuda: torch.cuda.synchronize()
            t1 = time.perf_counter()
            preds = ort_predict(task, outs, img_hw, score_thr, topk)

        else:
            raise ValueError("backend must be torch|ort|trt")

        if batch_idx >= warmup_batches:
            t_sum += (t1 - t0)
            n_imgs_timed += images.shape[0]

        metric.update(preds, targets)
        batch_imgs.clear()
        batch_tgts.clear()

    max_images = max_batches * batch_size
    for i, ex in enumerate(samples_iter):
        if i >= max_images:
            break
        img = ex["image"]
        if not isinstance(img, Image.Image):
            try:
                img = img.convert("RGB")
            except Exception:
                img = Image.fromarray(np.array(img))
        x = to_tensor_normalized(img, img_hw)
        tgt = coco_targets_from_example(ex)

        batch_imgs.append(x)
        batch_tgts.append(tgt)
        if len(batch_imgs) == batch_size:
            flush(i // batch_size)

    if batch_imgs:
        flush((i // batch_size) + 1)

    res = metric.compute()
    ms = (t_sum / n_imgs_timed) * 1000.0 if n_imgs_timed else float("nan")
    fps = (n_imgs_timed / t_sum) if t_sum > 0 else float("nan")
    return {
        "map": float(res["map"]),
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
        "mar_100": float(res["mar_100"]),
        "avg_ms_per_img": ms,
        "fps": fps,
        "timed_images": int(n_imgs_timed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ssd", "detr"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--source", default="torchvision", choices=["torchvision", "hub"])

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--num-classes", type=int, required=True)

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="val")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max-samples", type=int, default=200)

    ap.add_argument("--img-size", nargs=2, type=int, default=[800, 800])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-batches", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=100)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ort-provider", choices=["cuda", "cpu", "auto"], default="auto")
    ap.add_argument("--skip-ort", action="store_true")
    ap.add_argument("--skip-trt", action="store_true")

    ap.add_argument("--csv", required=True)
    ap.add_argument("--drop-ucx", action="store_true")

    ap.add_argument("--hf-token", default="")
    ap.add_argument("--hf-cache-dir", default="")
    ap.add_argument("--hf-revision", default="")
    args = ap.parse_args()

    sanitize_ld_library_path(args.drop_ucx)

    img_hw = (int(args.img_size[0]), int(args.img_size[1]))
    device = args.device

    model = build_torch_model(args.task, args.arch, args.source, args.num_classes)
    load_checkpoint_flexible(model, args.checkpoint)

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    common = dict(
        timestamp=ts,
        task=args.task,
        arch=args.arch,
        source=args.source,
        num_classes=args.num_classes,
        dataset=f"{args.dataset}/{args.config}:{args.split}",
        streaming=bool(args.streaming),
        max_samples=args.max_samples,
        img_h=img_hw[0],
        img_w=img_hw[1],
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
        score_thr=args.score_thr,
        topk=args.topk,
        checkpoint=os.path.abspath(args.checkpoint),
        onnx=os.path.abspath(args.onnx),
        engine=os.path.abspath(args.engine),
        device=args.device,
        ort_provider=args.ort_provider,
    )

    def make_iter():
        return iter_hf_samples(args.dataset, args.config, args.split, args.streaming, args.max_samples,
                               args.hf_token, args.hf_cache_dir, args.hf_revision)

    # Torch
    try:
        res = eval_backend(args.task, "torch", model, None, None, make_iter(), img_hw, device,
                           args.score_thr, args.topk, args.warmup_batches, args.max_batches, args.batch_size)
        append_csv(args.csv, {**common, "backend": "torch", **res, "error": ""})
        print(f"Torch: mAP={res['map']:.4f} mAP50={res['map_50']:.4f} | {res['avg_ms_per_img']} ms/img | FPS={res['fps']}")
    except Exception as e:
        append_csv(args.csv, {**common, "backend": "torch", "error": f"{type(e).__name__}: {e}"})
        print(f"[ERR] Torch failed: {type(e).__name__}: {e}")

    # ORT
    if args.skip_ort:
        append_csv(args.csv, {**common, "backend": "onnxruntime", "error": "SKIPPED"})
    else:
        try:
            sess = ort_session(args.onnx, args.ort_provider)
            res = eval_backend(args.task, "ort", None, sess, None, make_iter(), img_hw, device,
                               args.score_thr, args.topk, args.warmup_batches, args.max_batches, args.batch_size)
            used = "|".join(sess.get_providers())
            append_csv(args.csv, {**common, "backend": "onnxruntime", **res, "ort_used": used, "error": ""})
            print(f"ORT({used.split('|')[0]}): mAP={res['map']:.4f} mAP50={res['map_50']:.4f} | {res['avg_ms_per_img']} ms/img | FPS={res['fps']}")
        except Exception as e:
            append_csv(args.csv, {**common, "backend": "onnxruntime", "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] ORT failed: {type(e).__name__}: {e}")

    # TRT
    if args.skip_trt:
        append_csv(args.csv, {**common, "backend": "tensorrt_engine", "error": "SKIPPED"})
    else:
        try:
            engine = trt_load_engine(args.engine)
            runner = TrtRunnerMultiOut(engine)
            try:
                res = eval_backend(args.task, "trt", None, None, runner, make_iter(), img_hw, device,
                                   args.score_thr, args.topk, args.warmup_batches, args.max_batches, args.batch_size)
            finally:
                runner.close()
            append_csv(args.csv, {**common, "backend": "tensorrt_engine", **res, "trt_backend": "ctypes-cudart", "error": ""})
            print(f"TRT: mAP={res['map']:.4f} mAP50={res['map_50']:.4f} | {res['avg_ms_per_img']} ms/img | FPS={res['fps']}")
        except Exception as e:
            append_csv(args.csv, {**common, "backend": "tensorrt_engine", "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] TRT failed: {type(e).__name__}: {e}")

    print(f"[OK] CSV appended: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
