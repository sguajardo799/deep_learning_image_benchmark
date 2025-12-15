#!/usr/bin/env python3
"""
benchmark_backends.py

Given:
  - a Torch checkpoint (.pth) for resnet18 or vit_b_16,
  - an ONNX file (benchmarked with ONNX Runtime CUDA EP),
  - a TensorRT engine (.engine) built in FP16,

Compute:
  - top-1 accuracy,
  - average inference latency (ms / image),
  - FPS,

and append results to a CSV.

Notes:
  - Default dataset is HuggingFace "timm/mini-imagenet" (validation split), as in your notebook.
  - TensorRT engine execution uses the TensorRT Python API + CUDA (prefers cuda-python; falls back to pycuda).
"""
from __future__ import annotations
import argparse, os, time, csv, datetime
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
import torchvision

def _sync_if_cuda(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def build_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in ("vit_b_16", "vit16", "vit_b16", "vit-b-16"):
        model = torchvision.models.vit_b_16(weights=None)
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

def make_loader(dataset_name: str, split: str, batch_size: int, num_workers: int):
    from datasets import load_dataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    ds_dict = load_dataset(dataset_name)
    ds = ds_dict[split]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def apply_transforms(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    ds.set_transform(apply_transforms, columns=["image"], output_all_columns=True)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

@torch.no_grad()
def eval_torch(
    model: torch.nn.Module,
    loader,
    device: str,
    warmup_batches: int,
    max_batches: Optional[int],
    torch_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[float, float, int]:
    model.eval().to(device)
    correct = 0
    total = 0
    infer_time_sum = 0.0
    n_imgs_timed = 0

    use_cuda = (isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available())

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch["image"] if isinstance(batch, dict) else batch[0]
        y = batch["label"] if isinstance(batch, dict) else batch[1]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if torch_amp and use_cuda:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            infer_time_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    acc = correct / total if total else 0.0
    return acc, infer_time_sum, n_imgs_timed

def ort_input_dtype(session) -> np.dtype:
    t = session.get_inputs()[0].type
    # examples: 'tensor(float)', 'tensor(float16)'
    if "float16" in t:
        return np.float16
    return np.float32

def eval_onnx(
    session,
    loader,
    warmup_batches: int,
    max_batches: Optional[int],
    device: str,
) -> Tuple[float, float, int]:
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    dtype = ort_input_dtype(session)

    correct = 0
    total = 0
    infer_time_sum = 0.0
    n_imgs_timed = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch["image"] if isinstance(batch, dict) else batch[0]
        y = batch["label"] if isinstance(batch, dict) else batch[1]

        x_np = x.detach().cpu().numpy().astype(dtype, copy=False)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        logits = session.run([out_name], {inp_name: x_np})[0]
        _sync_if_cuda(device)
        t1 = time.perf_counter()

        if i >= warmup_batches:
            infer_time_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        preds = logits.argmax(axis=1)
        correct += (preds == y.cpu().numpy()).sum()
        total += y.shape[0]

    acc = float(correct) / float(total) if total else 0.0
    return acc, infer_time_sum, n_imgs_timed

def load_trt_engine(engine_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")
    return engine

def trt_infer_fn(engine, device_id: int = 0):
    """
    Returns a callable: (x_np_float32_or_float16) -> logits_np
    Prefers cuda-python. Falls back to pycuda if needed.
    """
    # --- try cuda-python first ---
    try:
        from cuda import cudart
        import tensorrt as trt

        ctx = engine.create_execution_context()
        if ctx is None:
            raise RuntimeError("Failed to create TRT execution context.")

        # assume single input / single output classification
        in_idx = 0
        out_idx = 1 if engine.num_io_tensors >= 2 else 0

        in_name = engine.get_tensor_name(in_idx)
        out_name = engine.get_tensor_name(out_idx)

        def infer(x_np: np.ndarray) -> np.ndarray:
            assert x_np.ndim == 4, f"expected NCHW, got {x_np.shape}"
            # set shapes for dynamic
            ctx.set_input_shape(in_name, x_np.shape)

            # allocate output
            out_shape = tuple(ctx.get_tensor_shape(out_name))
            out_dtype = np.float16 if engine.get_tensor_dtype(out_name) == trt.DataType.HALF else np.float32
            y = np.empty(out_shape, dtype=out_dtype)

            # malloc
            err, d_in = cudart.cudaMalloc(x_np.nbytes)
            if err != 0:
                raise RuntimeError(f"cudaMalloc input failed with code {err}")
            err, d_out = cudart.cudaMalloc(y.nbytes)
            if err != 0:
                raise RuntimeError(f"cudaMalloc output failed with code {err}")

            # create stream
            err, stream = cudart.cudaStreamCreate()
            if err != 0:
                raise RuntimeError(f"cudaStreamCreate failed with code {err}")

            try:
                # H2D
                err = cudart.cudaMemcpyAsync(d_in, x_np.ctypes.data, x_np.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)[0]
                if err != 0:
                    raise RuntimeError(f"cudaMemcpyAsync H2D failed with code {err}")

                # bind pointers
                ctx.set_tensor_address(in_name, int(d_in))
                ctx.set_tensor_address(out_name, int(d_out))

                # execute
                if not ctx.execute_async_v3(stream):
                    raise RuntimeError("TensorRT execution failed.")

                # D2H
                err = cudart.cudaMemcpyAsync(y.ctypes.data, d_out, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)[0]
                if err != 0:
                    raise RuntimeError(f"cudaMemcpyAsync D2H failed with code {err}")

                # sync
                err = cudart.cudaStreamSynchronize(stream)[0]
                if err != 0:
                    raise RuntimeError(f"cudaStreamSynchronize failed with code {err}")
            finally:
                cudart.cudaStreamDestroy(stream)
                cudart.cudaFree(d_in)
                cudart.cudaFree(d_out)

            return y

        return infer

    except Exception as e_cuda:
        print(f"[INFO] cuda-python path not available / failed ({type(e_cuda).__name__}: {e_cuda}). Trying pycuda...")

    # --- fallback: pycuda ---
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("Failed to create TRT execution context.")

    in_idx = 0
    out_idx = 1 if engine.num_io_tensors >= 2 else 0
    in_name = engine.get_tensor_name(in_idx)
    out_name = engine.get_tensor_name(out_idx)

    def infer(x_np: np.ndarray) -> np.ndarray:
        assert x_np.ndim == 4, f"expected NCHW, got {x_np.shape}"
        ctx.set_input_shape(in_name, x_np.shape)

        out_shape = tuple(ctx.get_tensor_shape(out_name))
        out_dtype = np.float16 if engine.get_tensor_dtype(out_name) == trt.DataType.HALF else np.float32
        y = np.empty(out_shape, dtype=out_dtype)

        d_in = cuda.mem_alloc(x_np.nbytes)
        d_out = cuda.mem_alloc(y.nbytes)
        stream = cuda.Stream()

        # H2D
        cuda.memcpy_htod_async(d_in, x_np, stream)

        ctx.set_tensor_address(in_name, int(d_in))
        ctx.set_tensor_address(out_name, int(d_out))

        if not ctx.execute_async_v3(stream.handle):
            raise RuntimeError("TensorRT execution failed.")

        # D2H
        cuda.memcpy_dtoh_async(y, d_out, stream)
        stream.synchronize()

        d_in.free()
        d_out.free()
        return y

    return infer

def eval_trt(
    engine_path: str,
    loader,
    warmup_batches: int,
    max_batches: Optional[int],
    device: str,
) -> Tuple[float, float, int]:
    engine = load_trt_engine(engine_path)
    infer = trt_infer_fn(engine)

    correct = 0
    total = 0
    infer_time_sum = 0.0
    n_imgs_timed = 0

    # decide input dtype (engine input)
    try:
        import tensorrt as trt
        in_name = engine.get_tensor_name(0)
        in_dtype = engine.get_tensor_dtype(in_name)
        wanted = np.float16 if in_dtype == trt.DataType.HALF else np.float32
    except Exception:
        wanted = np.float32

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        x = batch["image"] if isinstance(batch, dict) else batch[0]
        y = batch["label"] if isinstance(batch, dict) else batch[1]

        x_np = x.detach().cpu().numpy().astype(wanted, copy=False)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        logits = infer(x_np)
        _sync_if_cuda(device)
        t1 = time.perf_counter()

        if i >= warmup_batches:
            infer_time_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        preds = logits.argmax(axis=1)
        correct += (preds == y.cpu().numpy()).sum()
        total += y.shape[0]

    acc = float(correct) / float(total) if total else 0.0
    return acc, infer_time_sum, n_imgs_timed

def ms_per_img_and_fps(t_sum: float, n_imgs: int) -> Tuple[float, float]:
    if n_imgs <= 0 or t_sum <= 0:
        return float("nan"), float("nan")
    ms_per_img = (t_sum / n_imgs) * 1000.0
    fps = n_imgs / t_sum
    return ms_per_img, fps

def append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vit_b_16"])
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True, help="TensorRT .engine (FP16) path.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="timm/mini-imagenet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--warmup-batches", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick tests.")
    ap.add_argument("--torch-amp", action="store_true", help="Use torch autocast (fp16) during Torch eval.")
    ap.add_argument("--csv", required=True, help="Output CSV path (appended).")
    args = ap.parse_args()

    max_batches = int(args.max_batches) if args.max_batches is not None else None
    loader = make_loader(args.dataset, args.split, args.batch_size, args.num_workers)

    # ----- Torch -----
    model = build_model(args.model, args.num_classes)
    load_checkpoint_flexible(model, args.checkpoint)

    acc_t, t_sum_t, n_imgs_t = eval_torch(
        model=model,
        loader=loader,
        device=args.device,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches,
        torch_amp=args.torch_amp,
        amp_dtype=torch.float16,
    )
    ms_t, fps_t = ms_per_img_and_fps(t_sum_t, n_imgs_t)

    # ----- ONNX Runtime (CUDA EP) -----
    import onnxruntime as ort
    providers = [("CUDAExecutionProvider", {"device_id": 0})]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    acc_o, t_sum_o, n_imgs_o = eval_onnx(
        session=sess,
        loader=loader,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches,
        device=args.device,
    )
    ms_o, fps_o = ms_per_img_and_fps(t_sum_o, n_imgs_o)

    # ----- TensorRT engine -----
    acc_e, t_sum_e, n_imgs_e = eval_trt(
        engine_path=args.engine,
        loader=loader,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches,
        device=args.device,
    )
    ms_e, fps_e = ms_per_img_and_fps(t_sum_e, n_imgs_e)

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    common = dict(
        timestamp=ts,
        model=args.model,
        num_classes=args.num_classes,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches if max_batches is not None else "",
        checkpoint=os.path.abspath(args.checkpoint),
        onnx=os.path.abspath(args.onnx),
        engine=os.path.abspath(args.engine),
    )

    # append three rows
    append_csv(args.csv, {**common, "backend": "torch", "accuracy": acc_t, "avg_ms_per_img": ms_t, "fps": fps_t, "timed_images": n_imgs_t})
    append_csv(args.csv, {**common, "backend": "onnxruntime_cuda", "accuracy": acc_o, "avg_ms_per_img": ms_o, "fps": fps_o, "timed_images": n_imgs_o})
    append_csv(args.csv, {**common, "backend": "tensorrt_engine", "accuracy": acc_e, "avg_ms_per_img": ms_e, "fps": fps_e, "timed_images": n_imgs_e})

    print("---- RESULTS ----")
    print(f"Torch:          acc={acc_t:.4f} | {ms_t:.3f} ms/img | FPS={fps_t:.2f} | timed_imgs={n_imgs_t}")
    print(f"ONNXRuntime:    acc={acc_o:.4f} | {ms_o:.3f} ms/img | FPS={fps_o:.2f} | timed_imgs={n_imgs_o}")
    print(f"TensorRT eng.:  acc={acc_e:.4f} | {ms_e:.3f} ms/img | FPS={fps_e:.2f} | timed_imgs={n_imgs_e}")
    print(f"[OK] Appended CSV: {os.path.abspath(args.csv)}")

if __name__ == "__main__":
    main()
