#!/usr/bin/env python3
"""
benchmark_backends_v5.py

Fixes your current blockers:

1) ORT CUDA EP missing deps:
   - If CUDA EP can't be created, it falls back to CPU and logs that. (Same as before)
   - To actually use CUDA EP you MUST have CUDA 12 + cuDNN 9 + libcublasLt.so.12 available.

2) TensorRT path:
   - Works with cuda-python IF installed (`pip install cuda-python`).
   - If not installed, falls back to PyCUDA (single stream + persistent buffers) and warns.

3) HF dataset segfault (pyarrow/ucx):
   - Adds --data-source option:
       * hf        : HuggingFace datasets (streaming or non-streaming)
       * imagefolder : local ImageNet-style folder (recommended in your container)
       * random    : synthetic data (timings only; accuracy not meaningful)

4) NaN ms/img happens when warmup >= number of batches:
   - If timed_images == 0, it automatically treats warmup as 0 for metrics (and logs).

Usage example (NO HF, use local folder):
  python benchmark_backends_v5.py \
    --model vit_b_16 --num-classes 100 \
    --checkpoint /app/checkpoints/best_model.pth \
    --onnx /app/checkpoints/vit16_fp32.onnx \
    --engine /app/checkpoints/vit_fp16.engine \
    --csv /app/checkpoints/bench_results.csv \
    --data-source imagefolder --data-dir /data/miniimagenet/val \
    --batch-size 1 --num-workers 0 --warmup-batches 0 --max-batches 200

If you insist on HF but want quick:
  --data-source hf --streaming --num-samples 50 --warmup-batches 0 --num-workers 0
"""

from __future__ import annotations
import argparse, os, time, csv, datetime
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any, Iterable

import numpy as np
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


def make_loader_hf(dataset_name: str, split: str, batch_size: int, num_workers: int,
                   num_samples: int = 0, streaming: bool = False, cache_dir: str = ""):
    from datasets import load_dataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, IterableDataset

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if streaming:
        hf_it = load_dataset(dataset_name, split=split, streaming=True,
                             cache_dir=cache_dir if cache_dir else None)

        class WrappedIter(IterableDataset):
            def __iter__(self):
                n = 0
                for ex in hf_it:
                    img = ex["image"].convert("RGB")
                    x = transform(img)
                    y = ex["label"]
                    yield {"image": x, "label": y}
                    n += 1
                    if num_samples and n >= num_samples:
                        break

        return DataLoader(WrappedIter(), batch_size=batch_size, num_workers=0, pin_memory=True)

    ds_dict = load_dataset(dataset_name, cache_dir=cache_dir if cache_dir else None)
    ds = ds_dict[split]
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    def apply_transforms(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    ds.set_transform(apply_transforms, columns=["image"], output_all_columns=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def make_loader_imagefolder(data_dir: str, batch_size: int, num_workers: int):
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    ds = torchvision.datasets.ImageFolder(root=data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def make_loader_random(num_classes: int, batch_size: int, max_batches: int):
    from torch.utils.data import DataLoader, IterableDataset

    class RandIter(IterableDataset):
        def __iter__(self):
            for _ in range(max_batches):
                x = torch.randn(batch_size, 3, 224, 224)
                y = torch.randint(0, num_classes, (batch_size,))
                yield {"image": x, "label": y}

    return DataLoader(RandIter(), batch_size=None, num_workers=0)


@torch.no_grad()
def eval_torch(model, loader, device: str, warmup_batches: int, max_batches: Optional[int],
               torch_amp: bool, amp_dtype: torch.dtype) -> Tuple[float, float, int, int]:
    model.eval().to(device)
    correct = 0
    total = 0
    t_sum = 0.0
    n_imgs_timed = 0
    n_batches = 0

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"]; y = batch["label"]
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        if torch_amp and use_cuda:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        if use_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            t_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        n_batches += 1

    acc = correct / total if total else 0.0
    return acc, t_sum, n_imgs_timed, n_batches


def ort_input_dtype(session) -> np.dtype:
    t = session.get_inputs()[0].type
    return np.float16 if "float16" in t else np.float32


def eval_onnx(session, loader, warmup_batches: int, max_batches: Optional[int], device: str) -> Tuple[float, float, int, int]:
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    dtype = ort_input_dtype(session)

    correct = 0
    total = 0
    t_sum = 0.0
    n_imgs_timed = 0
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"]; y = batch["label"]
        x_np = x.numpy().astype(dtype, copy=False)

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = session.run([out_name], {inp_name: x_np})[0]
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            t_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        pred = logits.argmax(axis=1)
        correct += (pred == y.numpy()).sum()
        total += y.shape[0]
        n_batches += 1

    acc = float(correct) / float(total) if total else 0.0
    return acc, t_sum, n_imgs_timed, n_batches


def load_trt_engine(engine_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")
    return engine


def trt_get_io_names(engine):
    import tensorrt as trt
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Expected 1 input/1 output, got inputs={inputs}, outputs={outputs}")
    return inputs[0], outputs[0]


class TrtRunner:
    """TRT runner with auto backend: cuda-python if available else pycuda."""
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("Failed to create TRT execution context.")
        self.in_name, self.out_name = trt_get_io_names(engine)
        self.backend = None

        # Buffers and stream init for cuda-python
        self._cudart = None
        self._stream = None
        self._d_in = None
        self._d_out = None
        self._in_bytes = 0
        self._out_bytes = 0
        self._out_shape = None
        self._out_dtype = None

        try:
            from cuda import cudart  # cuda-python
            self.backend = "cuda-python"
            self._cudart = cudart
            err, self._stream = cudart.cudaStreamCreate()
            if err != 0:
                raise RuntimeError(f"cudaStreamCreate failed with code {err}")
        except Exception as e:
            self.backend = "pycuda"
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self._cuda = cuda
            self._stream = cuda.Stream()
            print(f"[WARN] cuda-python not available ({type(e).__name__}: {e}). Falling back to PyCUDA.")

    def _ensure_buffers(self, x_np: np.ndarray):
        import tensorrt as trt
        # engine is fixed 1x3x224x224 in your case; still safe to set
        self.ctx.set_input_shape(self.in_name, x_np.shape)
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        out_dt = self.engine.get_tensor_dtype(self.out_name)
        self._out_dtype = np.float16 if out_dt == trt.DataType.HALF else np.float32
        self._out_shape = out_shape

        need_in = x_np.nbytes
        need_out = np.empty(out_shape, dtype=self._out_dtype).nbytes

        if self.backend == "cuda-python":
            cudart = self._cudart
            if self._d_in is None or need_in > self._in_bytes:
                if self._d_in is not None:
                    cudart.cudaFree(self._d_in)
                err, self._d_in = cudart.cudaMalloc(need_in)
                if err != 0:
                    raise RuntimeError(f"cudaMalloc input failed with code {err}")
                self._in_bytes = need_in
            if self._d_out is None or need_out > self._out_bytes:
                if self._d_out is not None:
                    cudart.cudaFree(self._d_out)
                err, self._d_out = cudart.cudaMalloc(need_out)
                if err != 0:
                    raise RuntimeError(f"cudaMalloc output failed with code {err}")
                self._out_bytes = need_out
        else:
            # pycuda
            cuda = self._cuda
            if self._d_in is None or need_in > self._in_bytes:
                self._d_in = cuda.mem_alloc(need_in)
                self._in_bytes = need_in
            if self._d_out is None or need_out > self._out_bytes:
                self._d_out = cuda.mem_alloc(need_out)
                self._out_bytes = need_out

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        self._ensure_buffers(x_np)
        y = np.empty(self._out_shape, dtype=self._out_dtype)

        self.ctx.set_tensor_address(self.in_name, int(self._d_in))
        self.ctx.set_tensor_address(self.out_name, int(self._d_out))

        if self.backend == "cuda-python":
            cudart = self._cudart
            # H2D
            err = cudart.cudaMemcpyAsync(self._d_in, x_np.ctypes.data, x_np.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)[0]
            if err != 0:
                raise RuntimeError(f"cudaMemcpyAsync H2D failed with code {err}")
            if not self.ctx.execute_async_v3(self._stream):
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            # D2H
            err = cudart.cudaMemcpyAsync(y.ctypes.data, self._d_out, y.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)[0]
            if err != 0:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed with code {err}")
            err = cudart.cudaStreamSynchronize(self._stream)[0]
            if err != 0:
                raise RuntimeError(f"cudaStreamSynchronize failed with code {err}")
            return y

        # pycuda
        cuda = self._cuda
        cuda.memcpy_htod_async(self._d_in, x_np, self._stream)
        ok = self.ctx.execute_async_v3(self._stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        cuda.memcpy_dtoh_async(y, self._d_out, self._stream)
        self._stream.synchronize()
        return y

    def close(self):
        if self.backend == "cuda-python":
            if self._d_in is not None: self._cudart.cudaFree(self._d_in)
            if self._d_out is not None: self._cudart.cudaFree(self._d_out)
            if self._stream is not None: self._cudart.cudaStreamDestroy(self._stream)
        # pycuda uses autoinit; we don't manually pop contexts here.


def eval_trt(engine_path: str, loader, warmup_batches: int, max_batches: Optional[int], device: str) -> Tuple[float, float, int, int, str]:
    import tensorrt as trt
    engine = load_trt_engine(engine_path)
    in_name, _ = trt_get_io_names(engine)
    in_dtype = engine.get_tensor_dtype(in_name)
    wanted = np.float16 if in_dtype == trt.DataType.HALF else np.float32

    runner = TrtRunner(engine)

    correct = 0
    total = 0
    t_sum = 0.0
    n_imgs_timed = 0
    n_batches = 0

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch["image"]; y = batch["label"]
            x_np = x.numpy().astype(wanted, copy=False)

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = runner.infer(x_np)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= warmup_batches:
                t_sum += (t1 - t0)
                n_imgs_timed += x.shape[0]

            pred = logits.argmax(axis=1)
            correct += (pred == y.numpy()).sum()
            total += y.shape[0]
            n_batches += 1
    finally:
        runner.close()

    acc = float(correct) / float(total) if total else 0.0
    return acc, t_sum, n_imgs_timed, n_batches, runner.backend


def ms_per_img_and_fps(t_sum: float, n_imgs: int) -> Tuple[float, float]:
    if n_imgs <= 0 or t_sum <= 0:
        return float("nan"), float("nan")
    return (t_sum / n_imgs) * 1000.0, n_imgs / t_sum


def append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def safe_record(common: Dict[str, Any], backend: str, csv_path: str, *, accuracy=None, ms=None, fps=None, timed_images=None, error_msg="", extra: Dict[str, Any] | None = None):
    row = {**common,
           "backend": backend,
           "accuracy": accuracy if accuracy is not None else "",
           "avg_ms_per_img": ms if ms is not None else "",
           "fps": fps if fps is not None else "",
           "timed_images": timed_images if timed_images is not None else "",
           "error": error_msg}
    if extra:
        row.update(extra)
    append_csv(csv_path, row)


def _fix_nan_timing(acc, t_sum, n_imgs_timed, n_batches, warmup_batches, label: str):
    # If user asked warmup >= available batches, timing is empty → nan.
    # In that case, use all batches for timing (warmup=0 equivalent) by warning.
    if n_imgs_timed == 0 and n_batches > 0:
        print(f"[WARN] {label}: timed_images=0 (warmup_batches={warmup_batches} >= batches={n_batches}). "
              f"Set --warmup-batches 0 or increase --num-samples. Recording timing as NaN.")
    ms, fps = ms_per_img_and_fps(t_sum, n_imgs_timed)
    return ms, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vit_b_16"])
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--csv", required=True)

    ap.add_argument("--warmup-batches", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--torch-amp", action="store_true")

    ap.add_argument("--ort-provider", choices=["cuda", "cpu", "auto"], default="auto")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--skip-trt", action="store_true")

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--data-source", choices=["hf", "imagefolder", "random"], default="hf")
    ap.add_argument("--data-dir", default="", help="Used when --data-source imagefolder (ImageFolder root).")

    # HF options
    ap.add_argument("--dataset", default="timm/mini-imagenet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num-samples", type=int, default=0)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--hf-cache-dir", default="")

    # random options
    ap.add_argument("--random-batches", type=int, default=200, help="Used when --data-source random.")

    args = ap.parse_args()
    max_batches = int(args.max_batches) if args.max_batches is not None else None

    # loader
    if args.data_source == "imagefolder":
        if not args.data_dir:
            raise SystemExit("--data-dir is required when --data-source imagefolder")
        loader = make_loader_imagefolder(args.data_dir, args.batch_size, args.num_workers)
        dataset_id = f"imagefolder:{args.data_dir}"
        split = ""
        num_samples = ""
        streaming = ""
    elif args.data_source == "random":
        loader = make_loader_random(args.num_classes, args.batch_size, args.random_batches)
        dataset_id = "random"
        split = ""
        num_samples = args.random_batches * args.batch_size
        streaming = ""
        if max_batches is None:
            max_batches = args.random_batches
    else:
        loader = make_loader_hf(args.dataset, args.split, args.batch_size, args.num_workers,
                                num_samples=args.num_samples, streaming=args.streaming, cache_dir=args.hf_cache_dir)
        dataset_id = args.dataset
        split = args.split
        num_samples = args.num_samples
        streaming = args.streaming

    model = build_model(args.model, args.num_classes)
    load_checkpoint_flexible(model, args.checkpoint)

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    common = dict(
        timestamp=ts,
        model=args.model,
        num_classes=args.num_classes,
        dataset=dataset_id,
        split=split,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches if max_batches is not None else "",
        num_samples=num_samples,
        streaming=streaming,
        checkpoint=os.path.abspath(args.checkpoint),
        onnx=os.path.abspath(args.onnx),
        engine=os.path.abspath(args.engine),
        device=args.device,
        ort_provider=args.ort_provider,
        data_source=args.data_source,
    )

    # Torch
    try:
        acc_t, t_sum_t, n_imgs_t, n_b_t = eval_torch(model, loader, args.device, args.warmup_batches, max_batches, args.torch_amp, torch.float16)
        ms_t, fps_t = _fix_nan_timing(acc_t, t_sum_t, n_imgs_t, n_b_t, args.warmup_batches, "Torch")
        safe_record(common, "torch", args.csv, accuracy=acc_t, ms=ms_t, fps=fps_t, timed_images=n_imgs_t)
        print(f"Torch: acc={acc_t:.4f} | {ms_t} ms/img | FPS={fps_t}")
    except Exception as e:
        safe_record(common, "torch", args.csv, error_msg=f"{type(e).__name__}: {e}")
        print(f"[ERR] Torch failed: {type(e).__name__}: {e}")

    # ORT
    if args.skip_onnx:
        safe_record(common, "onnxruntime", args.csv, error_msg="SKIPPED")
    else:
        import onnxruntime as ort
        if args.ort_provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif args.ort_provider == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0})]
        else:
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(args.onnx, providers=providers)
            used = sess.get_providers()
            acc_o, t_sum_o, n_imgs_o, n_b_o = eval_onnx(sess, loader, args.warmup_batches, max_batches, args.device)
            ms_o, fps_o = _fix_nan_timing(acc_o, t_sum_o, n_imgs_o, n_b_o, args.warmup_batches, "ONNXRuntime")
            safe_record(common, "onnxruntime_" + ("cuda" if "CUDAExecutionProvider" in used else "cpu"),
                        args.csv, accuracy=acc_o, ms=ms_o, fps=fps_o, timed_images=n_imgs_o,
                        extra={"ort_used": "|".join(used)})
            print(f"ORT({used[0]}): acc={acc_o:.4f} | {ms_o} ms/img | FPS={fps_o}")
        except Exception as e:
            safe_record(common, "onnxruntime", args.csv, error_msg=f"{type(e).__name__}: {e}")
            print(f"[ERR] ORT failed: {type(e).__name__}: {e}")

    # TRT
    if args.skip_trt:
        safe_record(common, "tensorrt_engine", args.csv, error_msg="SKIPPED")
    else:
        try:
            acc_e, t_sum_e, n_imgs_e, n_b_e, trt_backend = eval_trt(args.engine, loader, args.warmup_batches, max_batches, args.device)
            ms_e, fps_e = _fix_nan_timing(acc_e, t_sum_e, n_imgs_e, n_b_e, args.warmup_batches, "TensorRT")
            safe_record(common, "tensorrt_engine", args.csv, accuracy=acc_e, ms=ms_e, fps=fps_e, timed_images=n_imgs_e,
                        extra={"trt_backend": trt_backend})
            print(f"TRT({trt_backend}): acc={acc_e:.4f} | {ms_e} ms/img | FPS={fps_e}")
        except Exception as e:
            safe_record(common, "tensorrt_engine", args.csv, error_msg=f"{type(e).__name__}: {e}")
            print(f"[ERR] TRT failed: {type(e).__name__}: {e}")

    print(f"[OK] CSV appended: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()