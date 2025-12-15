#!/usr/bin/env python3
"""benchmark_backends_v9.py

Adds precision/recall/F1 (micro|macro|weighted) to the CSV + prints.
Works with ImageFolder subset (tuple batches) and keeps TRT via ctypes+libcudart.

Example:
python /app/benchmark_backends_v9.py \
  --model vit_b_16 --num-classes 100 \
  --checkpoint /app/checkpoints/best_model.pth \
  --onnx /app/checkpoints/vit16_fp32.onnx \
  --engine /app/checkpoints/vit_fp16.engine \
  --csv /app/checkpoints/bench_results.csv \
  --data-dir /data/mini_imagenet_subset/validation \
  --batch-size 1 --num-workers 0 --warmup-batches 0 --max-batches 200 \
  --avg macro \
  --ort-provider cpu
"""

from __future__ import annotations
import argparse, os, time, csv, datetime, ctypes
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
import torchvision


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name in ("vit_b_16", "vit16", "vit_b16", "vit-b-16"):
        m = torchvision.models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise ValueError(f"Unsupported --model '{name}'.")


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


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise TypeError(f"Batch tuple/list must have >=2 elements, got {len(batch)}")
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


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


def safe_record(common: Dict[str, Any], backend: str, csv_path: str, *, accuracy=None,
                precision=None, recall=None, f1=None, avg=None,
                ms=None, fps=None, timed_images=None, error_msg="",
                extra: Dict[str, Any] | None = None):
    row = {**common,
           "backend": backend,
           "avg": avg if avg is not None else "",
           "accuracy": accuracy if accuracy is not None else "",
           "precision": precision if precision is not None else "",
           "recall": recall if recall is not None else "",
           "f1": f1 if f1 is not None else "",
           "avg_ms_per_img": ms if ms is not None else "",
           "fps": fps if fps is not None else "",
           "timed_images": timed_images if timed_images is not None else "",
           "error": error_msg}
    if extra:
        row.update(extra)
    append_csv(csv_path, row)


def update_confusion(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    y_true = y_true.astype(np.int64).ravel()
    y_pred = y_pred.astype(np.int64).ravel()
    m = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_true = y_true[m]; y_pred = y_pred[m]
    idx = y_true * num_classes + y_pred
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    cm += binc.reshape(num_classes, num_classes)


def prf_from_confusion(cm: np.ndarray, avg: str) -> Tuple[float, float, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    if avg == "micro":
        TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return float(prec), float(rec), float(f1)

    prec_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec_c  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_c   = np.divide(2 * prec_c * rec_c, prec_c + rec_c, out=np.zeros_like(tp), where=(prec_c + rec_c) > 0)

    if avg == "macro":
        m = support > 0
        if not np.any(m):
            return 0.0, 0.0, 0.0
        return float(prec_c[m].mean()), float(rec_c[m].mean()), float(f1_c[m].mean())

    if avg == "weighted":
        tot = support.sum()
        if tot <= 0:
            return 0.0, 0.0, 0.0
        w = support / tot
        return float((prec_c * w).sum()), float((rec_c * w).sum()), float((f1_c * w).sum())

    raise ValueError("avg must be micro|macro|weighted")


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


@torch.no_grad()
def eval_torch(model, loader, device: str, warmup_batches: int, max_batches: Optional[int],
               torch_amp: bool, amp_dtype: torch.dtype, num_classes: int):
    model.eval().to(device)
    correct = 0; total = 0
    t_sum = 0.0; n_imgs_timed = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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
        update_confusion(cm, y.detach().cpu().numpy(), pred.detach().cpu().numpy(), num_classes)

    acc = correct / total if total else 0.0
    return acc, t_sum, n_imgs_timed, cm


def ort_input_dtype(session) -> np.dtype:
    t = session.get_inputs()[0].type
    return np.float16 if "float16" in t else np.float32


def eval_onnx(session, loader, warmup_batches: int, max_batches: Optional[int], device: str, num_classes: int):
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    dtype = ort_input_dtype(session)

    correct = 0; total = 0
    t_sum = 0.0; n_imgs_timed = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = unpack_batch(batch)
        x_np = x.numpy().astype(dtype, copy=False)

        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = session.run([out_name], {inp_name: x_np})[0]
        if use_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            t_sum += (t1 - t0)
            n_imgs_timed += x.shape[0]

        pred = logits.argmax(axis=1)
        y_np = y.numpy()
        correct += (pred == y_np).sum()
        total += y_np.shape[0]
        update_confusion(cm, y_np, pred, num_classes)

    acc = float(correct) / float(total) if total else 0.0
    return acc, t_sum, n_imgs_timed, cm


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


class TrtRunnerCtypes:
    def __init__(self, engine):
        import tensorrt as trt
        self.trt = trt
        self.engine = engine
        self.ctx = engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("Failed to create TRT execution context.")
        self.in_name, self.out_name = trt_get_io_names(engine)

        self.cudart = Cudart()
        self.stream = ctypes.c_void_p()
        err = self.cudart.cudaStreamCreate(ctypes.byref(self.stream))
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

        self.d_in = ctypes.c_void_p()
        self.d_out = ctypes.c_void_p()
        self.in_bytes = 0
        self.out_bytes = 0
        self.out_shape = None
        self.out_dtype = None

    def _ensure_buffers(self, x_np: np.ndarray):
        self.ctx.set_input_shape(self.in_name, x_np.shape)
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        out_dt = self.engine.get_tensor_dtype(self.out_name)
        self.out_dtype = np.float16 if out_dt == self.trt.DataType.HALF else np.float32
        self.out_shape = out_shape

        need_in = x_np.nbytes
        need_out = np.empty(out_shape, dtype=self.out_dtype).nbytes

        if need_in > self.in_bytes:
            if self.d_in.value:
                self.cudart.cudaFree(self.d_in)
            tmp = ctypes.c_void_p()
            err = self.cudart.cudaMalloc(ctypes.byref(tmp), need_in)
            if err != 0:
                raise RuntimeError(f"cudaMalloc input failed: {err}")
            self.d_in = tmp
            self.in_bytes = need_in

        if need_out > self.out_bytes:
            if self.d_out.value:
                self.cudart.cudaFree(self.d_out)
            tmp = ctypes.c_void_p()
            err = self.cudart.cudaMalloc(ctypes.byref(tmp), need_out)
            if err != 0:
                raise RuntimeError(f"cudaMalloc output failed: {err}")
            self.d_out = tmp
            self.out_bytes = need_out

    def infer(self, x_np: np.ndarray) -> np.ndarray:
        self._ensure_buffers(x_np)
        y = np.empty(self.out_shape, dtype=self.out_dtype)

        err = self.cudart.cudaMemcpyAsync(self.d_in, ctypes.c_void_p(x_np.ctypes.data), x_np.nbytes,
                                         self.cudaMemcpyHostToDevice, self.stream)
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

        self.ctx.set_tensor_address(self.in_name, int(self.d_in.value))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out.value))

        if not self.ctx.execute_async_v3(int(self.stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 returned False")

        err = self.cudart.cudaMemcpyAsync(ctypes.c_void_p(y.ctypes.data), self.d_out, y.nbytes,
                                         self.cudaMemcpyDeviceToHost, self.stream)
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2H failed: {err}")

        err = self.cudart.cudaStreamSynchronize(self.stream)
        if err != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

        return y

    @property
    def cudaMemcpyHostToDevice(self):  # for shorthand above
        return self.cudart.cudaMemcpyHostToDevice

    @property
    def cudaMemcpyDeviceToHost(self):
        return self.cudart.cudaMemcpyDeviceToHost

    def close(self):
        if self.d_in.value:
            self.cudart.cudaFree(self.d_in)
        if self.d_out.value:
            self.cudart.cudaFree(self.d_out)
        if self.stream.value:
            self.cudart.cudaStreamDestroy(self.stream)


def eval_trt(engine_path: str, loader, warmup_batches: int, max_batches: Optional[int], device: str, num_classes: int):
    import tensorrt as trt
    engine = load_trt_engine(engine_path)
    in_name, _ = trt_get_io_names(engine)
    in_dtype = engine.get_tensor_dtype(in_name)
    wanted = np.float16 if in_dtype == trt.DataType.HALF else np.float32

    runner = TrtRunnerCtypes(engine)

    correct = 0; total = 0
    t_sum = 0.0; n_imgs_timed = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = unpack_batch(batch)
            x_np = x.numpy().astype(wanted, copy=False)

            if use_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = runner.infer(x_np)
            if use_cuda: torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= warmup_batches:
                t_sum += (t1 - t0)
                n_imgs_timed += x.shape[0]

            pred = logits.argmax(axis=1)
            y_np = y.numpy()
            correct += (pred == y_np).sum()
            total += y_np.shape[0]
            update_confusion(cm, y_np, pred, num_classes)
    finally:
        runner.close()

    acc = float(correct) / float(total) if total else 0.0
    return acc, t_sum, n_imgs_timed, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vit_b_16"])
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", required=True)

    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup-batches", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--torch-amp", action="store_true")

    ap.add_argument("--avg", choices=["micro", "macro", "weighted"], default="macro")

    ap.add_argument("--ort-provider", choices=["cuda", "cpu", "auto"], default="auto")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--skip-trt", action="store_true")

    args = ap.parse_args()
    max_batches = int(args.max_batches) if args.max_batches is not None else None

    loader = make_loader_imagefolder(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(args.model, args.num_classes)
    load_checkpoint_flexible(model, args.checkpoint)

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    common = dict(
        timestamp=ts,
        model=args.model,
        num_classes=args.num_classes,
        dataset=f"imagefolder:{os.path.abspath(args.data_dir)}",
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        max_batches=max_batches if max_batches is not None else "",
        checkpoint=os.path.abspath(args.checkpoint),
        onnx=os.path.abspath(args.onnx),
        engine=os.path.abspath(args.engine),
        device=args.device,
        ort_provider=args.ort_provider,
    )

    try:
        acc_t, t_sum_t, n_imgs_t, cm_t = eval_torch(model, loader, args.device, args.warmup_batches, max_batches, args.torch_amp, torch.float16, args.num_classes)
        p_t, r_t, f_t = prf_from_confusion(cm_t, args.avg)
        ms_t, fps_t = ms_per_img_and_fps(t_sum_t, n_imgs_t)
        safe_record(common, "torch", args.csv, avg=args.avg, accuracy=acc_t, precision=p_t, recall=r_t, f1=f_t, ms=ms_t, fps=fps_t, timed_images=n_imgs_t)
        print(f"Torch: acc={acc_t:.4f} | P={p_t:.4f} R={r_t:.4f} F1={f_t:.4f} ({args.avg}) | {ms_t} ms/img | FPS={fps_t}")
    except Exception as e:
        safe_record(common, "torch", args.csv, avg=args.avg, error_msg=f"{type(e).__name__}: {e}")
        print(f"[ERR] Torch failed: {type(e).__name__}: {e}")

    if args.skip_onnx:
        safe_record(common, "onnxruntime", args.csv, avg=args.avg, error_msg="SKIPPED")
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
            acc_o, t_sum_o, n_imgs_o, cm_o = eval_onnx(sess, loader, args.warmup_batches, max_batches, args.device, args.num_classes)
            p_o, r_o, f_o = prf_from_confusion(cm_o, args.avg)
            ms_o, fps_o = ms_per_img_and_fps(t_sum_o, n_imgs_o)
            safe_record(common, "onnxruntime_" + ("cuda" if "CUDAExecutionProvider" in used else "cpu"),
                        args.csv, avg=args.avg, accuracy=acc_o, precision=p_o, recall=r_o, f1=f_o, ms=ms_o, fps=fps_o, timed_images=n_imgs_o,
                        extra={"ort_used": "|".join(used)})
            print(f"ORT({used[0]}): acc={acc_o:.4f} | P={p_o:.4f} R={r_o:.4f} F1={f_o:.4f} ({args.avg}) | {ms_o} ms/img | FPS={fps_o}")
        except Exception as e:
            safe_record(common, "onnxruntime", args.csv, avg=args.avg, error_msg=f"{type(e).__name__}: {e}")
            print(f"[ERR] ORT failed: {type(e).__name__}: {e}")

    if args.skip_trt:
        safe_record(common, "tensorrt_engine", args.csv, avg=args.avg, error_msg="SKIPPED")
    else:
        try:
            acc_e, t_sum_e, n_imgs_e, cm_e = eval_trt(args.engine, loader, args.warmup_batches, max_batches, args.device, args.num_classes)
            p_e, r_e, f_e = prf_from_confusion(cm_e, args.avg)
            ms_e, fps_e = ms_per_img_and_fps(t_sum_e, n_imgs_e)
            safe_record(common, "tensorrt_engine", args.csv, avg=args.avg, accuracy=acc_e, precision=p_e, recall=r_e, f1=f_e, ms=ms_e, fps=fps_e, timed_images=n_imgs_e,
                        extra={"trt_backend": "ctypes-cudart"})
            print(f"TRT(ctypes-cudart): acc={acc_e:.4f} | P={p_e:.4f} R={r_e:.4f} F1={f_e:.4f} ({args.avg}) | {ms_e} ms/img | FPS={fps_e}")
        except Exception as e:
            safe_record(common, "tensorrt_engine", args.csv, avg=args.avg, error_msg=f"{type(e).__name__}: {e}")
            print(f"[ERR] TRT failed: {type(e).__name__}: {e}")

    print(f"[OK] CSV appended: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()