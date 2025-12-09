import argparse
import torch
import time
import numpy as np
from src.models import create_model
from src.data import load_data
from src.inference import PyTorchSession, OnnxSession, TensorRTSession

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model performance (PyTorch vs ONNX vs TensorRT)")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "detection"], help="Task type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--engine", type=str, default=None, help="Path to TensorRT engine")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    return parser.parse_args()

def evaluate_session(session, loader, task, limit=None):
    latencies = []
    correct = 0
    total = 0
    
    print(f"Starting evaluation with {session.__class__.__name__}...")
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(loader):
        if i >= 10: break
        if task == "classification":
            inputs, _ = batch
        else:
            inputs, _ = batch # Detection batch unpacking might differ slightly based on collate
        
        try:
            session.predict(inputs)
        except Exception as e:
            print(f"Warmup failed: {e}")
            break

    print("Running evaluation...")
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
            
        if task == "classification":
            inputs, labels = batch
            # Unpack if needed (handled by session or here)
            # PyTorch DataLoader yields stacked tensors for classification
        elif task == "detection":
            inputs, targets = batch
            # Detection inputs are list of tensors usually
            
        start_time = time.time()
        try:
            outputs = session.predict(inputs)
        except Exception as e:
            print(f"Inference failed at batch {i}: {e}")
            continue
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000) # ms
        
        if task == "classification":
            # Calculate accuracy
            if isinstance(outputs, torch.Tensor):
                preds = torch.argmax(outputs, dim=1).cpu()
            elif isinstance(outputs, (list, np.ndarray)):
                # ONNX/TRT output
                if isinstance(outputs, list): outputs = outputs[0]
                preds = np.argmax(outputs, axis=1)
                preds = torch.tensor(preds)
            
            labels = labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    avg_latency = np.mean(latencies) if latencies else 0
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return {
        "avg_latency_ms": avg_latency,
        "throughput_fps": throughput,
        "accuracy": accuracy if task == "classification" else "N/A"
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    print(f"Loading dataset {args.dataset}...")
    loader, _ = load_data(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        task=args.task
    )
    
    results = {}
    
    # 1. PyTorch Evaluation
    if args.checkpoint:
        print("Loading PyTorch model...")
        model = create_model(task=args.task, model_name=args.model, num_classes=args.num_classes)
        try:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Fix keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            
            session = PyTorchSession(model, device)
            results["PyTorch"] = evaluate_session(session, loader, args.task, args.limit)
        except Exception as e:
            print(f"Failed to load/evaluate PyTorch model: {e}")

    # 2. ONNX Evaluation
    if args.onnx:
        print("Loading ONNX model...")
        try:
            session = OnnxSession(args.onnx)
            results["ONNX"] = evaluate_session(session, loader, args.task, args.limit)
        except Exception as e:
            print(f"Failed to load/evaluate ONNX model: {e}")

    # 3. TensorRT Evaluation
    if args.engine:
        print("Loading TensorRT engine...")
        try:
            session = TensorRTSession(args.engine)
            results["TensorRT"] = evaluate_session(session, loader, args.task, args.limit)
        except Exception as e:
            print(f"Failed to load/evaluate TensorRT engine: {e}")

    # Print Results
    print("\n" + "="*60)
    print(f"{'Format':<15} | {'Latency (ms)':<15} | {'FPS':<10} | {'Accuracy':<10}")
    print("-" * 60)
    for name, res in results.items():
        acc_str = f"{res['accuracy']:.4f}" if isinstance(res['accuracy'], float) else res['accuracy']
        print(f"{name:<15} | {res['avg_latency_ms']:<15.2f} | {res['throughput_fps']:<10.2f} | {acc_str:<10}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
