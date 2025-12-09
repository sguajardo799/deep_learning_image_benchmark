import argparse
import torch
import os
from src.models import create_model
from src.quantization import export_to_onnx, quantize_onnx_model, build_tensorrt_engine

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize models using ONNX and TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., resnet18, detr)")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "detection"], help="Task type")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--output_dir", type=str, default="output/quantized", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dummy input")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 for TensorRT")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model {args.model} for {args.task}...")
    model = create_model(task=args.task, model_name=args.model, num_classes=args.num_classes)
    
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        try:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            # Handle different checkpoint formats (e.g., state_dict only vs full checkpoint dict)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Handle potential key mismatches (e.g., 'module.' prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return

    model.eval()
    
    # 2. Prepare Dummy Input
    if args.task == "classification":
        # Standard image input: [B, 3, 224, 224]
        dummy_input = torch.randn(args.batch_size, 3, 224, 224)
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    elif args.task == "detection":
        # DETR expects list of tensors, but ONNX export usually requires a single tensor or tuple
        # For DETR export, we usually export the model that takes a NestedTensor or just the tensor if modified
        # However, standard torchvision DETR forward takes list[Tensor].
        # We might need to wrap it or use a specific export configuration.
        # For simplicity, let's assume standard image tensor input if the model supports it, 
        # or we might need to adjust the model wrapper for ONNX export.
        
        # NOTE: DETR export is complex. Torchvision's DETR might need specific handling.
        # Let's try with standard tensor and see if the wrapper handles it or if we need to unwrap.
        
        if args.model == "detr":
            # The DetrWrapper in src/models.py takes list[Tensor].
            # ONNX export doesn't support list[Tensor] as input directly in the signature usually.
            # We might need to export the inner model or a modified wrapper.
            print("Warning: DETR export can be complex. Attempting with standard tensor input...")
            dummy_input = torch.randn(args.batch_size, 3, 800, 800)
            
            # If using DetrWrapper, we might need to adjust it to accept a tensor for export
            # Or we can try to export the inner model if we know how to handle inputs.
            # For now, let's try passing the tensor. If DetrWrapper expects list, this might fail.
            # Let's wrap it in a tuple if needed, but torch.onnx.export expects args tuple.
            
            # Actually, let's check src/models.py again. DetrWrapper.forward takes (images, targets=None).
            # images is list of tensors.
            # We can create a wrapper for ONNX that takes a tensor and converts it to list.
            
            class OnnxWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    # x is [B, 3, H, W]
                    # Convert to list of [3, H, W]
                    images = [x[i] for i in range(x.shape[0])]
                    return self.model(images)
            
            model = OnnxWrapper(model)
            input_names = ['input']
            output_names = ['pred_logits', 'pred_boxes'] # DETR outputs dict
            dynamic_axes = {'input': {0: 'batch_size'}}
            
        else:
             dummy_input = torch.randn(args.batch_size, 3, 300, 300) # SSD/FasterRCNN size guess
             input_names = ['input']
             output_names = ['boxes', 'scores', 'labels']
             dynamic_axes = {'input': {0: 'batch_size'}}

    # 3. Export to ONNX
    onnx_filename = f"{args.model}_{args.task}.onnx"
    onnx_path = os.path.join(args.output_dir, onnx_filename)
    
    try:
        export_to_onnx(
            model, 
            dummy_input, 
            onnx_path, 
            input_names=input_names, 
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
    except Exception as e:
        print(f"Skipping further steps due to export failure: {e}")
        return

    # 4. Quantize ONNX
    quantized_filename = f"{args.model}_{args.task}_quantized.onnx"
    quantized_path = os.path.join(args.output_dir, quantized_filename)
    quantize_onnx_model(onnx_path, quantized_path)
    
    # 5. Build TensorRT Engine
    engine_filename = f"{args.model}_{args.task}.engine"
    engine_path = os.path.join(args.output_dir, engine_filename)
    build_tensorrt_engine(onnx_path, engine_path, fp16=args.fp16)

    print("\n------------------------------------------------")
    print(f"Processing complete for {args.model}")
    print(f"ONNX Model: {onnx_path}")
    print(f"Quantized ONNX: {quantized_path}")
    print(f"TensorRT Engine: {engine_path}")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    main()
