import torch
from src.models import create_model

def verify_detr():
    print("Verifying DETR implementation...")
    
    # 1. Instantiate model
    try:
        model = create_model(task="detection", model_name="detr", num_classes=91)
        print("✅ Model instantiated successfully.")
    except Exception as e:
        print(f"❌ Failed to instantiate model: {e}")
        return

    # 2. Create dummy input
    # DETR expects a list of tensors
    dummy_image = torch.randn(3, 800, 800)
    images = [dummy_image, dummy_image] # Batch of 2
    
    # DETR expects targets to be a list of dicts
    dummy_target = {
        "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64)
    }
    targets = [dummy_target, dummy_target]

    # 3. Test Training Mode (Forward pass returns losses)
    model.train()
    try:
        losses = model(images, targets)
        if isinstance(losses, dict) and "loss_ce" in losses:
            print(f"✅ Training forward pass successful. Losses keys: {list(losses.keys())}")
        else:
            print(f"❌ Training forward pass returned unexpected output: {type(losses)}")
    except Exception as e:
        print(f"❌ Training forward pass failed: {e}")

    # 4. Test Eval Mode (Forward pass returns detections)
    model.eval()
    try:
        with torch.no_grad():
            detections = model(images)
            if isinstance(detections, list) and len(detections) == 2 and "boxes" in detections[0]:
                print("✅ Inference forward pass successful (List format).")
            elif isinstance(detections, dict) and "pred_logits" in detections:
                print("✅ Inference forward pass successful (Dict format - DETR).")
            else:
                print(f"❌ Inference forward pass returned unexpected output: {type(detections)}")
    except Exception as e:
        print(f"❌ Inference forward pass failed: {e}")

if __name__ == "__main__":
    verify_detr()
