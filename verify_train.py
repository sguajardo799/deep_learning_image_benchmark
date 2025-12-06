import torch
from src.models import create_model
from src.train import train_model
from torch.utils.data import DataLoader

def verify_train_loop():
    print("Verifying Training Loop with DETR...")
    
    # 1. Create Model
    model = create_model(task="detection", model_name="detr", num_classes=91)
    
    # 2. Create Dummy DataLoader
    # We need a custom collate_fn to handle list of images/targets
    def collate_fn(batch):
        return tuple(zip(*batch))

    dummy_image = torch.randn(3, 224, 224) # DETR usually takes larger images but this should work
    dummy_target = {
        "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64)
    }
    dataset = [(dummy_image, dummy_target)] * 4
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # 3. Run Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model, history = train_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            device=device,
            task="detection",
            epochs=0,
            patience=1
        )
        print("✅ Training loop completed successfully.")
        print(f"History: {history}")
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_train_loop()
