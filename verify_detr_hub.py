import torch
import sys
import os
import glob

def verify_detr_hub():
    print("Verifying DETR Hub components...")
    
    # 1. Load model to ensure repo is downloaded
    try:
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False, num_classes=91)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Find hub directory
    hub_dir = torch.hub.get_dir()
    repo_dir = glob.glob(os.path.join(hub_dir, 'facebookresearch_detr_*'))[0]
    print(f"Repo dir: {repo_dir}")
    
    # 3. Add to sys.path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        
    # 4. Import Criterion and Matcher
    try:
        from models.matcher import build_matcher
        from models.detr import SetCriterion
        print("✅ Imported SetCriterion and build_matcher.")
    except ImportError as e:
        print(f"❌ Failed to import components: {e}")
        return

    # 5. Instantiate Criterion
    class Args:
        set_cost_class = 1
        set_cost_bbox = 5
        set_cost_giou = 2
        masks = False
        dice_loss_coef = 1
        bbox_loss_coef = 5
        giou_loss_coef = 2
        eos_coef = 0.1
        
    args = Args()
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict.update({'loss_giou': args.giou_loss_coef})
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(91, matcher, weight_dict, args.eos_coef, losses)
    print("✅ Criterion instantiated.")

    # 6. Dummy Forward Pass
    # We need util.misc.nested_tensor_from_tensor_list
    from util.misc import nested_tensor_from_tensor_list
    
    dummy_image = torch.randn(3, 800, 800)
    images = [dummy_image, dummy_image]
    samples = nested_tensor_from_tensor_list(images)
    
    dummy_target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32), # cx, cy, w, h (normalized)
        "labels": torch.tensor([1], dtype=torch.int64)
    }
    targets = [dummy_target, dummy_target]
    
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    print(f"✅ Loss calculated: {loss_dict.keys()}")

if __name__ == "__main__":
    verify_detr_hub()
