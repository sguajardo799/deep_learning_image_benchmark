import torch
import torchvision
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

try:
    from torchvision.models.detection import detr_resnet50
    print("✅ detr_resnet50 found in torchvision.models.detection")
except ImportError:
    print("❌ detr_resnet50 NOT found in torchvision.models.detection")
    import torchvision.models.detection as d
    print(f"Available in detection: {dir(d)}")

try:
    import scipy
    print(f"✅ Scipy version: {scipy.__version__}")
except ImportError:
    print("❌ Scipy NOT installed")
