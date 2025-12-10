# Repositorio de benchmark para tareas de clasificación y detección de objetos

## Uso
+ Compilar imagen ``` docker build -t [TAG] . ```
+ Correr contenedor con volumen para datasets 
``` 
mkdir -p ./data
docker run --rm -v $(pwd)/data:/app/data [TAG] --mode download_data --dataset_name [DATASET_NAME]
```
+ Entrenar modelo seleccionado en ``` config/default.yaml ```
```
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/config/default.yaml:/app/config/default.yaml:ro [TAG] --mode train_model  
```


+ Cuantizar modelo (ONNX/TensorRT)
```
docker run --rm --gpus all -v $(pwd)/output:/app/output --entrypoint python [TAG] quantize.py --model [MODEL] --task [TASK] --output_dir /app/output/quantized
```
Ejemplo:
```
docker run --rm --gpus all -v $(pwd)/output:/app/output --entrypoint python [TAG] quantize.py --model resnet18 --task classification
```
Ejemplo con checkpoint:
```
docker run --rm --gpus all -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] quantize.py --model resnet18 --task classification --checkpoint /app/checkpoints/model.pth
```

+ Evaluar rendimiento (PyTorch/ONNX/TensorRT)
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model [MODEL] --task [TASK] --dataset [DATASET] --split [SPLIT] --checkpoint /app/checkpoints/[CHECKPOINT] --onnx /app/output/quantized/[ONNX_MODEL] --engine /app/output/quantized/[ENGINE]
```
Ejemplo:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine
```
Ejemplo con limit:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10
```
Ejemplo con limit y batch_size:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16
```
Ejemplo con limit, batch_size y num_workers:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4
```
Ejemplo con limit, batch_size, num_workers y img_size:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224
```
Ejemplo con limit, batch_size, num_workers, img_size y num_classes:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes y device:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10 --device cuda
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes, device y num_repeats:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10 --device cuda --num_repeats 10
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes, device, num_repeats y warmup:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10 --device cuda --num_repeats 10 --warmup 5
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes, device, num_repeats, warmup y num_warmup:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10 --device cuda --num_repeats 10 --warmup 5 --num_warmup 5
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes, device, num_repeats, warmup, num_warmup y num_warmup_steps:
```
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] evaluate.py --model resnet18 --task classification --dataset cifar10 --split test --checkpoint /app/checkpoints/resnet18_best.pth --onnx /app/output/quantized/resnet18_classification.onnx --engine /app/output/quantized/resnet18_classification.engine --limit 10 --batch_size 16 --num_workers 4 --img_size 224 --num_classes 10 --device cuda --num_repeats 10 --warmup 5 --num_warmup 5 --num_warmup_steps 5
```
Ejemplo con limit, batch_size, num_workers, img_size, num_classes, device, num_repeats, warmup, num_warmup y num_warmup_steps y num_warmup_steps:
```
    --dataset cifar10 \
    --split test \
    --checkpoint /app/checkpoints/resnet18_best.pth \
    --onnx /app/output/quantized/resnet18_classification.onnx \
    --engine /app/output/quantized/resnet18_classification.engine
```
